"""Migrate document chunks to a Hypencoder Qdrant collection with a 100-neighbor graph.

This script implements Section 4 of the Hypencoder integration strategy:
1. Creates a separate Qdrant collection with 768-dim Euclidean distance.
2. Encodes passage chunks using Hypencoder's passage encoder (jfkback/hypencoder.6_layer).
3. Constructs a 100-nearest neighbor graph for the corpus using L2 distance in embedding space.
4. Stores vectors and payloads (including the 'neighbors' list of neighbor IDs) in Qdrant.
5. Retains the original source collection completely untouched.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient, models
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_retriever import load_hypencoder, DEFAULT_MODEL_NAME, get_default_device

TARGET_DIMENSION = 768
DEFAULT_K_NEIGHBORS = 100
DEFAULT_SCROLL_PAGE = 256
DEFAULT_EMBED_BATCH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-collection",
        default=None,
        help="Source Qdrant collection (defaults to QDRANT_COLLECTION env or 'document_chunks')",
    )
    parser.add_argument(
        "--target-collection",
        default=None,
        help="Target Qdrant collection (defaults to QDRANT_COLLECTION_HYPENCODER or '<source>_hypencoder')",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Hypencoder model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help="Number of nearest neighbors to compute per document chunk (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH,
        help="Batch size for passage encoding (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of points to migrate (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted migration",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use for encoding ('cpu', 'cuda', 'mps')",
    )
    return parser.parse_args()


def compute_neighbor_graph(
    embeddings: torch.Tensor,
    point_ids: List[Any],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    batch_size: int = 256,
) -> Dict[Any, List[Any]]:
    """Compute k nearest neighbors in Euclidean space for every embedding.

    Args:
        embeddings: Tensor of shape (N, 768).
        point_ids: List of point IDs of length N.
        k_neighbors: Number of nearest neighbors to find for each point.
        batch_size: Batch size for pairwise distance calculation.

    Returns:
        Dict mapping each point_id to a list of neighbor point_ids.
    """
    n_points = embeddings.shape[0]
    if n_points <= 1:
        return {pid: [] for pid in point_ids}

    actual_k = min(k_neighbors, n_points - 1)
    graph: Dict[Any, List[Any]] = {}

    print(f"Building {actual_k}-neighbor graph for {n_points} items...", flush=True)

    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)
        batch = embeddings[start_idx:end_idx]

        # Euclidean distances: shape (batch_len, n_points)
        dists = torch.cdist(batch, embeddings, p=2)

        # We look for top (actual_k + 1) smallest distances (to account for self)
        # Using -dists with topk finds the smallest distances
        neg_dists = -dists
        _, top_indices = torch.topk(neg_dists, k=min(actual_k + 1, n_points), dim=1)

        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            pid = point_ids[global_i]
            nbr_indices = top_indices[local_i].tolist()
            # Filter out self
            neighbors = [point_ids[idx] for idx in nbr_indices if idx != global_i][:actual_k]
            graph[pid] = neighbors

    return graph


def main() -> int:
    args = parse_args()
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL must be set in environment or .env")

    source_name = args.source_collection or os.getenv("QDRANT_COLLECTION", "document_chunks")
    target_name = args.target_collection or os.getenv(
        "QDRANT_COLLECTION_HYPENCODER", f"{source_name}_hypencoder"
    )

    if source_name == target_name:
        raise RuntimeError("Source and target collection names must differ.")

    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key.strip() if api_key else None,
        timeout=60.0,
        prefer_grpc=False,
    )

    if not client.collection_exists(source_name):
        raise RuntimeError(f"Source collection {source_name!r} does not exist.")

    source_info = client.get_collection(source_name)
    total_source_points = client.count(source_name, exact=True).count
    print(
        f"Source collection: {source_name!r} ({total_source_points} points)",
        flush=True,
    )
    if total_source_points == 0:
        raise RuntimeError("Source collection is empty; aborting.")

    # Target collection initialization
    target_exists = client.collection_exists(target_name)
    if target_exists and not args.resume:
        print(
            f"Target collection {target_name!r} already exists. Use --resume to continue or choose a new name.",
            flush=True,
        )
        return 0

    if not target_exists:
        print(
            f"Creating target collection {target_name!r} with dimension {TARGET_DIMENSION} and Distance.EUCLID...",
            flush=True,
        )
        client.create_collection(
            collection_name=target_name,
            vectors_config=models.VectorParams(
                size=TARGET_DIMENSION,
                distance=models.Distance.EUCLID,
            ),
        )

    # Load Hypencoder model
    device = args.device or get_default_device()
    print(f"Loading Hypencoder ({args.model_name}) on device: {device}...", flush=True)
    model, tokenizer = load_hypencoder(args.model_name, device=device)
    passage_encoder = model.passage_encoder.to(device)
    passage_encoder.eval()

    # Read points from source
    print(f"Reading points from source {source_name!r}...", flush=True)
    offset = None
    all_points_data: List[Tuple[Any, Dict[str, Any], str]] = []
    scanned = 0

    max_to_fetch = args.limit if args.limit is not None else total_source_points

    while scanned < max_to_fetch:
        batch_limit = min(DEFAULT_SCROLL_PAGE, max_to_fetch - scanned)
        points, next_offset = client.scroll(
            collection_name=source_name,
            limit=batch_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "")
            if isinstance(text, str) and text.strip():
                all_points_data.append((p.id, payload, text.strip()))
                scanned += 1
                if scanned >= max_to_fetch:
                    break

        if next_offset is None or scanned >= max_to_fetch:
            break
        offset = next_offset

    print(f"Retrieved {len(all_points_data)} valid document chunks.", flush=True)
    if not all_points_data:
        print("No valid text chunks to embed.", flush=True)
        return 0

    # Check already migrated points if resume
    existing_ids: Set[Any] = set()
    if args.resume and client.collection_exists(target_name):
        res_offset = None
        while True:
            t_points, next_res_offset = client.scroll(
                collection_name=target_name,
                limit=DEFAULT_SCROLL_PAGE,
                offset=res_offset,
                with_payload=True,
                with_vectors=True,
            )
            for tp in t_points:
                if (
                    tp.vector is not None
                    and len(tp.vector) == TARGET_DIMENSION
                    and tp.payload
                    and "neighbors" in tp.payload
                ):
                    existing_ids.add(tp.id)
            if next_res_offset is None:
                break
            res_offset = next_res_offset
        print(f"Resume: Found {len(existing_ids)} already migrated valid points.", flush=True)

    # Encode all passage embeddings
    print("Encoding passages with Hypencoder passage encoder...", flush=True)
    all_embeddings_list: List[torch.Tensor] = []
    all_ids: List[Any] = []
    all_payloads: List[Dict[str, Any]] = []

    for i in tqdm(range(0, len(all_points_data), args.batch_size), desc="Passage Encoding"):
        batch = all_points_data[i : i + args.batch_size]
        batch_ids = [item[0] for item in batch]
        batch_payloads = [item[1] for item in batch]
        batch_texts = [item[2] for item in batch]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = passage_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            reps = outputs.representation.cpu()

        for pid, payload, rep in zip(batch_ids, batch_payloads, reps):
            all_ids.append(pid)
            all_payloads.append(payload)
            all_embeddings_list.append(rep)

    corpus_embeddings = torch.stack(all_embeddings_list)
    print(
        f"Generated {corpus_embeddings.shape[0]} embeddings of shape {corpus_embeddings.shape[1:]}.",
        flush=True,
    )

    # Compute 100-neighbor graph for our corpus
    neighbor_graph = compute_neighbor_graph(
        corpus_embeddings,
        all_ids,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size * 4,
    )

    # Upsert into target collection
    print(f"Upserting points and neighbor graph into {target_name!r}...", flush=True)
    upsert_batch_size = 64
    for i in tqdm(range(0, len(all_ids), upsert_batch_size), desc="Upserting to Qdrant"):
        batch_slice = slice(i, i + upsert_batch_size)
        sub_ids = all_ids[batch_slice]
        sub_payloads = all_payloads[batch_slice]
        sub_vectors = corpus_embeddings[batch_slice].numpy()

        points_to_upsert = []
        for pid, payload, vec in zip(sub_ids, sub_payloads, sub_vectors):
            if pid in existing_ids:
                continue

            merged_payload = dict(payload)
            merged_payload["neighbors"] = neighbor_graph.get(pid, [])
            points_to_upsert.append(
                models.PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload=merged_payload,
                )
            )

        if points_to_upsert:
            client.upsert(
                collection_name=target_name,
                points=points_to_upsert,
                wait=True,
            )

    dest_count = client.count(target_name, exact=True).count
    print(
        f"Migration complete! Target collection {target_name!r} now has {dest_count} points with 100-neighbor graphs.",
        flush=True,
    )
    print("Source collection remains untouched.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Migration error: {exc}", file=sys.stderr)
        sys.exit(1)
