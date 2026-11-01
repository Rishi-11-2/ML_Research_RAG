"""Migrate document chunks to a Hypencoder Qdrant collection with nearest-neighbor graphs.

This script implements streaming, resumable migration for Hypencoder:
1. Validates or creates the target Qdrant collection with 768-dim Euclidean distance.
2. Quickly collects existing target point IDs to skip already migrated points immediately.
3. Streams unmigrated points from the source collection in configurable chunk sizes (default 2500).
4. Encodes passage chunks using Hypencoder's passage encoder (jfkback/hypencoder.6_layer).
5. Constructs a k-nearest neighbor graph (default 100 neighbors) connecting each chunk to
   the corpus pool using Euclidean distance in Hypencoder embedding space.
6. Upserts points with 768-dim vectors and payloads (including the 'neighbors' list) to Qdrant.
7. Leaves the source collection completely untouched.
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
DEFAULT_SCROLL_PAGE = 500
DEFAULT_EMBED_BATCH = 16
DEFAULT_CHUNK_SIZE = 2500
MAX_POOL_SIZE = 25000


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
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of chunks to embed, graph-link, and upsert per streaming batch (default: 2500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of points to migrate in this run (for testing/partial runs)",
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
    chunk_embeddings: torch.Tensor,
    chunk_ids: List[Any],
    pool_embeddings: torch.Tensor,
    pool_ids: List[Any],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    batch_size: int = 256,
) -> Dict[Any, List[Any]]:
    """Compute k nearest neighbors in Euclidean space for chunk points against the pool.

    Args:
        chunk_embeddings: Tensor of shape (M, 768) for points being migrated.
        chunk_ids: List of point IDs of length M.
        pool_embeddings: Tensor of shape (P, 768) representing search pool.
        pool_ids: List of point IDs of length P in the search pool.
        k_neighbors: Number of nearest neighbors to find for each point.
        batch_size: Batch size for pairwise distance calculation.

    Returns:
        Dict mapping each chunk point_id to a list of neighbor point_ids.
    """
    n_chunk = chunk_embeddings.shape[0]
    n_pool = pool_embeddings.shape[0]
    if n_chunk == 0 or n_pool <= 1:
        return {pid: [] for pid in chunk_ids}

    actual_k = min(k_neighbors, n_pool - 1)
    graph: Dict[Any, List[Any]] = {}

    for start_idx in range(0, n_chunk, batch_size):
        end_idx = min(start_idx + batch_size, n_chunk)
        batch = chunk_embeddings[start_idx:end_idx]

        # Euclidean distances: shape (batch_len, n_pool)
        dists = torch.cdist(batch, pool_embeddings, p=2)

        # Look for top (actual_k + 1) smallest distances (to account for self)
        neg_dists = -dists
        _, top_indices = torch.topk(neg_dists, k=min(actual_k + 1, n_pool), dim=1)

        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            pid = chunk_ids[global_i]
            nbr_indices = top_indices[local_i].tolist()
            # Filter out self
            neighbors = [pool_ids[idx] for idx in nbr_indices if pool_ids[idx] != pid][:actual_k]
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

    total_source_points = client.count(source_name, exact=True).count
    print(f"Source collection: {source_name!r} ({total_source_points} points)", flush=True)
    if total_source_points == 0:
        raise RuntimeError("Source collection is empty; aborting.")

    # Target collection check & creation
    target_exists = client.collection_exists(target_name)
    if target_exists and not args.resume:
        existing_count = client.count(target_name, exact=True).count
        if existing_count >= total_source_points:
            print(
                f"Target collection {target_name!r} already contains all {existing_count} points. Migration complete.",
                flush=True,
            )
            return 0
        print(
            f"Target collection {target_name!r} already exists with {existing_count} points. Use --resume to continue.",
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

    # Collect existing target IDs to skip them instantly
    existing_ids: Set[Any] = set()
    pool_ids: List[Any] = []
    pool_embeddings: Optional[torch.Tensor] = None

    if target_exists or args.resume:
        print(f"Scanning existing points in {target_name!r}...", flush=True)
        res_offset = None
        seed_vectors: List[List[float]] = []

        while True:
            fetch_vectors = len(pool_ids) < 10000
            t_points, next_res_offset = client.scroll(
                collection_name=target_name,
                limit=1000,
                offset=res_offset,
                with_payload=False,
                with_vectors=fetch_vectors,
            )
            for tp in t_points:
                existing_ids.add(tp.id)
                if fetch_vectors and tp.vector is not None and len(tp.vector) == TARGET_DIMENSION:
                    pool_ids.append(tp.id)
                    seed_vectors.append(tp.vector)

            if next_res_offset is None:
                break
            res_offset = next_res_offset

        print(
            f"Found {len(existing_ids)} points already migrated in {target_name!r}.",
            flush=True,
        )
        if seed_vectors:
            pool_embeddings = torch.tensor(seed_vectors, dtype=torch.float32)
            print(
                f"Seeded neighbor pool with {pool_embeddings.shape[0]} existing vectors for cross-batch connectivity.",
                flush=True,
            )

        if len(existing_ids) >= total_source_points:
            print(
                f"All {total_source_points} points are already migrated to {target_name!r}. Nothing to do.",
                flush=True,
            )
            return 0

    # Load Hypencoder model
    device = args.device or get_default_device()
    print(f"Loading Hypencoder ({args.model_name}) on device: {device}...", flush=True)
    model, tokenizer = load_hypencoder(args.model_name, device=device)
    passage_encoder = model.passage_encoder.to(device)
    passage_encoder.eval()

    # Stream points from source
    print(
        f"Streaming source chunks (chunk_size={args.chunk_size}, batch_size={args.batch_size}, limit={args.limit})...",
        flush=True,
    )
    offset = None
    migrated_this_run = 0
    current_chunk: List[Tuple[Any, Dict[str, Any], str]] = []

    def flush_chunk(chunk: List[Tuple[Any, Dict[str, Any], str]]) -> int:
        nonlocal pool_ids, pool_embeddings
        if not chunk:
            return 0

        chunk_ids = [item[0] for item in chunk]
        chunk_payloads = [item[1] for item in chunk]
        chunk_texts = [item[2] for item in chunk]

        # 1. Encode passages
        chunk_embeddings_list: List[torch.Tensor] = []
        for b_start in range(0, len(chunk_texts), args.batch_size):
            b_texts = chunk_texts[b_start : b_start + args.batch_size]
            inputs = tokenizer(
                b_texts,
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
                chunk_embeddings_list.append(outputs.representation.cpu())

        chunk_tensor = torch.cat(chunk_embeddings_list, dim=0)

        # 2. Combine with neighbor pool for graph computation
        if pool_embeddings is not None and pool_embeddings.shape[0] > 0:
            combined_embeddings = torch.cat([pool_embeddings, chunk_tensor], dim=0)
            combined_ids = pool_ids + chunk_ids
        else:
            combined_embeddings = chunk_tensor
            combined_ids = chunk_ids

        neighbor_graph = compute_neighbor_graph(
            chunk_embeddings=chunk_tensor,
            chunk_ids=chunk_ids,
            pool_embeddings=combined_embeddings,
            pool_ids=combined_ids,
            k_neighbors=args.k_neighbors,
            batch_size=256,
        )

        # 3. Upsert points and neighbor lists to Qdrant
        upsert_batch_size = 64
        for u_idx in range(0, len(chunk_ids), upsert_batch_size):
            u_slice = slice(u_idx, u_idx + upsert_batch_size)
            pts_to_upsert = []
            for pid, pld, vec in zip(
                chunk_ids[u_slice],
                chunk_payloads[u_slice],
                chunk_tensor[u_slice],
            ):
                merged_payload = dict(pld)
                merged_payload["neighbors"] = neighbor_graph.get(pid, [])
                pts_to_upsert.append(
                    models.PointStruct(
                        id=pid,
                        vector=vec.tolist(),
                        payload=merged_payload,
                    )
                )

            client.upsert(
                collection_name=target_name,
                points=pts_to_upsert,
                wait=True,
            )

        # 4. Update running state and bounded neighbor pool
        existing_ids.update(chunk_ids)
        pool_ids = combined_ids[-MAX_POOL_SIZE:]
        pool_embeddings = combined_embeddings[-MAX_POOL_SIZE:]

        total_target_now = client.count(target_name, exact=True).count
        pct = (total_target_now / total_source_points) * 100
        print(
            f"Upserted chunk of {len(chunk_ids)} points. Total in {target_name!r}: {total_target_now}/{total_source_points} ({pct:.1f}%)",
            flush=True,
        )
        return len(chunk_ids)

    # Main scroll loop
    while True:
        points, next_offset = client.scroll(
            collection_name=source_name,
            limit=DEFAULT_SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            if p.id in existing_ids:
                continue

            payload = p.payload or {}
            text = payload.get("text", "")
            if isinstance(text, str) and text.strip():
                current_chunk.append((p.id, payload, text.strip()))

                if len(current_chunk) >= args.chunk_size:
                    chunk_migrated = flush_chunk(current_chunk)
                    migrated_this_run += chunk_migrated
                    current_chunk = []

                    if args.limit is not None and migrated_this_run >= args.limit:
                        break

        if args.limit is not None and migrated_this_run >= args.limit:
            print(f"Reached requested limit of {args.limit} points for this run.", flush=True)
            break

        if next_offset is None:
            break
        offset = next_offset

    # Flush remaining points in final chunk
    if current_chunk and (args.limit is None or migrated_this_run < args.limit):
        remaining_chunk = current_chunk
        if args.limit is not None:
            remaining_chunk = remaining_chunk[: args.limit - migrated_this_run]
        chunk_migrated = flush_chunk(remaining_chunk)
        migrated_this_run += chunk_migrated

    final_count = client.count(target_name, exact=True).count
    print(
        f"Migration run complete! Migrated {migrated_this_run} points this run. Target collection {target_name!r} now has {final_count}/{total_source_points} points.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Migration error: {exc}", file=sys.stderr)
        sys.exit(1)
