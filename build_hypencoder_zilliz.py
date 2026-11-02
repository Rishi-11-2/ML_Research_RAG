"""Build fresh Hypencoder vectors and a nearest-neighbor graph in Zilliz.

Text and stable IDs are read from the original Qdrant document collection. Existing
Hypencoder vectors in Qdrant are never used as migration input.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient
from qdrant_client import QdrantClient
import torch

from hypencoder_retriever import DEFAULT_MODEL_NAME, get_default_device, load_hypencoder

VECTOR_DIMENSION = 768
PAPER_PASSAGE_MAX_LENGTH = 196
GRAPH_BUILD_FIELD = "hypencoder_graph_build_id"
BUILD_ID_FIELD = "hypencoder_build_id"
ENTRY_POINT_FIELD = "hypencoder_entry_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection", default=None)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--graph-degree", type=int, default=100)
    parser.add_argument("--scroll-page-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", default="hypencoder_zilliz_build_report.json")
    return parser.parse_args()


def _milvus_ids_filter(ids: Sequence[str]) -> str:
    return f"id in {json.dumps(list(ids))}"


def _query_by_ids(
    client: MilvusClient,
    collection: str,
    ids: Sequence[str],
    fields: Sequence[str],
) -> List[Dict[str, Any]]:
    if not ids:
        return []
    return client.query(
        collection_name=collection,
        filter=_milvus_ids_filter(ids),
        output_fields=list(fields),
        limit=len(ids),
        consistency_level="Strong",
    )


def _milvus_id(record: Dict[str, Any]) -> Optional[str]:
    record_id = record.get("id")
    if record_id is None:
        record_id = (record.get("entity") or {}).get("id")
    return str(record_id) if record_id is not None else None


def _milvus_row(point: Any, vector: Sequence[float], build_id: str) -> Dict[str, Any]:
    payload = point.payload or {}
    row: Dict[str, Any] = {
        "id": str(point.id),
        "vector": [float(value) for value in vector],
        "text": str(payload.get("text", "")),
        "source": str(payload.get("source") or payload.get("filename") or "unknown"),
        "_stable_id": str(payload.get("_stable_id") or point.id),
        BUILD_ID_FIELD: build_id,
    }
    for key, value in payload.items():
        if key in row or key == "neighbors":
            continue
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            row[key] = value
        else:
            row[key] = str(value)
    return row


def _collection_count(client: MilvusClient, collection: str) -> int:
    stats = client.get_collection_stats(collection_name=collection)
    try:
        return int(stats["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Zilliz did not return a usable row_count for {collection!r}: {stats}") from exc


def _existing_ids_for_build(
    client: MilvusClient,
    collection: str,
    ids: Sequence[str],
    build_id: str,
) -> Set[str]:
    records = _query_by_ids(client, collection, ids, ["id", BUILD_ID_FIELD])
    existing_ids: Set[str] = set()
    for record in records:
        record_id = _milvus_id(record)
        if record_id is None:
            continue
        if record.get(BUILD_ID_FIELD) != build_id:
            raise RuntimeError(
                f"Target point {record_id!r} belongs to a different build; refusing to overwrite it. "
                "Use a new staging collection or resume the build that created this collection."
            )
        existing_ids.add(record_id)
    return existing_ids


def _encode_passages(
    texts: Sequence[str],
    encoder: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=PAPER_PASSAGE_MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            output = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).representation
        embeddings.extend(output.detach().to(torch.float32).cpu().tolist())
    return embeddings


def _ensure_collection(client: MilvusClient, collection: str, resume: bool) -> None:
    if client.has_collection(collection):
        if not resume:
            raise RuntimeError(
                f"Target Zilliz collection {collection!r} already exists. Choose a fresh staging name, "
                "or use --resume only for a previous run of this workflow."
            )
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )


def _embed_source_documents(
    qdrant: QdrantClient,
    milvus: MilvusClient,
    source_collection: str,
    target_collection: str,
    source_count: int,
    build_id: str,
    encoder: Any,
    tokenizer: Any,
    device: str,
    page_size: int,
    embedding_batch_size: int,
) -> int:
    offset = None
    scanned = 0
    newly_embedded = 0
    page_number = 0

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=source_collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        page_number += 1
        ids = [str(point.id) for point in points]
        completed_ids = _existing_ids_for_build(milvus, target_collection, ids, build_id)
        pending = []
        for point in points:
            if str(point.id) in completed_ids:
                continue
            text = (point.payload or {}).get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Source point {point.id!r} has no non-empty text payload; refusing partial build.")
            pending.append((point, text.strip()))

        for start in range(0, len(pending), embedding_batch_size):
            batch = pending[start : start + embedding_batch_size]
            vectors = _encode_passages(
                [text for _, text in batch], encoder, tokenizer, device, embedding_batch_size
            )
            rows = [_milvus_row(point, vector, build_id) for (point, _), vector in zip(batch, vectors)]
            for row in rows:
                row["neighbors"] = []
                row[GRAPH_BUILD_FIELD] = "pending"
                row[ENTRY_POINT_FIELD] = False
            milvus.upsert(collection_name=target_collection, data=rows)
            newly_embedded += len(rows)

        scanned += len(points)
        print(
            f"Embedded source pages: {scanned}/{source_count}; "
            f"new vectors this run: {newly_embedded}",
            flush=True,
        )
        if next_offset is None:
            break
        offset = next_offset

    if scanned != source_count:
        raise RuntimeError(f"Qdrant scroll returned {scanned} points, but exact count is {source_count}.")
    milvus.flush(collection_name=target_collection)
    target_count = _collection_count(milvus, target_collection)
    if target_count != source_count:
        raise RuntimeError(
            f"Fresh Hypencoder vector count mismatch: Qdrant source={source_count}, Zilliz target={target_count}."
        )
    return newly_embedded


def _build_graph(
    qdrant: QdrantClient,
    milvus: MilvusClient,
    source_collection: str,
    target_collection: str,
    source_count: int,
    build_id: str,
    graph_degree: int,
    page_size: int,
    search_batch_size: int,
) -> Dict[str, Any]:
    milvus.load_collection(collection_name=target_collection)
    offset = None
    processed = 0
    edge_count = 0
    min_degree: Optional[int] = None
    max_degree = 0
    expected_degree = min(graph_degree, max(source_count - 1, 0))
    page_number = 0
    resume_skipped_points = 0

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=source_collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        page_number += 1
        id_list = [str(point.id) for point in points]
        graph_rows = _query_by_ids(
            milvus,
            target_collection,
            id_list,
            ["id", "neighbors", GRAPH_BUILD_FIELD],
        )
        graph_by_id = {
            record_id: record
            for record in graph_rows
            if (record_id := _milvus_id(record)) is not None
        }
        already_built: Set[str] = set()
        for point in points:
            point_id = str(point.id)
            row = graph_by_id.get(point_id)
            neighbors = row.get("neighbors") if row else None
            if (
                row is not None
                and row.get(GRAPH_BUILD_FIELD) == build_id
                and isinstance(neighbors, list)
                and len(neighbors) == expected_degree
                and point_id not in {str(value) for value in neighbors}
            ):
                already_built.add(point_id)
                processed += 1
                resume_skipped_points += 1
                edge_count += len(neighbors)
                min_degree = len(neighbors) if min_degree is None else min(min_degree, len(neighbors))
                max_degree = max(max_degree, len(neighbors))

        pending_points = [point for point in points if str(point.id) not in already_built]
        stored = _query_by_ids(
            milvus,
            target_collection,
            [str(point.id) for point in pending_points],
            ["id", "vector"],
        )
        vectors_by_id = {
            record_id: record.get("vector")
            for record in stored
            if (record_id := _milvus_id(record)) is not None
        }
        pending_ids = [str(point.id) for point in pending_points]
        if any(point_id not in vectors_by_id or vectors_by_id[point_id] is None for point_id in pending_ids):
            raise RuntimeError(f"Zilliz is missing vectors for points in graph batch {page_number}.")

        for start in range(0, len(pending_points), search_batch_size):
            batch_points = pending_points[start : start + search_batch_size]
            batch_ids = [str(point.id) for point in batch_points]
            batch_vectors = [vectors_by_id[point_id] for point_id in batch_ids]
            search_results = milvus.search(
                collection_name=target_collection,
                anns_field="vector",
                data=batch_vectors,
                limit=min(expected_degree + 1, source_count),
                output_fields=["id"],
                search_params={"metric_type": "L2", "params": {}},
            )
            if len(search_results) != len(batch_points):
                raise RuntimeError(
                    f"Zilliz returned {len(search_results)} search rows for {len(batch_points)} queries."
                )

            rows = []
            for batch_offset, (point, vector, hits) in enumerate(zip(batch_points, batch_vectors, search_results)):
                point_id = str(point.id)
                neighbors: List[str] = []
                seen: Set[str] = set()
                for hit in hits:
                    neighbor_id = _milvus_id(hit)
                    if neighbor_id is None or neighbor_id == point_id or neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)
                    neighbors.append(neighbor_id)
                    if len(neighbors) == expected_degree:
                        break
                if len(neighbors) != expected_degree:
                    raise RuntimeError(
                        f"Point {point_id!r} received {len(neighbors)} neighbors; expected {expected_degree}."
                    )

                row = _milvus_row(point, vector, build_id)
                row["neighbors"] = neighbors
                row[GRAPH_BUILD_FIELD] = build_id
                # Deterministic, spread-out seed set for global graph traversal.
                digest = hashlib.sha256(point_id.encode("utf-8")).digest()
                first_source_point = page_number == 1 and start + batch_offset == 0
                row[ENTRY_POINT_FIELD] = first_source_point or int.from_bytes(digest[:4], "big") % 100 == 0
                rows.append(row)
                degree = len(neighbors)
                min_degree = degree if min_degree is None else min(min_degree, degree)
                max_degree = max(max_degree, degree)
                edge_count += degree

            milvus.upsert(collection_name=target_collection, data=rows)
            processed += len(rows)

        print(
            f"Graph batches: {processed}/{source_count} points; "
            f"{edge_count} edges written",
            flush=True,
        )
        if next_offset is None:
            break
        offset = next_offset

    if processed != source_count:
        raise RuntimeError(f"Graph pass covered {processed}/{source_count} source points.")
    milvus.flush(collection_name=target_collection)
    target_count = _collection_count(milvus, target_collection)
    if target_count != source_count:
        raise RuntimeError(f"Zilliz row count changed during graph build: expected {source_count}, got {target_count}.")

    sample_points, _ = qdrant.scroll(
        collection_name=source_collection,
        limit=min(32, source_count),
        with_payload=False,
        with_vectors=False,
    )
    sample_ids = [str(point.id) for point in sample_points]
    sample_rows = _query_by_ids(
        milvus,
        target_collection,
        sample_ids,
        ["id", "neighbors", GRAPH_BUILD_FIELD, ENTRY_POINT_FIELD],
    )
    sample_by_id = {record_id: record for record in sample_rows if (record_id := _milvus_id(record)) is not None}
    for point_id in sample_ids:
        record = sample_by_id.get(point_id)
        if record is None or record.get(GRAPH_BUILD_FIELD) != build_id:
            raise RuntimeError(f"Graph validation failed for sample point {point_id!r}.")
        neighbors = record.get("neighbors") or []
        if len(neighbors) != expected_degree or point_id in {str(value) for value in neighbors}:
            raise RuntimeError(f"Invalid neighbor list for sample point {point_id!r}.")

    return {
        "points": processed,
        "edges": edge_count,
        "neighbor_degree": {"minimum": min_degree or 0, "maximum": max_degree},
        "sample_points_validated": len(sample_ids),
        "target_count": target_count,
        "resume_skipped_points": resume_skipped_points,
    }


def main() -> int:
    load_dotenv()
    args = parse_args()
    if min(args.graph_degree, args.scroll_page_size, args.embedding_batch_size, args.search_batch_size) < 1:
        raise RuntimeError("Graph degree and batch sizes must be positive integers.")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_token = os.getenv("ZILLIZ_TOKEN")
    if not qdrant_url or not zilliz_uri or not zilliz_token:
        raise RuntimeError("QDRANT_URL, ZILLIZ_URI, and ZILLIZ_TOKEN must be configured.")

    source_collection = args.source_collection or os.getenv("QDRANT_COLLECTION", "document_chunks")
    old_hypencoder_collection = os.getenv("QDRANT_COLLECTION_HYPENCODER", "document_chunks_hypencoder")
    if source_collection == old_hypencoder_collection:
        raise RuntimeError("Source must be the original document collection, not the old Hypencoder collection.")

    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key.strip() if qdrant_key else None,
        timeout=120.0,
        prefer_grpc=False,
    )
    if not qdrant.collection_exists(source_collection):
        raise RuntimeError(f"Qdrant source collection {source_collection!r} does not exist.")
    source_count = qdrant.count(source_collection, exact=True).count
    if source_count < 1:
        raise RuntimeError("Qdrant source collection is empty.")

    milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    try:
        _ensure_collection(milvus, args.target_collection, args.resume)
        current_count = _collection_count(milvus, args.target_collection)
        if current_count > source_count:
            raise RuntimeError(
                f"Staging Zilliz collection has {current_count} rows, more than Qdrant source {source_count}."
            )

        device = args.device or get_default_device()
        print(
            f"Fresh build from Qdrant payloads: {source_collection} ({source_count} chunks) -> "
            f"Zilliz {args.target_collection}; model={args.model_name}; device={device}",
            flush=True,
        )
        started = time.perf_counter()

        model, tokenizer = load_hypencoder(args.model_name, device=device)
        passage_encoder = model.passage_encoder.to(device)
        passage_encoder.eval()
        new_vectors = _embed_source_documents(
            qdrant=qdrant,
            milvus=milvus,
            source_collection=source_collection,
            target_collection=args.target_collection,
            source_count=source_count,
            build_id=args.build_id,
            encoder=passage_encoder,
            tokenizer=tokenizer,
            device=device,
            page_size=args.scroll_page_size,
            embedding_batch_size=args.embedding_batch_size,
        )

        graph = _build_graph(
            qdrant=qdrant,
            milvus=milvus,
            source_collection=source_collection,
            target_collection=args.target_collection,
            source_count=source_count,
            build_id=args.build_id,
            graph_degree=args.graph_degree,
            page_size=args.scroll_page_size,
            search_batch_size=args.search_batch_size,
        )

        report = {
            "status": "validated",
            "build_id": args.build_id,
            "embedding_model": args.model_name,
            "source_collection": source_collection,
            "source_count": source_count,
            "target_collection": args.target_collection,
            "target_count": graph["target_count"],
            "new_vectors_embedded_this_run": new_vectors,
            "passage_max_length": PAPER_PASSAGE_MAX_LENGTH,
            "graph_metric": "L2 approximate nearest neighbors via Zilliz vector search",
            "graph_degree_requested": args.graph_degree,
            "graph": graph,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        with open(args.report, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as summary_file:
                summary_file.write(
                    f"## Hypencoder Zilliz build validated\n\n"
                    f"- Source: `{source_collection}` ({source_count:,} chunks)\n"
                    f"- Zilliz: `{args.target_collection}` ({graph['target_count']:,} chunks)\n"
                    f"- Graph edges: {graph['edges']:,} (degree {args.graph_degree})\n"
                    f"- Build ID: `{args.build_id}`\n"
                )
        return 0
    finally:
        milvus.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Hypencoder-to-Zilliz build failed: {exc}", file=sys.stderr)
        sys.exit(1)
