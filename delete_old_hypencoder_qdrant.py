"""Delete the legacy Qdrant Hypencoder collection after validating the Zilliz replacement."""

import json
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pymilvus import MilvusClient
from qdrant_client import QdrantClient

GRAPH_BUILD_FIELD = "hypencoder_graph_build_id"
GRAPH_PAGE_SIZE = 256


def _validate_destination_graph(
    milvus: MilvusClient,
    qdrant: QdrantClient,
    source_collection: str,
    zilliz_collection: str,
    source_count: int,
    build_id: str,
    graph_degree: int,
) -> None:
    expected_degree = min(graph_degree, max(source_count - 1, 0))
    offset: Optional[Any] = None
    validated = 0
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=source_collection,
            limit=GRAPH_PAGE_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        ids = [str(point.id) for point in points]
        records = milvus.query(
            collection_name=zilliz_collection,
            filter=f"id in {json.dumps(ids)}",
            output_fields=["id", "neighbors", GRAPH_BUILD_FIELD],
            limit=len(ids),
            consistency_level="Strong",
        )
        by_id: Dict[str, Dict[str, Any]] = {str(record.get("id")): record for record in records}
        for point_id in ids:
            record = by_id.get(point_id)
            neighbors = record.get("neighbors") if record else None
            if (
                record is None
                or record.get(GRAPH_BUILD_FIELD) != build_id
                or not isinstance(neighbors, list)
                or len(neighbors) != expected_degree
                or point_id in {str(neighbor) for neighbor in neighbors}
            ):
                raise RuntimeError(
                    f"Refusing deletion: destination graph validation failed for source point {point_id!r}."
                )
        validated += len(ids)
        print(f"Validated destination graph for {validated}/{source_count} source points.", flush=True)
        if next_offset is None:
            break
        offset = next_offset

    if validated != source_count:
        raise RuntimeError(f"Destination validation scanned {validated}/{source_count} source points.")


def main() -> int:
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_token = os.getenv("ZILLIZ_TOKEN")
    source_collection = os.getenv("QDRANT_COLLECTION", "document_chunks")
    old_collection = os.getenv("QDRANT_COLLECTION_HYPENCODER", "document_chunks_hypencoder")
    zilliz_collection = os.getenv("ZILLIZ_COLLECTION_HYPENCODER")
    build_id = os.getenv("HYPENCODER_BUILD_ID")
    graph_degree = int(os.getenv("GRAPH_DEGREE", "100"))

    if not all((qdrant_url, zilliz_uri, zilliz_token, zilliz_collection, build_id)):
        raise RuntimeError("Qdrant, Zilliz, target collection, and build ID settings are required.")
    if graph_degree < 1:
        raise RuntimeError("GRAPH_DEGREE must be a positive integer.")
    if old_collection == source_collection:
        raise RuntimeError("Refusing to delete the original document collection.")

    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key.strip() if qdrant_key else None,
        timeout=120.0,
        prefer_grpc=False,
    )
    if not qdrant.collection_exists(source_collection):
        raise RuntimeError(f"Original Qdrant source {source_collection!r} is missing.")
    source_count = qdrant.count(source_collection, exact=True).count

    milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    try:
        if not milvus.has_collection(zilliz_collection):
            raise RuntimeError(f"Validated Zilliz collection {zilliz_collection!r} is missing.")
        milvus.load_collection(collection_name=zilliz_collection)
        milvus.flush(collection_name=zilliz_collection)
        stats = milvus.get_collection_stats(collection_name=zilliz_collection)
        try:
            target_count = int(stats["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Could not verify Zilliz row count: {stats}") from exc
        if target_count != source_count:
            raise RuntimeError(f"Refusing deletion: source has {source_count}, Zilliz has {target_count} points.")
        _validate_destination_graph(
            milvus=milvus,
            qdrant=qdrant,
            source_collection=source_collection,
            zilliz_collection=zilliz_collection,
            source_count=source_count,
            build_id=build_id,
            graph_degree=graph_degree,
        )
    finally:
        milvus.close()

    if not qdrant.collection_exists(old_collection):
        print(f"Legacy Qdrant collection {old_collection!r} is already absent; nothing to delete.", flush=True)
        return 0

    old_count = qdrant.count(old_collection, exact=True).count
    print(
        f"Validated Zilliz {zilliz_collection!r} ({target_count}/{source_count} points and graph links for every source ID). "
        f"Deleting legacy Qdrant Hypencoder collection {old_collection!r} ({old_count} points).",
        flush=True,
    )
    qdrant.delete_collection(collection_name=old_collection)
    if qdrant.collection_exists(old_collection):
        raise RuntimeError(f"Qdrant collection {old_collection!r} still exists after delete request.")
    print(f"Deleted legacy Qdrant Hypencoder collection {old_collection!r}; original source remains intact.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Refused or failed to delete legacy Qdrant collection: {exc}", file=sys.stderr)
        sys.exit(1)
