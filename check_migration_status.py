"""Print true only when the EmbeddingGemma 2 destination matches the source IDs."""

import os
import sys

from qdrant_client import QdrantClient


PAGE_SIZE = 1000
TARGET_DIMENSION = 256


def collection_ids(client: QdrantClient, collection: str) -> set[int | str]:
    result = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=PAGE_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        result.update(point.id for point in points)
        if next_offset is None:
            return result
        offset = next_offset


def main() -> int:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")

    source_name = os.getenv("QDRANT_COLLECTION", "document_chunks")
    target_name = os.getenv("QDRANT_COLLECTION_EG2", f"{source_name}_embeddinggemma2")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None)

    if source_name == target_name or not client.collection_exists(source_name):
        print("false")
        return 0
    if not client.collection_exists(target_name):
        print("false")
        return 0

    source_info = client.get_collection(source_name)
    target_info = client.get_collection(target_name)
    source_vectors = source_info.config.params.vectors
    target_vectors = target_info.config.params.vectors
    if isinstance(source_vectors, dict) or isinstance(target_vectors, dict):
        print("false")
        return 0
    if target_vectors.size != TARGET_DIMENSION or target_vectors.distance != source_vectors.distance:
        print("false")
        return 0

    source_count = client.count(source_name, exact=True).count
    target_count = client.count(target_name, exact=True).count
    if source_count == 0 or source_count != target_count:
        print("false")
        return 0

    print("true" if collection_ids(client, source_name) == collection_ids(client, target_name) else "false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Migration status check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
