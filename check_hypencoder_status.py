"""Check migration and configuration status of the Hypencoder collection in Qdrant."""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

TARGET_DIMENSION = 768


def main() -> int:
    load_dotenv()
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")

    source_name = os.getenv("QDRANT_COLLECTION", "document_chunks")
    target_name = os.getenv("QDRANT_COLLECTION_HYPENCODER", f"{source_name}_hypencoder")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None, timeout=30.0, prefer_grpc=False)

    if not client.collection_exists(target_name):
        print(f"Target collection {target_name!r} does not exist.")
        print("false")
        return 0

    target_info = client.get_collection(target_name)
    vectors = target_info.config.params.vectors

    if isinstance(vectors, dict):
        print(f"Collection {target_name!r} uses named vectors, expected single dense vector.")
        print("false")
        return 0

    if vectors.size != TARGET_DIMENSION:
        print(f"Vector size {vectors.size} does not match target {TARGET_DIMENSION}.")
        print("false")
        return 0

    if vectors.distance != models.Distance.EUCLID:
        print(f"Vector distance {vectors.distance} does not match {models.Distance.EUCLID}.")
        print("false")
        return 0

    target_count = client.count(target_name, exact=True).count
    print(f"Collection {target_name!r} verified: {target_count} points, dim={vectors.size}, distance={vectors.distance}")

    # Check sample point
    points, _ = client.scroll(target_name, limit=1, with_payload=True, with_vectors=False)
    if points:
        sample = points[0]
        has_neighbors = "neighbors" in (sample.payload or {})
        num_neighbors = len(sample.payload.get("neighbors", [])) if has_neighbors else 0
        print(f"Sample point ID: {sample.id}, has_neighbors={has_neighbors}, neighbor_count={num_neighbors}")

    print("true")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Check error: {exc}", file=sys.stderr)
        sys.exit(1)
