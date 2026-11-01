"""Check whether Qdrant points have been migrated to Zilliz Cloud.

Outputs 'true' if the target collection in Zilliz exists and matches source point count,
otherwise outputs 'false'.
"""

import os
import sys
from dotenv import load_dotenv
from pymilvus import MilvusClient
from qdrant_client import QdrantClient

from migrate_to_zilliz import DEFAULT_ZILLIZ_URI, get_zilliz_token


def main() -> int:
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_URI", DEFAULT_ZILLIZ_URI)
    zilliz_token = os.getenv("ZILLIZ_TOKEN") or get_zilliz_token()

    if not qdrant_url or not zilliz_token:
        print("false")
        return 0

    source_collection = (
        os.getenv("QDRANT_COLLECTION_HYPENCODER")
        or os.getenv("QDRANT_COLLECTION", "document_chunks")
    )
    target_collection = os.getenv("ZILLIZ_COLLECTION") or source_collection

    try:
        qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key.strip() if qdrant_api_key else None,
            timeout=30.0,
            prefer_grpc=False,
        )
        if not qdrant.collection_exists(source_collection):
            print("false")
            return 0
        source_count = qdrant.count(source_collection, exact=True).count
        if source_count == 0:
            print("false")
            return 0
    except Exception:
        print("false")
        return 0

    try:
        milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)
        if not milvus.has_collection(target_collection):
            print("false")
            milvus.close()
            return 0

        stats = milvus.get_collection_stats(collection_name=target_collection)
        target_count = stats.get("row_count", 0)
        milvus.close()

        if target_count >= source_count:
            print("true")
        else:
            print("false")
        return 0
    except Exception:
        print("false")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("false")
        sys.exit(0)
