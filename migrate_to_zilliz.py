"""Migrate document chunks and vectors from Qdrant to Zilliz Cloud (Milvus).

Supports Hypencoder representations (768-dim L2 with neighbor graph) as well as
standard bi-encoder collections.

Configured via environment variables, CLI arguments, or local credentials in zilliz_token.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from pymilvus import MilvusClient
from qdrant_client import QdrantClient
from tqdm import tqdm

DEFAULT_ZILLIZ_URI = "https://in03-e6db1074d3f3592.serverless.aws-eu-central-1.cloud.zilliz.com"
DEFAULT_PAGE_SIZE = 256
DEFAULT_UPSERT_BATCH = 100


def get_zilliz_token() -> Optional[str]:
    """Retrieve Zilliz token from environment or local token file."""
    token = os.getenv("ZILLIZ_TOKEN")
    if token and token.strip():
        return token.strip()

    # Fallback to local token file if present
    token_path = os.path.join(os.path.dirname(__file__), "zilliz_token", "token.txt")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-collection",
        default=None,
        help="Source Qdrant collection name (default: from QDRANT_COLLECTION or 'document_chunks')",
    )
    parser.add_argument(
        "--target-collection",
        default=None,
        help="Target Zilliz collection name (default: from ZILLIZ_COLLECTION or matches source)",
    )
    parser.add_argument(
        "--zilliz-uri",
        default=None,
        help=f"Zilliz Cloud cluster URI (default: {DEFAULT_ZILLIZ_URI})",
    )
    parser.add_argument(
        "--zilliz-token",
        default=None,
        help="Zilliz Cloud API token / key",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant cluster URL",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Qdrant API key",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_UPSERT_BATCH,
        help=f"Batch size for Zilliz upsert (default: {DEFAULT_UPSERT_BATCH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of items to migrate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted migration",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL must be specified or set in environment.")

    zilliz_uri = args.zilliz_uri or os.getenv("ZILLIZ_URI", DEFAULT_ZILLIZ_URI)
    zilliz_token = args.zilliz_token or get_zilliz_token()
    if not zilliz_token:
        raise RuntimeError(
            "Zilliz token must be specified via --zilliz-token, ZILLIZ_TOKEN env, or zilliz_token/token.txt."
        )

    source_collection = (
        args.source_collection
        or os.getenv("QDRANT_COLLECTION_HYPENCODER")
        or os.getenv("QDRANT_COLLECTION", "document_chunks")
    )
    target_collection = (
        args.target_collection
        or os.getenv("ZILLIZ_COLLECTION")
        or source_collection
    )

    print(f"Source Qdrant collection: {source_collection!r}", flush=True)
    print(f"Target Zilliz collection: {target_collection!r}", flush=True)
    print(f"Zilliz URI: {zilliz_uri}", flush=True)

    # Initialize Qdrant client
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key.strip() if qdrant_api_key else None,
        timeout=60.0,
        prefer_grpc=False,
    )

    if not qdrant.collection_exists(source_collection):
        raise RuntimeError(f"Source Qdrant collection {source_collection!r} does not exist.")

    source_info = qdrant.get_collection(source_collection)
    source_count = qdrant.count(source_collection, exact=True).count
    print(f"Source points available: {source_count}", flush=True)

    # Determine vector dimension and metric from source
    source_vectors = source_info.config.params.vectors
    if isinstance(source_vectors, dict):
        raise RuntimeError("Named vectors are not currently supported for automated migration.")

    dim = source_vectors.size
    dist = str(source_vectors.distance).lower()
    if "euclid" in dist or "l2" in dist:
        metric_type = "L2"
    elif "cosine" in dist:
        metric_type = "COSINE"
    else:
        metric_type = "IP"

    print(f"Vector configuration: dimension={dim}, metric={metric_type}", flush=True)

    # Initialize Zilliz Milvus client
    milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)

    try:
        # Check target collection in Zilliz
        if not milvus.has_collection(target_collection):
            print(f"Creating Zilliz collection {target_collection!r}...", flush=True)
            milvus.create_collection(
                collection_name=target_collection,
                dimension=dim,
                metric_type=metric_type,
                primary_field_name="id",
                id_type="string",
                max_length=128,
                vector_field_name="vector",
                auto_id=False,
                enable_dynamic_field=True,
            )
        else:
            print(f"Target Zilliz collection {target_collection!r} already exists.", flush=True)

        existing_ids: Set[str] = set()
        if args.resume:
            print("Checking existing points in Zilliz for resume...", flush=True)
            try:
                # Query existing IDs
                existing_records = milvus.query(
                    collection_name=target_collection,
                    filter="id != ''",
                    output_fields=["id"],
                    limit=16384,
                )
                existing_ids = {r["id"] for r in existing_records}
                print(f"Resume: found {len(existing_ids)} points already in Zilliz.", flush=True)
            except Exception as e:
                print(f"[WARN] Could not retrieve existing IDs: {e}", flush=True)

        # Scroll points from Qdrant and upsert to Zilliz
        offset = None
        scanned = 0
        migrated = 0
        max_items = args.limit if args.limit is not None else source_count

        pbar = tqdm(total=min(source_count, max_items), desc="Migrating to Zilliz")

        while scanned < max_items:
            page_limit = min(DEFAULT_PAGE_SIZE, max_items - scanned)
            points, next_offset = qdrant.scroll(
                collection_name=source_collection,
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            batch_to_upsert: List[Dict[str, Any]] = []

            for p in points:
                pid_str = str(p.id)
                if pid_str in existing_ids:
                    scanned += 1
                    pbar.update(1)
                    continue

                if p.vector is None or len(p.vector) != dim:
                    continue

                payload = p.payload or {}
                row: Dict[str, Any] = {
                    "id": pid_str,
                    "vector": [float(x) for x in p.vector],
                    "text": str(payload.get("text", "")),
                    "source": str(payload.get("source") or payload.get("filename") or "unknown"),
                    "_stable_id": str(payload.get("_stable_id") or pid_str),
                }

                # Add neighbor IDs if present (Hypencoder graph)
                if "neighbors" in payload and isinstance(payload["neighbors"], list):
                    row["neighbors"] = [str(n) for n in payload["neighbors"]]

                # Copy additional metadata
                for k, v in payload.items():
                    if k not in row:
                        if isinstance(v, (str, int, float, bool, list)):
                            row[k] = v
                        else:
                            row[k] = str(v)

                batch_to_upsert.append(row)
                scanned += 1
                pbar.update(1)

                if len(batch_to_upsert) >= args.batch_size:
                    milvus.upsert(collection_name=target_collection, data=batch_to_upsert)
                    migrated += len(batch_to_upsert)
                    batch_to_upsert = []

                if scanned >= max_items:
                    break

            if batch_to_upsert:
                milvus.upsert(collection_name=target_collection, data=batch_to_upsert)
                migrated += len(batch_to_upsert)

            if next_offset is None or scanned >= max_items:
                break
            offset = next_offset

        pbar.close()

        # Check total points in Zilliz target
        print(f"Migration completed. Upserted {migrated} points this run.", flush=True)
        stats = milvus.get_collection_stats(collection_name=target_collection)
        row_count = stats.get("row_count", "unknown")
        print(f"Target collection stats: row_count={row_count}", flush=True)
        return 0

    finally:
        milvus.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Migration to Zilliz failed: {exc}", file=sys.stderr)
        sys.exit(1)
