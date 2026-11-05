"""Create a deterministic full query set for a sharded Hypencoder benchmark."""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from benchmark_hypencoder_large import (
    MAXIMUM_QUERY_COUNT,
    MINIMUM_QUERY_COUNT,
    _hash_query_set,
    build_benchmark_queries,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--source-collection", default=None)
    parser.add_argument("--output", default="benchmark_queries.json")
    args = parser.parse_args()
    if not MINIMUM_QUERY_COUNT <= args.query_count <= MAXIMUM_QUERY_COUNT:
        parser.error(f"--query-count must be between {MINIMUM_QUERY_COUNT} and {MAXIMUM_QUERY_COUNT}")

    load_dotenv(dotenv_path=".env")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")
    source_collection = args.source_collection or os.getenv(
        "QDRANT_COLLECTION_EG2", "document_chunks_embeddinggemma2"
    )
    client = QdrantClient(
        url=url,
        api_key=api_key.strip() if api_key else None,
        timeout=60.0,
        prefer_grpc=False,
    )
    if not client.collection_exists(source_collection):
        raise RuntimeError(f"Qdrant source collection {source_collection!r} does not exist")
    source_count = client.count(source_collection, exact=True).count
    if source_count < args.query_count:
        raise RuntimeError(f"Requested {args.query_count:,} probes but source has only {source_count:,} points.")

    started = time.perf_counter()
    queries = build_benchmark_queries(client, source_collection, args.query_count)
    query_set_sha256 = _hash_query_set(queries)
    bundle = {
        "schema_version": 1,
        "source_collection": source_collection,
        "source_count": source_count,
        "query_count": len(queries),
        "query_set_sha256": query_set_sha256,
        "queries": queries,
    }
    temporary_path = f"{args.output}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output:
        json.dump(bundle, output, ensure_ascii=False, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, args.output)

    print(
        f"Prepared {len(queries):,} unique probes from {source_collection} "
        f"({source_count:,} chunks) in {time.perf_counter() - started:.1f}s; "
        f"SHA-256={query_set_sha256}; saved {args.output}",
        flush=True,
    )
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as output:
            output.write(
                f"## Prepared {len(queries):,} benchmark probes\n\n"
                f"- Source: `{source_collection}` ({source_count:,} chunks)\n"
                f"- Query-set SHA-256: `{query_set_sha256}`\n"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Benchmark probe preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
