"""Generate deterministic passage-derived query labels for embedding comparison."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from benchmark_hypencoder_large import build_benchmark_queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--source-collection", default=None)
    parser.add_argument("--output", default="embedding_benchmark_queries.jsonl")
    args = parser.parse_args()
    if not 1 <= args.query_count <= 10_000:
        parser.error("--query-count must be between 1 and 10000")

    load_dotenv(Path(".env"))
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")
    source_collection = args.source_collection or os.getenv("QDRANT_COLLECTION", "document_chunks")
    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None, timeout=60.0)
    if not client.collection_exists(source_collection):
        raise RuntimeError(f"Qdrant source collection {source_collection!r} does not exist")
    source_count = client.count(source_collection, exact=True).count
    if source_count < args.query_count:
        raise RuntimeError(
            f"Requested {args.query_count:,} probes but source has only {source_count:,} points."
        )

    started = time.perf_counter()
    probes = build_benchmark_queries(client, source_collection, args.query_count)
    temporary_path = Path(f"{args.output}.tmp")
    query_hash = hashlib.sha256()
    with temporary_path.open("w", encoding="utf-8") as output:
        for probe in probes:
            line = json.dumps(probe, ensure_ascii=False, separators=(",", ":"))
            output.write(line + "\n")
            query_hash.update(line.encode("utf-8"))
            query_hash.update(b"\n")
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(args.output)

    message = (
        f"Prepared {len(probes):,} deterministic probes from {source_collection} "
        f"({source_count:,} source chunks); SHA-256={query_hash.hexdigest()}; "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    print(message, flush=True)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(
                f"## Prepared {len(probes):,} embedding benchmark probes\n\n"
                f"- Source: `{source_collection}` ({source_count:,} chunks)\n"
                f"- Query set SHA-256: `{query_hash.hexdigest()}`\n"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Embedding benchmark probe preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
