"""Validate and combine completed Hypencoder benchmark shard results."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


METRIC_FIELDS = (
    "candidate_recall",
    "candidate_precision",
    "reranked_recall",
    "reranked_precision",
    "mrr",
)


def _weighted_phase(shards: List[Dict[str, Any]], phase_name: str) -> Dict[str, Any]:
    completed = sum(int(shard[phase_name]["queries_completed"]) for shard in shards)
    if completed == 0:
        raise RuntimeError(f"No completed queries for {phase_name}.")
    merged = {
        field: sum(
            float(shard[phase_name][field]) * int(shard[phase_name]["queries_completed"])
            for shard in shards
        ) / completed
        for field in METRIC_FIELDS
    }
    latency_fields = set().union(*(shard[phase_name]["latency_ms"] for shard in shards))
    merged["latency_ms"] = {
        field: sum(
            float(shard[phase_name]["latency_ms"][field])
            * int(shard[phase_name]["queries_completed"])
            for shard in shards
        ) / completed
        for field in latency_fields
    }
    merged["queries_completed"] = completed
    return merged


def _load_shards(results_dir: Path, expected_count: int) -> List[Dict[str, Any]]:
    by_start: Dict[int, Dict[str, Any]] = {}
    # Prefer resume artifacts when the initial job timed out; a completed initial
    # result is also emitted by the resume job after loading its checkpoint.
    result_paths = sorted(results_dir.rglob("benchmark_results_*.json"))
    for path in result_paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read shard result {path}: {exc}") from exc
        if result.get("status") != "complete":
            continue
        start = int(result.get("query_start", -1))
        count = int(result.get("query_count", 0))
        if start < 0 or count <= 0:
            raise RuntimeError(f"Shard result {path} has an invalid query range.")
        current = by_start.get(start)
        if current is None or path.parts[-2].startswith("benchmark-shard-resume-"):
            by_start[start] = result

    shards = sorted(by_start.values(), key=lambda result: int(result["query_start"]))
    if not shards:
        raise RuntimeError(f"No completed shard results found under {results_dir}.")

    expected_start = 0
    full_hash = shards[0].get("full_query_set_sha256")
    compatibility_fields = (
        "total_query_count",
        "candidate_k",
        "final_k",
        "source_corpus_count",
        "hypencoder_corpus_count",
        "hypencoder_traversal",
        "evaluation_batch_size",
    )
    baseline = shards[0]
    for result in shards:
        start = int(result["query_start"])
        count = int(result["query_count"])
        if start != expected_start:
            raise RuntimeError(
                f"Benchmark shards are missing or overlap at query {expected_start}; next shard starts at {start}."
            )
        if result.get("full_query_set_sha256") != full_hash:
            raise RuntimeError("Shard results use different full query sets.")
        for field in compatibility_fields:
            if result.get(field) != baseline.get(field):
                raise RuntimeError(f"Shard results disagree on {field!r}.")
        if int(result.get("total_query_count", -1)) != expected_count:
            raise RuntimeError("Shard total query count does not match the requested benchmark size.")
        for phase in ("biencoder", "hypencoder"):
            if int(result[phase].get("queries_completed", 0)) != count:
                raise RuntimeError(f"Shard {start} has incomplete {phase} results.")
        expected_start += count
    if expected_start != expected_count:
        raise RuntimeError(f"Only {expected_start:,}/{expected_count:,} queries are covered by complete shards.")
    return shards


def _summary(shards: List[Dict[str, Any]], query_count: int) -> Dict[str, Any]:
    first = shards[0]
    return {
        "status": "complete",
        "query_count": query_count,
        "shard_count": len(shards),
        "query_type": first["query_type"],
        "full_query_set_sha256": first["full_query_set_sha256"],
        "evaluation_batch_size": first["evaluation_batch_size"],
        "latency_interpretation": first["latency_interpretation"],
        "hypencoder_random_seed": first["hypencoder_random_seed"],
        "hypencoder_traversal": first["hypencoder_traversal"],
        "candidate_k": first["candidate_k"],
        "final_k": first["final_k"],
        "source_corpus_count": first["source_corpus_count"],
        "hypencoder_corpus_count": first["hypencoder_corpus_count"],
        "biencoder": _weighted_phase(shards, "biencoder"),
        "hypencoder": _weighted_phase(shards, "hypencoder"),
    }


def _render_markdown(result: Dict[str, Any]) -> str:
    bi = result["biencoder"]
    hyp = result["hypencoder"]
    candidate_k = result["candidate_k"]
    final_k = result["final_k"]
    rows = (
        (f"Candidate Recall@{candidate_k}", "candidate_recall"),
        (f"Candidate Precision@{candidate_k}", "candidate_precision"),
        (f"Reranked Recall@{final_k}", "reranked_recall"),
        (f"Reranked Precision@{final_k}", "reranked_precision"),
        ("MRR", "mrr"),
    )
    metrics = "\n".join(
        f"| {label} | {bi[key]:.4f} | {hyp[key]:.4f} | {hyp[key] - bi[key]:+.4f} |"
        for label, key in rows
    )
    return f"""# End-to-End Hypencoder Benchmark Results

- **Queries evaluated:** {result['query_count']:,}
- **Completed shards:** {result['shard_count']}
- **Corpus:** {result['source_corpus_count']:,} Qdrant chunks / {result['hypencoder_corpus_count']:,} Hypencoder chunks
- **Query-set SHA-256:** `{result['full_query_set_sha256']}`
- **Queries:** synthetic corpus-derived probes, one labeled source chunk per query; not human-judged QA relevance
- **Latency:** mean batch-amortized milliseconds per query across all shard results

| Metric | Bi-Encoder (MiniLM) | Hypencoder | Delta |
| --- | ---: | ---: | ---: |
{metrics}

| Latency (ms/query) | Bi-Encoder | Hypencoder |
| --- | ---: | ---: |
| Retrieval / traversal | {bi['latency_ms']['search']:.2f} | {hyp['latency_ms']['query_encoding_and_traversal']:.2f} |
| Cross-encoder reranking | {bi['latency_ms']['rerank']:.2f} | {hyp['latency_ms']['rerank']:.2f} |
| Total end-to-end | {bi['latency_ms']['total']:.2f} | {hyp['latency_ms']['total']:.2f} |
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--output-json", default="benchmark_results.json")
    args = parser.parse_args()
    shards = _load_shards(Path(args.results_dir), args.query_count)
    result = _summary(shards, args.query_count)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = _render_markdown(result)
    print(report, flush=True)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(report + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Benchmark aggregation failed: {exc}", file=sys.stderr)
        sys.exit(1)
