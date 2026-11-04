"""Compare MiniLM and EmbeddingGemma 2 on a labeled JSONL query set.

Each line must look like: {"query": "...", "relevant_ids": ["stable-chunk-id"]}
IDs should match the `_stable_id` payload stored in Qdrant.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EG2_MODEL = "google/embeddinggemma-2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def percentile(values, pct):
    return float(np.percentile(values, pct)) if values else 0.0


def load_queries(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = row.get("query")
            relevant = row.get("relevant_ids")
            if not isinstance(query, str) or not query.strip() or not isinstance(relevant, list) or not relevant:
                raise ValueError(f"Line {line_number} needs a non-empty query and relevant_ids list")
            rows.append((query, {str(value) for value in relevant}))
    if not rows:
        raise ValueError("The labeled query set is empty")
    return rows


def point_relevance_id(point):
    payload = point.payload or {}
    return str(payload.get("_stable_id", point.id))


def metrics_at_k(ranked_ids, relevant_ids, k):
    top_ids = ranked_ids[:k]
    hits = sum(item in relevant_ids for item in top_ids)
    return hits / k, hits / len(relevant_ids)


def evaluate_collection(client, collection, model, reranker, queries, candidate_k, final_k, is_eg2):
    embedding_ms, qdrant_ms, rerank_ms, total_ms = [], [], [], []
    candidate_precision, candidate_recall = [], []
    final_precision, final_recall = [], []

    for index, (query, relevant_ids) in enumerate(queries, start=1):
        started = time.perf_counter()
        embedded_at = started
        if is_eg2:
            query_vector = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
        else:
            query_vector = model.encode(query, normalize_embeddings=True)
        retrieved_at = time.perf_counter()
        points = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=candidate_k,
            with_payload=True,
            with_vectors=False,
        ).points
        rerank_started = time.perf_counter()
        candidate_ids = [point_relevance_id(point) for point in points]
        candidate_p, candidate_r = metrics_at_k(candidate_ids, relevant_ids, candidate_k)
        pairs = [[query, (point.payload or {}).get("text", "")] for point in points]
        scores = reranker.predict(pairs).tolist() if pairs else []
        ranked_ids = [point_relevance_id(point) for point, _ in sorted(zip(points, scores), key=lambda pair: pair[1], reverse=True)]
        finished = time.perf_counter()
        final_p, final_r = metrics_at_k(ranked_ids, relevant_ids, final_k)

        embedding_ms.append((retrieved_at - embedded_at) * 1000)
        qdrant_ms.append((rerank_started - retrieved_at) * 1000)
        rerank_ms.append((finished - rerank_started) * 1000)
        total_ms.append((finished - started) * 1000)
        candidate_precision.append(candidate_p)
        candidate_recall.append(candidate_r)
        final_precision.append(final_p)
        final_recall.append(final_r)
        if index % 100 == 0 or index == len(queries):
            print(f"{collection}: evaluated {index}/{len(queries)} queries", flush=True)

    return {
        "collection": collection,
        "queries": len(queries),
        "candidate_precision_at_k": statistics.mean(candidate_precision),
        "candidate_recall_at_k": statistics.mean(candidate_recall),
        "reranked_precision_at_k": statistics.mean(final_precision),
        "reranked_recall_at_k": statistics.mean(final_recall),
        "latency_ms": {
            "embedding_mean": statistics.mean(embedding_ms),
            "qdrant_mean": statistics.mean(qdrant_ms),
            "rerank_mean": statistics.mean(rerank_ms),
            "total_mean": statistics.mean(total_ms),
            "total_median": statistics.median(total_ms),
            "total_p95": percentile(total_ms, 95),
        },
    }


def main():
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", help="Labeled JSONL query set")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--minilm-collection", default=os.getenv("QDRANT_COLLECTION", "document_chunks"))
    parser.add_argument("--eg2-collection", default=os.getenv("QDRANT_COLLECTION_EG2", "document_chunks_embeddinggemma2"))
    parser.add_argument("--only", choices=("both", "minilm", "eg2"), default="both")
    parser.add_argument("--output-json", default=None, help="Optional path for combined benchmark results")
    args = parser.parse_args()

    url, api_key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")
    queries = load_queries(args.queries)
    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None, timeout=60.0)
    reranker = CrossEncoder(RERANKER_MODEL)

    configurations = (
        (args.minilm_collection, MINILM_MODEL, False),
        (args.eg2_collection, EG2_MODEL, True),
    )
    if args.only == "minilm":
        configurations = configurations[:1]
    elif args.only == "eg2":
        configurations = configurations[1:]
    results = []
    for collection, model_id, is_eg2 in configurations:
        model_options = (
            {"config_kwargs": {"vision_config": None, "audio_config": None}, "truncate_dim": 256}
            if is_eg2 else {}
        )
        model = SentenceTransformer(model_id, **model_options)
        result = evaluate_collection(
            client, collection, model, reranker, queries,
            args.candidate_k, args.final_k, is_eg2,
        )
        result["model"] = model_id
        result["candidate_k"] = args.candidate_k
        result["final_k"] = args.final_k
        results.append(result)
        print(json.dumps(result, indent=2))

    if args.output_json:
        report = {
            "status": "complete",
            "query_count": len(queries),
            "query_type": "synthetic corpus-derived probes with one labeled source chunk per query",
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "results": results,
        }
        Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        if os.getenv("GITHUB_STEP_SUMMARY"):
            by_collection = {result["collection"]: result for result in results}
            minilm = by_collection.get(args.minilm_collection)
            eg2 = by_collection.get(args.eg2_collection)
            lines = [
                "## EmbeddingGemma 2 vs MiniLM retrieval benchmark",
                "",
                f"- Queries: {len(queries):,} synthetic probes; each has one labeled source chunk.",
                "- These metrics measure corpus-derived retrieval probes, not human-judged QA accuracy.",
                "",
                f"| Metric | MiniLM | EmbeddingGemma 2 |",
                "| --- | ---: | ---: |",
            ]
            for label, key in (
                (f"Candidate Precision@{args.candidate_k}", "candidate_precision_at_k"),
                (f"Candidate Recall@{args.candidate_k}", "candidate_recall_at_k"),
                (f"Reranked Precision@{args.final_k}", "reranked_precision_at_k"),
                (f"Reranked Recall@{args.final_k}", "reranked_recall_at_k"),
            ):
                lines.append(
                    f"| {label} | {minilm[key]:.4f} | {eg2[key]:.4f} |" if minilm and eg2
                    else f"| {label} | — | — |"
                )
            for label, key in (
                ("Embedding mean latency (ms/query)", "embedding_mean"),
                ("Qdrant mean latency (ms/query)", "qdrant_mean"),
                ("Rerank mean latency (ms/query)", "rerank_mean"),
                ("End-to-end mean latency (ms/query)", "total_mean"),
                ("End-to-end p95 latency (ms/query)", "total_p95"),
            ):
                lines.append(
                    f"| {label} | {minilm['latency_ms'][key]:.2f} | {eg2['latency_ms'][key]:.2f} |"
                    if minilm and eg2 else f"| {label} | — | — |"
                )
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
                summary.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
