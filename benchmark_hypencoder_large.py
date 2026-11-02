"""Compare Hypencoder and MiniLM with reproducible corpus-derived query probes.

Builds at least 10,000 unique extractive queries from migrated Hypencoder passages,
labels each query with its source chunk, and reports retrieval/reranking metrics and
latency. These probes support a large workload benchmark; they are not human-authored
relevance judgments. Generates a GitHub Step Summary when running in Actions.
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

from hypencoder_retriever import HypencoderGraphRetriever, load_hypencoder

MINIMUM_QUERY_COUNT = 10_000
QUERY_SCROLL_PAGE_SIZE = 1000


def build_benchmark_queries(client, collection: str, query_count: int) -> List[Dict[str, Any]]:
    """Build reproducible, corpus-derived probes with their source chunk as a positive label."""
    queries: List[Dict[str, Any]] = []
    seen_queries: Set[str] = set()
    offset = None

    while len(queries) < query_count:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=min(QUERY_SCROLL_PAGE_SIZE, query_count - len(queries)),
            offset=offset,
            with_payload=["text", "filename", "source", "_stable_id"],
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            text = re.sub(r"\s+", " ", text).strip()

            chunk_id = str(payload.get("_stable_id") or point.id)
            sentences = [
                re.sub(r"\s+", " ", sentence).strip()
                for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
                if len(sentence.split()) >= 8
            ]
            seed = hashlib.sha256(chunk_id.encode("utf-8")).digest()
            snippet = sentences[int.from_bytes(seed[:4], "big") % len(sentences)] if sentences else text
            snippet = " ".join(snippet.split()[:36]).strip(" \t\r\n.,;:")

            paper = payload.get("filename") or payload.get("source") or "the research paper"
            paper = re.sub(r"[_-]+", " ", os.path.splitext(os.path.basename(str(paper)))[0]).strip()
            template = int.from_bytes(seed[4:8], "big") % 3
            base_query = (
                f"What does {paper} report about {snippet}?"
                if template == 0
                else f"Find the research passage about {snippet} from {paper}."
                if template == 1
                else f"According to {paper}, what is explained by: {snippet}?"
            )
            query = base_query
            collision_index = 1
            point_marker = hashlib.sha256(str(point.id).encode("utf-8")).hexdigest()[:10]
            while query in seen_queries:
                query = f"{base_query} [passage ref {point_marker}-{collision_index}]"
                collision_index += 1
            seen_queries.add(query)
            queries.append({"query": query, "relevant_ids": [chunk_id]})
            if len(queries) == query_count:
                break

        if next_offset is None:
            break
        offset = next_offset

    if len(queries) < query_count:
        raise RuntimeError(
            f"Requested {query_count} unique queries but built only {len(queries)} from {collection!r}. "
            "Increase the Hypencoder collection size or check that its points have text and stable IDs."
        )
    return queries


def compute_metrics(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> Tuple[float, float, float]:
    top_k = ranked_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    rr = 0.0
    for rank_idx, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            rr = 1.0 / rank_idx
            break

    return precision, recall, rr


def evaluate_biencoder(client, collection, minilm, reranker, queries, candidate_k, final_k):
    q_encode_ms, search_ms, rerank_ms, total_ms = [], [], [], []
    cand_p_list, cand_r_list = [], []
    final_p_list, final_r_list, mrr_list = [], [], []

    for query_index, item in enumerate(queries, start=1):
        query = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        query_vec = minilm.encode(query, normalize_embeddings=True)
        t1 = time.perf_counter()

        res = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=candidate_k,
            with_payload=True,
            with_vectors=False,
        )
        t2 = time.perf_counter()

        points = getattr(res, "points", None) or res
        cand_ids = [str((p.payload or {}).get("_stable_id") or p.id) for p in points]
        cand_p, cand_r, _ = compute_metrics(cand_ids, relevant, candidate_k)

        pairs = [[query, (p.payload or {}).get("text", "")] for p in points]
        scores = reranker.predict(pairs).tolist() if pairs else []
        ranked_points = [p for p, _ in sorted(zip(points, scores), key=lambda x: x[1], reverse=True)]
        t3 = time.perf_counter()

        ranked_ids = [str((p.payload or {}).get("_stable_id") or p.id) for p in ranked_points]
        final_p, final_r, rr = compute_metrics(ranked_ids, relevant, final_k)

        q_encode_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)
        rerank_ms.append((t3 - t2) * 1000)
        total_ms.append((t3 - t0) * 1000)

        cand_p_list.append(cand_p)
        cand_r_list.append(cand_r)
        final_p_list.append(final_p)
        final_r_list.append(final_r)
        mrr_list.append(rr)

        if query_index % 100 == 0 or query_index == len(queries):
            print(
                f"Bi-encoder queries: {query_index}/{len(queries)}; "
                f"mean end-to-end {statistics.mean(total_ms):.1f} ms/query",
                flush=True,
            )

    return {
        "candidate_recall": statistics.mean(cand_r_list),
        "candidate_precision": statistics.mean(cand_p_list),
        "reranked_recall": statistics.mean(final_r_list),
        "reranked_precision": statistics.mean(final_p_list),
        "mrr": statistics.mean(mrr_list),
        "latency_ms": {
            "query_encoding": statistics.mean(q_encode_ms),
            "search": statistics.mean(search_ms),
            "rerank": statistics.mean(rerank_ms),
            "total": statistics.mean(total_ms),
        },
    }


def evaluate_hypencoder(
    retriever,
    reranker,
    queries,
    candidate_k,
    final_k,
    entry_points=12,
    expansion_candidates=10,
    max_iter=4,
):
    traversal_ms, rerank_ms, total_ms = [], [], []
    cand_p_list, cand_r_list = [], []
    final_p_list, final_r_list, mrr_list = [], [], []

    for query_index, item in enumerate(queries, start=1):
        query = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        candidates = retriever.retrieve(
            query=query,
            k=candidate_k,
            num_entry_points=min(entry_points, len(retriever._all_known_ids)),
            ncandidates=expansion_candidates,
            max_iter=max_iter,
        )
        t1 = time.perf_counter()

        cand_ids = [str(c.get("id")) for c in candidates]
        cand_p, cand_r, _ = compute_metrics(cand_ids, relevant, candidate_k)

        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = reranker.predict(pairs).tolist() if pairs else []
        ranked_cands = [c for c, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]
        t2 = time.perf_counter()

        ranked_ids = [str(c.get("id")) for c in ranked_cands]
        final_p, final_r, rr = compute_metrics(ranked_ids, relevant, final_k)

        traversal_ms.append((t1 - t0) * 1000)
        rerank_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)

        cand_p_list.append(cand_p)
        cand_r_list.append(cand_r)
        final_p_list.append(final_p)
        final_r_list.append(final_r)
        mrr_list.append(rr)

        if query_index % 100 == 0 or query_index == len(queries):
            print(
                f"Hypencoder queries: {query_index}/{len(queries)}; "
                f"mean traversal {statistics.mean(traversal_ms):.1f} ms/query",
                flush=True,
            )

    return {
        "candidate_recall": statistics.mean(cand_r_list),
        "candidate_precision": statistics.mean(cand_p_list),
        "reranked_recall": statistics.mean(final_r_list),
        "reranked_precision": statistics.mean(final_p_list),
        "mrr": statistics.mean(mrr_list),
        "latency_ms": {
            "query_encoding_and_traversal": statistics.mean(traversal_ms),
            "rerank": statistics.mean(rerank_ms),
            "total": statistics.mean(total_ms),
        },
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate pool size (default: 20)")
    parser.add_argument("--final-k", type=int, default=5, help="Final reranked pool size (default: 5)")
    parser.add_argument(
        "--query-count",
        type=int,
        default=MINIMUM_QUERY_COUNT,
        help=f"Number of unique corpus-derived query probes (minimum {MINIMUM_QUERY_COUNT})",
    )
    parser.add_argument("--entry-points", type=int, default=12, help="Hypencoder graph starting points (default: 12)")
    parser.add_argument(
        "--expansion-candidates",
        type=int,
        default=10,
        help="Hypencoder graph nodes expanded per iteration (default: 10)",
    )
    parser.add_argument("--max-iter", type=int, default=4, help="Hypencoder graph traversal iterations (default: 4)")
    parser.add_argument("--output-json", default="benchmark_results.json", help="Path to save results JSON")
    args = parser.parse_args()
    if args.query_count < MINIMUM_QUERY_COUNT:
        parser.error(f"--query-count must be at least {MINIMUM_QUERY_COUNT}")
    if args.candidate_k < 1 or args.final_k < 1:
        parser.error("--candidate-k and --final-k must be positive")
    if args.entry_points < 1 or args.expansion_candidates < 1 or args.max_iter < 1:
        parser.error("--entry-points, --expansion-candidates, and --max-iter must be positive")

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")

    src_coll = os.getenv("QDRANT_COLLECTION", "document_chunks")
    hyp_coll = os.getenv("QDRANT_COLLECTION_HYPENCODER", "document_chunks_hypencoder")

    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None, timeout=60.0, prefer_grpc=False)

    hyp_count = client.count(hyp_coll, exact=True).count if client.collection_exists(hyp_coll) else 0
    src_count = client.count(src_coll, exact=True).count if client.collection_exists(src_coll) else 0

    print(f"Total source corpus size: {src_count} chunks")
    print(f"Hypencoder collection size: {hyp_count} chunks")
    if hyp_count < args.query_count:
        raise RuntimeError(
            f"Need at least {args.query_count} migrated Hypencoder chunks to build that many labeled queries; "
            f"the collection has {hyp_count}. Increase --limit in migrate_hypencoder.py."
        )

    print(f"\nBuilding {args.query_count:,} deterministic query probes from Hypencoder passages...")
    queries = build_benchmark_queries(client, hyp_coll, args.query_count)
    query_set_sha256 = hashlib.sha256(
        "\n".join(
            json.dumps(query, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for query in queries
        ).encode("utf-8")
    ).hexdigest()
    print(f"Built {len(queries):,} unique corpus-derived queries.")

    print("\nLoading models for benchmark...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n--- Evaluating Bi-Encoder (MiniLM) on Qdrant ---")
    bi_res = evaluate_biencoder(
        client=client,
        collection=src_coll,
        minilm=minilm,
        reranker=reranker,
        queries=queries,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
    )

    print("\n--- Evaluating Hypencoder (q-net + Graph Traversal) ---")
    print(
        "Traversal budget: "
        f"{args.entry_points} entry points, {args.expansion_candidates} expansion candidates, "
        f"{args.max_iter} iterations",
        flush=True,
    )
    model, tokenizer = load_hypencoder()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    retriever = HypencoderGraphRetriever(
        client=client,
        collection_name=hyp_coll,
        model=model,
        tokenizer=tokenizer,
        preload_limit=10000,
    )
    hyp_res = evaluate_hypencoder(
        retriever=retriever,
        reranker=reranker,
        queries=queries,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        entry_points=args.entry_points,
        expansion_candidates=args.expansion_candidates,
        max_iter=args.max_iter,
    )

    summary_md = f"""# 🚀 End-to-End Retrieval Benchmark Results

- **Corpus evaluated:** `{hyp_count}` Hypencoder chunks vs `{src_count}` source chunks
- **Test queries:** `{len(queries):,}` reproducible corpus-derived probes (one positive source chunk each)
- **Query type:** Extractive synthetic probes for retrieval/load benchmarking; not human-authored QA judgments
- **Query set SHA-256:** `{query_set_sha256}`
- **Candidate Pool ($K$):** `{args.candidate_k}`
- **Final Reranked Context ($K$):** `{args.final_k}`
- **Hypencoder traversal budget:** `{args.entry_points}` entry points, `{args.expansion_candidates}` expansion candidates, `{args.max_iter}` iterations

### 📊 Metric Comparison

| Metric | Bi-Encoder (MiniLM) | Hypencoder (6-Layer q-net) | Gain / Delta |
| :--- | :---: | :---: | :---: |
| **Candidate Recall@{args.candidate_k}** | `{bi_res['candidate_recall']:.4f}` | **`{hyp_res['candidate_recall']:.4f}`** | **{hyp_res['candidate_recall'] - bi_res['candidate_recall']:+.4f}** |
| **Candidate Precision@{args.candidate_k}** | `{bi_res['candidate_precision']:.4f}` | **`{hyp_res['candidate_precision']:.4f}`** | **{hyp_res['candidate_precision'] - bi_res['candidate_precision']:+.4f}** |
| **Reranked Recall@{args.final_k}** | `{bi_res['reranked_recall']:.4f}` | **`{hyp_res['reranked_recall']:.4f}`** | **{hyp_res['reranked_recall'] - bi_res['reranked_recall']:+.4f}** |
| **Reranked Precision@{args.final_k}** | `{bi_res['reranked_precision']:.4f}` | **`{hyp_res['reranked_precision']:.4f}`** | **{hyp_res['reranked_precision'] - bi_res['reranked_precision']:+.4f}** |
| **MRR (Mean Reciprocal Rank)** | `{bi_res['mrr']:.4f}` | **`{hyp_res['mrr']:.4f}`** | **{hyp_res['mrr'] - bi_res['mrr']:+.4f}** |

### ⏱️ Latency (ms)

| Phase | Bi-Encoder (MiniLM + Qdrant) | Hypencoder (q-net + Graph) |
| :--- | :---: | :---: |
| **Search / Traversal** | `{bi_res['latency_ms']['search']:.1f} ms` | **`{hyp_res['latency_ms']['query_encoding_and_traversal']:.1f} ms`** |
| **Cross-Encoder Rerank** | `{bi_res['latency_ms']['rerank']:.1f} ms` | `{hyp_res['latency_ms']['rerank']:.1f} ms` |
| **Total End-to-End Latency** | `{bi_res['latency_ms']['total']:.1f} ms` | **`{hyp_res['latency_ms']['total']:.1f} ms`** |
"""

    print("\n" + summary_md)

    # Save outputs
    output_data = {
        "query_count": len(queries),
        "query_type": "corpus-derived extractive probes; one positive source chunk per query",
        "query_set_sha256": query_set_sha256,
        "hypencoder_random_seed": 0,
        "hypencoder_traversal": {
            "entry_points": args.entry_points,
            "expansion_candidates": args.expansion_candidates,
            "max_iter": args.max_iter,
        },
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "source_corpus_count": src_count,
        "hypencoder_corpus_count": hyp_count,
        "biencoder": bi_res,
        "hypencoder": hyp_res,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved benchmark results to {args.output_json}")

    # Output to GITHUB_STEP_SUMMARY if running in GitHub Actions
    step_summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_file:
        try:
            with open(step_summary_file, "a", encoding="utf-8") as f:
                f.write(summary_md + "\n")
        except Exception as e:
            print(f"[WARN] Could not write to GITHUB_STEP_SUMMARY: {e}")


if __name__ == "__main__":
    main()
