"""Large-scale end-to-end benchmark comparing Hypencoder and Bi-Encoder (MiniLM).

Runs over an expanded corpus (e.g., 1,000+ chunks) with ground-truth labeled queries
from across the research paper corpus, reporting Recall@K, Precision@K, MRR,
and latency breakdown. Generates GitHub Step Summary if running in GitHub Actions.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

from hypencoder_retriever import HypencoderGraphRetriever, load_hypencoder

BENCHMARK_QUERIES = [
    {
        "query": "Which technical report introduced Torch modular machine learning software library?",
        "relevant_ids": ["The need for open source software in machine learning.pdf::84.0::88dccca4e580"],
    },
    {
        "query": "How is reinforcement learning theory formalized using the Robbins-Siegmund approach?",
        "relevant_ids": ["Towards Formalizing Reinforcement Learning Theory_ A Robbins-Siegmund Approach.pdf::49.0::194c2aab03fc"],
    },
    {
        "query": "What was the goal of the FP7 ACROPOLIS project in cognitive radio spectrum sharing?",
        "relevant_ids": ["Reinforcement Learning-based Spectrum Sharing for Cognitive Radio.pdf::158.0::37636ed8d7e7"],
    },
    {
        "query": "How does agentic artificial intelligence socialization evaluate task division and trust building?",
        "relevant_ids": ["Beauty or the Borg_Agentic Artificial Intelligence Organizational Socialization in Synergistic Hybrid Transformative Dynamic Flows.pdf::73.0::d33f41c9b88e"],
    },
    {
        "query": "How does generalized Earley parser bridge symbolic grammars and sequence data for prediction?",
        "relevant_ids": ["Generalized Earley Parser_ Bridging Symbolic Grammars and Sequence Data for Future Prediction::chunk::12.0::ece1090ed768"],
    },
    {
        "query": "How do facial-preserving diffusion models evaluate custom graffiti art image editing?",
        "relevant_ids": ["CraftGraffiti-ExploringHumanIdentitywithCustomGraffitiArtviaFacial-PreservingDiffusionModels::chunk::26.0::dbe42a6be15f"],
    },
    {
        "query": "How is multi-source retrieval-augmented generation evaluated with BM25 and Llama models?",
        "relevant_ids": ["MSRS-EvaluatingMulti-SourceRetrieval-AugmentedGeneration::chunk::188.0::828f051b25b7"],
    },
    {
        "query": "How is the full model selection problem framed in meta-learning across winequality and datasets?",
        "relevant_ids": ["Meta-Learning and the Full Model Selection Problem.pdf::236.1::86e7913a6368"],
    },
]


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

    for item in queries:
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


def evaluate_hypencoder(retriever, reranker, queries, candidate_k, final_k):
    traversal_ms, rerank_ms, total_ms = [], [], []
    cand_p_list, cand_r_list = [], []
    final_p_list, final_r_list, mrr_list = [], [], []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        candidates = retriever.retrieve(
            query=query,
            k=candidate_k,
            num_entry_points=min(50, len(retriever._all_known_ids)),
            ncandidates=25,
            max_iter=16,
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
    parser.add_argument("--output-json", default="benchmark_results.json", help="Path to save results JSON")
    args = parser.parse_args()

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

    print("\nLoading models for benchmark...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n--- Evaluating Bi-Encoder (MiniLM) on Qdrant ---")
    bi_res = evaluate_biencoder(
        client=client,
        collection=src_coll,
        minilm=minilm,
        reranker=reranker,
        queries=BENCHMARK_QUERIES,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
    )

    print("\n--- Evaluating Hypencoder (q-net + Graph Traversal) ---")
    model, tokenizer = load_hypencoder()
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
        queries=BENCHMARK_QUERIES,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
    )

    summary_md = f"""# 🚀 End-to-End Retrieval Benchmark Results

- **Corpus evaluated:** `{hyp_count}` Hypencoder chunks vs `{src_count}` source chunks
- **Test queries:** `{len(BENCHMARK_QUERIES)}` domain-specific ML research questions
- **Candidate Pool ($K$):** `{args.candidate_k}`
- **Final Reranked Context ($K$):** `{args.final_k}`

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
