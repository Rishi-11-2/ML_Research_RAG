"""Benchmark comparison between the Existing Bi-Encoder (MiniLM) and Hypencoder (q-net).

Evaluates both approaches on:
1. Recall@K and Precision@K (candidate retrieval phase)
2. Recall@K, Precision@K, and MRR (final reranked phase)
3. Latency breakdown (Query Encoding, Index Search / Graph Traversal, Reranking, Total)
"""

import json
import os
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
    """Returns (Precision@k, Recall@k, ReciprocalRank)."""
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


def evaluate_biencoder(client, collection, minilm_model, reranker, queries, candidate_k=15, final_k=3):
    q_encode_ms, search_ms, rerank_ms, total_ms = [], [], [], []
    cand_p_list, cand_r_list = [], []
    final_p_list, final_r_list, mrr_list = [], [], []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        query_vec = minilm_model.encode(query, normalize_embeddings=True)
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

        # Cross-encoder rerank
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
        "candidate_k": candidate_k,
        "final_k": final_k,
        "candidate_recall": statistics.mean(cand_r_list),
        "candidate_precision": statistics.mean(cand_p_list),
        "reranked_recall": statistics.mean(final_r_list),
        "reranked_precision": statistics.mean(final_p_list),
        "mrr": statistics.mean(mrr_list),
        "latency_ms": {
            "query_encoding": statistics.mean(q_encode_ms),
            "search_or_traversal": statistics.mean(search_ms),
            "rerank": statistics.mean(rerank_ms),
            "total": statistics.mean(total_ms),
        },
    }


def evaluate_hypencoder(retriever, reranker, queries, candidate_k=15, final_k=3):
    q_encode_ms, traversal_ms, rerank_ms, total_ms = [], [], [], []
    cand_p_list, cand_r_list = [], []
    final_p_list, final_r_list, mrr_list = [], [], []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        # Measure retrieval using Hypencoder graph retriever
        candidates = retriever.retrieve(query=query, k=candidate_k, num_entry_points=25, ncandidates=15, max_iter=8)
        t1 = time.perf_counter()

        cand_ids = [str(c.get("id")) for c in candidates]
        cand_p, cand_r, _ = compute_metrics(cand_ids, relevant, candidate_k)

        # Cross-encoder rerank
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
        "candidate_k": candidate_k,
        "final_k": final_k,
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
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None, prefer_grpc=False, timeout=30.0)

    print("Loading reranker model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("\n--- 1. Evaluating Existing Bi-Encoder (MiniLM + Qdrant Similarity Search) ---")
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    biencoder_results = evaluate_biencoder(
        client=client,
        collection="document_chunks",  # Source 384-dim MiniLM collection
        minilm_model=minilm,
        reranker=reranker,
        queries=BENCHMARK_QUERIES,
        candidate_k=15,
        final_k=3,
    )

    print("\n--- 2. Evaluating Hypencoder Approach (q-net + Graph Traversal) ---")
    model, tokenizer = load_hypencoder()
    retriever = HypencoderGraphRetriever(
        client=client,
        collection_name="document_chunks_hypencoder",
        model=model,
        tokenizer=tokenizer,
        preload_limit=500,
    )
    hypencoder_results = evaluate_hypencoder(
        retriever=retriever,
        reranker=reranker,
        queries=BENCHMARK_QUERIES,
        candidate_k=15,
        final_k=3,
    )

    print("\n" + "=" * 60)
    print("               BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of test queries: {len(BENCHMARK_QUERIES)}")
    print("\n[Metric Comparison]")
    print(f"{'Metric':<28} | {'Bi-Encoder (MiniLM)':<20} | {'Hypencoder':<20} | {'Gain / Delta':<15}")
    print("-" * 90)

    metrics_to_compare = [
        ("Candidate Recall@15", biencoder_results["candidate_recall"], hypencoder_results["candidate_recall"]),
        ("Candidate Precision@15", biencoder_results["candidate_precision"], hypencoder_results["candidate_precision"]),
        ("Reranked Recall@3", biencoder_results["reranked_recall"], hypencoder_results["reranked_recall"]),
        ("Reranked Precision@3", biencoder_results["reranked_precision"], hypencoder_results["reranked_precision"]),
        ("MRR (Mean Reciprocal Rank)", biencoder_results["mrr"], hypencoder_results["mrr"]),
    ]

    for name, b_val, h_val in metrics_to_compare:
        gain = h_val - b_val
        gain_str = f"{gain:+.4f} ({gain/b_val*100:+.1f}%)" if b_val > 0 else f"{gain:+.4f}"
        print(f"{name:<28} | {b_val:<20.4f} | {h_val:<20.4f} | {gain_str:<15}")

    print("\n[Latency Breakdown (ms)]")
    print(f"Bi-Encoder Total Latency:  {biencoder_results['latency_ms']['total']:.2f} ms")
    print(f"  - Query Encoding:        {biencoder_results['latency_ms']['query_encoding']:.2f} ms")
    print(f"  - Qdrant Search:         {biencoder_results['latency_ms']['search_or_traversal']:.2f} ms")
    print(f"  - Cross-Encoder Rerank:  {biencoder_results['latency_ms']['rerank']:.2f} ms")
    print(f"\nHypencoder Total Latency:  {hypencoder_results['latency_ms']['total']:.2f} ms")
    print(f"  - Q-Net & Traversal:     {hypencoder_results['latency_ms']['query_encoding_and_traversal']:.2f} ms")
    print(f"  - Cross-Encoder Rerank:  {hypencoder_results['latency_ms']['rerank']:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
