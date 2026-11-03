"""Compare Hypencoder and MiniLM with reproducible corpus-derived query probes.

Builds at least 10,000 unique extractive queries from original corpus passages,
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
import time
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv
import numpy as np
from pymilvus import MilvusClient
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

from hypencoder_retriever import HypencoderZillizGraphRetriever, load_hypencoder

MINIMUM_QUERY_COUNT = 10_000
QUERY_SCROLL_PAGE_SIZE = 1000
METRIC_SUM_FIELDS = (
    "candidate_recall",
    "candidate_precision",
    "reranked_recall",
    "reranked_precision",
    "mrr",
)


def _new_phase_state(latency_fields: List[str]) -> Dict[str, Any]:
    return {
        "completed_queries": 0,
        "metric_sums": {key: 0.0 for key in METRIC_SUM_FIELDS},
        "latency_sums": {key: 0.0 for key in latency_fields},
    }


def _record_query(
    state: Dict[str, Any],
    candidate_metrics: Tuple[float, float],
    reranked_metrics: Tuple[float, float, float],
    latencies_ms: Dict[str, float],
) -> None:
    candidate_precision, candidate_recall = candidate_metrics
    reranked_precision, reranked_recall, reciprocal_rank = reranked_metrics
    state["metric_sums"]["candidate_precision"] += candidate_precision
    state["metric_sums"]["candidate_recall"] += candidate_recall
    state["metric_sums"]["reranked_precision"] += reranked_precision
    state["metric_sums"]["reranked_recall"] += reranked_recall
    state["metric_sums"]["mrr"] += reciprocal_rank
    for key, value in latencies_ms.items():
        state["latency_sums"][key] += value
    state["completed_queries"] += 1


def _phase_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    count = int(state["completed_queries"])
    if count < 1:
        raise RuntimeError("Benchmark phase has no completed queries.")
    return {
        **{key: value / count for key, value in state["metric_sums"].items()},
        "latency_ms": {key: value / count for key, value in state["latency_sums"].items()},
        "queries_completed": count,
    }


def _query_seed(query: str) -> int:
    return int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "big")


def _write_checkpoint(path: str, checkpoint: Dict[str, Any]) -> None:
    checkpoint["partial_metrics"] = {
        phase: _phase_summary(checkpoint[phase])
        for phase in ("biencoder", "hypencoder")
        if checkpoint.get(phase, {}).get("completed_queries", 0) > 0
    }
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output:
        json.dump(checkpoint, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


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


def evaluate_biencoder(
    client,
    collection,
    minilm,
    reranker,
    queries,
    candidate_k,
    final_k,
    state,
    batch_size,
    save_checkpoint,
):
    for start in range(state["completed_queries"], len(queries), batch_size):
        batch = queries[start : start + batch_size]
        batch_count = len(batch)
        t0 = time.perf_counter()
        query_vectors = minilm.encode(
            [item["query"] for item in batch],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        t1 = time.perf_counter()

        requests = [
            qdrant_models.QueryRequest(
                query=vector.tolist(),
                limit=candidate_k,
                with_payload=True,
            )
            for vector in query_vectors
        ]
        responses = client.query_batch_points(collection_name=collection, requests=requests)
        if len(responses) != batch_count:
            raise RuntimeError(f"Qdrant returned {len(responses)} results for {batch_count} batched queries.")
        t2 = time.perf_counter()

        batch_candidates = []
        all_pairs = []
        pair_ranges = []
        candidate_metrics = []
        for item, response in zip(batch, responses):
            points = getattr(response, "points", None) or []
            relevant = set(item["relevant_ids"])
            candidate_ids = [str((point.payload or {}).get("_stable_id") or point.id) for point in points]
            cand_precision, cand_recall, _ = compute_metrics(candidate_ids, relevant, candidate_k)
            candidate_metrics.append((cand_precision, cand_recall))
            pair_start = len(all_pairs)
            all_pairs.extend([[item["query"], (point.payload or {}).get("text", "")] for point in points])
            pair_ranges.append((pair_start, len(all_pairs)))
            batch_candidates.append(points)

        scores = reranker.predict(all_pairs, batch_size=64, show_progress_bar=False).tolist() if all_pairs else []
        t3 = time.perf_counter()
        encode_ms = (t1 - t0) * 1000 / batch_count
        search_ms = (t2 - t1) * 1000 / batch_count
        rerank_ms = (t3 - t2) * 1000 / batch_count

        for item, points, (cand_precision, cand_recall), (pair_start, pair_end) in zip(
            batch, batch_candidates, candidate_metrics, pair_ranges
        ):
            point_scores = scores[pair_start:pair_end]
            ranked_points = [
                point
                for point, _ in sorted(zip(points, point_scores), key=lambda candidate: candidate[1], reverse=True)
            ]
            ranked_ids = [str((point.payload or {}).get("_stable_id") or point.id) for point in ranked_points]
            final_precision, final_recall, reciprocal_rank = compute_metrics(
                ranked_ids, set(item["relevant_ids"]), final_k
            )
            _record_query(
                state,
                (cand_precision, cand_recall),
                (final_precision, final_recall, reciprocal_rank),
                {
                    "query_encoding": encode_ms,
                    "search": search_ms,
                    "rerank": rerank_ms,
                    "total": encode_ms + search_ms + rerank_ms,
                },
            )

        save_checkpoint(state)
        summary = _phase_summary(state)
        print(
            f"Bi-encoder queries: {state['completed_queries']}/{len(queries)}; "
            f"batch-amortized mean end-to-end {summary['latency_ms']['total']:.1f} ms/query",
            flush=True,
        )

    return _phase_summary(state)


def evaluate_hypencoder(
    retriever,
    reranker,
    queries,
    candidate_k,
    final_k,
    entry_points=12,
    expansion_candidates=10,
    max_iter=4,
    state=None,
    batch_size=64,
    save_checkpoint=None,
):
    if state is None:
        state = _new_phase_state(["query_encoding_and_traversal", "rerank", "total"])
    if save_checkpoint is None:
        save_checkpoint = lambda _: None

    for start in range(state["completed_queries"], len(queries), batch_size):
        batch = queries[start : start + batch_size]
        batch_candidates = []
        traversal_times = []
        candidate_metrics = []
        all_pairs = []
        pair_ranges = []

        for item in batch:
            query = item["query"]
            t0 = time.perf_counter()
            candidates = retriever.retrieve(
                query=query,
                k=candidate_k,
                num_entry_points=min(entry_points, len(retriever._all_known_ids)),
                ncandidates=expansion_candidates,
                max_iter=max_iter,
                random_seed=_query_seed(query),
            )
            traversal_times.append((time.perf_counter() - t0) * 1000)
            cand_ids = [str(candidate.get("id")) for candidate in candidates]
            cand_precision, cand_recall, _ = compute_metrics(
                cand_ids, set(item["relevant_ids"]), candidate_k
            )
            candidate_metrics.append((cand_precision, cand_recall))
            pair_start = len(all_pairs)
            all_pairs.extend([[query, candidate.get("text", "")] for candidate in candidates])
            pair_ranges.append((pair_start, len(all_pairs)))
            batch_candidates.append(candidates)

        rerank_started = time.perf_counter()
        scores = reranker.predict(all_pairs, batch_size=64, show_progress_bar=False).tolist() if all_pairs else []
        rerank_ms = (time.perf_counter() - rerank_started) * 1000 / len(batch)

        for item, candidates, (cand_precision, cand_recall), (pair_start, pair_end), traversal_ms in zip(
            batch, batch_candidates, candidate_metrics, pair_ranges, traversal_times
        ):
            point_scores = scores[pair_start:pair_end]
            ranked_candidates = [
                candidate
                for candidate, _ in sorted(zip(candidates, point_scores), key=lambda row: row[1], reverse=True)
            ]
            ranked_ids = [str(candidate.get("id")) for candidate in ranked_candidates]
            final_precision, final_recall, reciprocal_rank = compute_metrics(
                ranked_ids, set(item["relevant_ids"]), final_k
            )
            _record_query(
                state,
                (cand_precision, cand_recall),
                (final_precision, final_recall, reciprocal_rank),
                {
                    "query_encoding_and_traversal": traversal_ms,
                    "rerank": rerank_ms,
                    "total": traversal_ms + rerank_ms,
                },
            )

        save_checkpoint(state)
        summary = _phase_summary(state)
        print(
            f"Hypencoder queries: {state['completed_queries']}/{len(queries)}; "
            f"mean traversal {summary['latency_ms']['query_encoding_and_traversal']:.1f} ms/query",
            flush=True,
        )

    return _phase_summary(state)


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
    parser.add_argument("--checkpoint-json", default="benchmark_checkpoint.json")
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Resume from a compatible checkpoint")
    args = parser.parse_args()
    if args.query_count < MINIMUM_QUERY_COUNT:
        parser.error(f"--query-count must be at least {MINIMUM_QUERY_COUNT}")
    if args.candidate_k < 1 or args.final_k < 1:
        parser.error("--candidate-k and --final-k must be positive")
    if args.entry_points < 1 or args.expansion_candidates < 1 or args.max_iter < 1:
        parser.error("--entry-points, --expansion-candidates, and --max-iter must be positive")
    if args.evaluation_batch_size < 1:
        parser.error("--evaluation-batch-size must be positive")

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL must be configured")
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_token = os.getenv("ZILLIZ_TOKEN")
    if not zilliz_uri or not zilliz_token:
        raise RuntimeError("ZILLIZ_URI and ZILLIZ_TOKEN must be configured")

    src_coll = os.getenv("QDRANT_COLLECTION", "document_chunks")
    hyp_coll = os.getenv("ZILLIZ_COLLECTION_HYPENCODER", "document_chunks_hypencoder_zilliz_v2")

    client = QdrantClient(url=url, api_key=api_key.strip() if api_key else None, timeout=60.0, prefer_grpc=False)
    src_count = client.count(src_coll, exact=True).count if client.collection_exists(src_coll) else 0
    milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    if not milvus.has_collection(hyp_coll):
        milvus.close()
        raise RuntimeError(f"Zilliz Hypencoder collection {hyp_coll!r} does not exist")
    hyp_stats = milvus.get_collection_stats(collection_name=hyp_coll)
    try:
        hyp_count = int(hyp_stats["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        milvus.close()
        raise RuntimeError(f"Could not read Zilliz row count for {hyp_coll!r}: {hyp_stats}") from exc

    print(f"Total source corpus size: {src_count} chunks")
    print(f"Zilliz Hypencoder collection size: {hyp_count} chunks")
    if hyp_count != src_count:
        milvus.close()
        raise RuntimeError(
            f"This benchmark requires matching corpora; Qdrant has {src_count} points and Zilliz has {hyp_count}."
        )
    if hyp_count < args.query_count:
        milvus.close()
        raise RuntimeError(f"Need at least {args.query_count} Zilliz Hypencoder points; found {hyp_count}.")

    print(f"\nBuilding {args.query_count:,} deterministic query probes from Hypencoder passages...")
    queries = build_benchmark_queries(client, src_coll, args.query_count)
    query_set_sha256 = hashlib.sha256(
        "\n".join(
            json.dumps(query, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for query in queries
        ).encode("utf-8")
    ).hexdigest()
    print(f"Built {len(queries):,} unique corpus-derived queries.")

    checkpoint_config = {
        "schema_version": 1,
        "query_set_sha256": query_set_sha256,
        "query_count": len(queries),
        "source_count": src_count,
        "hypencoder_count": hyp_count,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "entry_points": args.entry_points,
        "expansion_candidates": args.expansion_candidates,
        "max_iter": args.max_iter,
        "evaluation_batch_size": args.evaluation_batch_size,
    }
    if args.resume and os.path.exists(args.checkpoint_json):
        with open(args.checkpoint_json, encoding="utf-8") as checkpoint_file:
            checkpoint_data = json.load(checkpoint_file)
        if checkpoint_data.get("config") != checkpoint_config:
            raise RuntimeError(
                "Benchmark checkpoint settings or query corpus do not match this run; "
                "use a fresh checkpoint file or restore the matching inputs."
            )
        print("Resuming compatible benchmark checkpoint.", flush=True)
    else:
        checkpoint_data = {"config": checkpoint_config}

    biencoder_state = checkpoint_data.get("biencoder") or _new_phase_state(
        ["query_encoding", "search", "rerank", "total"]
    )
    hypencoder_state = checkpoint_data.get("hypencoder") or _new_phase_state(
        ["query_encoding_and_traversal", "rerank", "total"]
    )

    def save_phase(phase_name):
        def save(state):
            checkpoint_data[phase_name] = state
            checkpoint_data["status"] = "in_progress"
            checkpoint_data["updated_at_unix"] = time.time()
            _write_checkpoint(args.checkpoint_json, checkpoint_data)

        return save

    _write_checkpoint(args.checkpoint_json, checkpoint_data)

    need_biencoder = biencoder_state["completed_queries"] < len(queries)
    need_hypencoder = hypencoder_state["completed_queries"] < len(queries)
    print("\nLoading models for benchmark...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if need_biencoder or need_hypencoder else None
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") if need_biencoder else None

    print("\n--- Evaluating Bi-Encoder (MiniLM) on Qdrant ---")
    bi_res = evaluate_biencoder(
        client=client,
        collection=src_coll,
        minilm=minilm,
        reranker=reranker,
        queries=queries,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        state=biencoder_state,
        batch_size=args.evaluation_batch_size,
        save_checkpoint=save_phase("biencoder"),
    )

    print("\n--- Evaluating Hypencoder (q-net + Graph Traversal) ---")
    print(
        "Traversal budget: "
        f"{args.entry_points} entry points, {args.expansion_candidates} expansion candidates, "
        f"{args.max_iter} iterations",
        flush=True,
    )
    if need_hypencoder:
        model, tokenizer = load_hypencoder()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        retriever = HypencoderZillizGraphRetriever(
            client=milvus,
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
            state=hypencoder_state,
            batch_size=args.evaluation_batch_size,
            save_checkpoint=save_phase("hypencoder"),
        )
    else:
        hyp_res = _phase_summary(hypencoder_state)

    summary_md = f"""# 🚀 End-to-End Retrieval Benchmark Results

- **Corpus evaluated:** `{hyp_count}` Hypencoder chunks vs `{src_count}` source chunks
- **Test queries:** `{len(queries):,}` reproducible corpus-derived probes (one positive source chunk each)
- **Query type:** Extractive synthetic probes for retrieval/load benchmarking; not human-authored QA judgments
- **Query set SHA-256:** `{query_set_sha256}`
- **Candidate Pool ($K$):** `{args.candidate_k}`
- **Final Reranked Context ($K$):** `{args.final_k}`
- **Hypencoder traversal budget:** `{args.entry_points}` entry points, `{args.expansion_candidates}` expansion candidates, `{args.max_iter}` iterations
- **Evaluation batch size:** `{args.evaluation_batch_size}` queries
- **Latency interpretation:** Batch-amortized mean ms/query; graph traversal is measured per query

### 📊 Metric Comparison

| Metric | Bi-Encoder (MiniLM) | Hypencoder (6-Layer q-net) | Gain / Delta |
| :--- | :---: | :---: | :---: |
| **Candidate Recall@{args.candidate_k}** | `{bi_res['candidate_recall']:.4f}` | **`{hyp_res['candidate_recall']:.4f}`** | **{hyp_res['candidate_recall'] - bi_res['candidate_recall']:+.4f}** |
| **Candidate Precision@{args.candidate_k}** | `{bi_res['candidate_precision']:.4f}` | **`{hyp_res['candidate_precision']:.4f}`** | **{hyp_res['candidate_precision'] - bi_res['candidate_precision']:+.4f}** |
| **Reranked Recall@{args.final_k}** | `{bi_res['reranked_recall']:.4f}` | **`{hyp_res['reranked_recall']:.4f}`** | **{hyp_res['reranked_recall'] - bi_res['reranked_recall']:+.4f}** |
| **Reranked Precision@{args.final_k}** | `{bi_res['reranked_precision']:.4f}` | **`{hyp_res['reranked_precision']:.4f}`** | **{hyp_res['reranked_precision'] - bi_res['reranked_precision']:+.4f}** |
| **MRR (Mean Reciprocal Rank)** | `{bi_res['mrr']:.4f}` | **`{hyp_res['mrr']:.4f}`** | **{hyp_res['mrr'] - bi_res['mrr']:+.4f}** |

### ⏱️ Batch-amortized Latency (ms/query)

| Phase | Bi-Encoder (MiniLM + Qdrant) | Hypencoder (q-net + Graph) |
| :--- | :---: | :---: |
| **Search / Traversal** | `{bi_res['latency_ms']['search']:.1f} ms` | **`{hyp_res['latency_ms']['query_encoding_and_traversal']:.1f} ms`** |
| **Cross-Encoder Rerank** | `{bi_res['latency_ms']['rerank']:.1f} ms` | `{hyp_res['latency_ms']['rerank']:.1f} ms` |
| **Total End-to-End Latency** | `{bi_res['latency_ms']['total']:.1f} ms` | **`{hyp_res['latency_ms']['total']:.1f} ms`** |
"""

    print("\n" + summary_md)

    # Save outputs
    output_data = {
        "status": "complete",
        "query_count": len(queries),
        "query_type": "corpus-derived extractive probes; one positive source chunk per query",
        "query_set_sha256": query_set_sha256,
        "evaluation_batch_size": args.evaluation_batch_size,
        "latency_interpretation": "Bi-encoder components and reranking are batch-amortized mean ms/query; Hypencoder traversal is measured per query.",
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
    checkpoint_data["biencoder"] = biencoder_state
    checkpoint_data["hypencoder"] = hypencoder_state
    checkpoint_data["status"] = "complete"
    checkpoint_data["updated_at_unix"] = time.time()
    _write_checkpoint(args.checkpoint_json, checkpoint_data)
    print(f"Saved benchmark results to {args.output_json}")

    # Output to GITHUB_STEP_SUMMARY if running in GitHub Actions
    step_summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_file:
        try:
            with open(step_summary_file, "a", encoding="utf-8") as f:
                f.write(summary_md + "\n")
        except Exception as e:
            print(f"[WARN] Could not write to GITHUB_STEP_SUMMARY: {e}")

    milvus.close()


if __name__ == "__main__":
    main()
