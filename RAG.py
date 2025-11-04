import os
import numpy as np
from typing import List, Tuple
import re
# import chromadb
import numpy as np

from dotenv import load_dotenv
from litellm import completion
from sentence_transformers import SentenceTransformer, CrossEncoder , util
from qdrant_client import QdrantClient , models

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "document_chunks")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

#--------------------------
# -----------------------
hf_token = os.getenv("HF_TOKEN")
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENROUTER_API_KEY"] = OPEN_ROUTER_API_KEY

# client = chromadb.PersistentClient(path="./.chromadb_store")
# collection = client.get_collection(name="document_chunks")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



# -----------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a and b are 1D arrays
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: 
        return 0.0
    return float(np.dot(a, b) / denom)

#-------------------------

def cross_encoder_rerank(query: str, docs: List, top_k: int = 5) -> List[dict]:
    """
    Accepts either:
      - docs: List[str]
      - docs: List[dict] where each dict has a 'text' key (as returned by call_vector_db)
    Returns list of dicts: {"id": None, "text": <doc string>, "score": <float>, "metadata": {}}
    """
    # Normalize texts
    texts = []
    for d in docs:
        if isinstance(d, dict):
            texts.append(d.get("text", ""))
        else:
            texts.append(str(d))

    if not texts:
        return []

    pairs = [[query, t] for t in texts]
    scores = cross_encoder.predict(pairs)  # numpy array or list of floats
    ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for i, (doc_text, score) in enumerate(ranked):
        results.append({
            "id": None,
            "text": doc_text,
            "score": float(score) if score is not None else None,
            "metadata": {}
        })
    return results


def call_vector_db(query: str, k: int) -> List[dict]:
    """
    

    uses the dense and sparse vectors, with their results combined
    using the Reciprocal Rank Fusion.
    """
    query_vec = embedder.encode([query])[0]
    client = QdrantClient(url=QDRANT_URL, api_key=(QDRANT_API_KEY.strip() ), prefer_grpc=False)
    # Search Qdrant
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec, 
        limit=k,
        with_vectors=False,
        # using = "late-interaction",
        # prefetch=sparse_dene_rrf_prefetch,
    )
    # print(results)
    out = []

    points_list = getattr(results, "points", None) or results

    for hit in points_list:
        # --- support object-like hits (ScoredPoint) ---
        payload = getattr(hit, "payload", None) or {}
        score = getattr(hit, "score", None)
        pid = getattr(hit, "id", None)

        text = payload.get("text")
        meta = {k: v for k, v in payload.items() if k not in ("text", "preview", "source", "_stable_id")}

        source = payload.get("source")
        score_f = float(score) if score is not None else None


        pid_str = payload.get("_stable_id")

        # print(pid_str, meta,score_f)
        out.append({
            "id": pid_str,
            "source":source,
            "text": text,
            "score": score_f,
            "metadata": meta,
        })

    return out

#---------------------------


def llm_call(prompt: str) -> str:
    messages = [
    {"role": "system", "content": "You are an expert assistant. Use ONLY the provided context for factual claims and cite sources. But don't say 'based on the context'. Give  a detailed answer based on the provided context ."},
    {"role": "user", "content": prompt}
    ]
    resp = completion(
        model="openrouter/z-ai/glm-4.5-air:free",
        messages=messages,
    )
    return resp.choices[0].message.content

#------------------------------


#JUST FOR Testing 
def rag_query(query: str, final_k: int = 3, candidate_k: int = 50) -> str:
    """
    1) get candidate_k candidates from Chroma
    2) rerank them (embedding or cross-encoder)
    3) build context using top final_k
    4) send to LLM
    """
    # fetch many candidates from chroma/qdrant (bigger pool -> better reranking)
    candidates = call_vector_db(query, candidate_k)
    if not candidates:
        raise ValueError("No documents returned from Chroma DB")

    # rerank
    # print(candidat)
    reranked = cross_encoder_rerank(query, candidates, top_k=final_k)

    # Build context text -> preserve ordering and include score for debugging if wanted
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(reranked))

   
    context = f"context:{context}, query : {query}"
    resp = llm_call(context)
    return resp

if __name__ == "__main__":
    # q = "How can we explain the whole process of FDB mathematically?"
    q = "what is U-NET?"
    print(rag_query(q, final_k=3, candidate_k=50))
