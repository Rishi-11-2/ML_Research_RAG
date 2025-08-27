#!/usr/bin/env python3
"""
Ingest PDFs (via docling DocumentConverter + HybridChunker) -> token-aware subchunks ->
embed with SentenceTransformer -> upsert into Qdrant.

Safe guards included:
 - overlap < chunk_size assertion
 - deterministic step advancement when splitting tokens
 - per-file instrumentation: tokens, estimated chunks, estimated batches
 - optional max_chunks_per_pdf safety limit
 - batched embedding + batched upserts
"""

import os
import gc
import hashlib
import uuid
import time
import math
from typing import List, Dict, Any, Optional

import tiktoken
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException
import httpx

# your docling tools
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker



from dotenv import load_dotenv

# ----------------- CONFIG (tweak these) -----------------
# Local project / doc locations
PDF_DIR = "/kaggle/input/ml-papers/papers"

# Qdrant config (prefer environment overrides)
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "document_chunks")

# Embedding model
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")  # "cuda" if available

# Chunking / token settings
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 450))
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", 75))

# initial char-based chunker settings (keeps your pipeline behavior)
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", 200))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", 600))
CHUNK_CHAR_OVERLAP = int(os.environ.get("CHUNK_CHAR_OVERLAP", 100))

# batching
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 128))   # docs per embed batch
QDRANT_UPSERT_BATCH = int(os.environ.get("QDRANT_UPSERT_BATCH", 512))  # points per upsert

# safety
MAX_CHUNKS_PER_PDF = int(os.environ.get("MAX_CHUNKS_PER_PDF", 200_000))

# tokenizer name for tiktoken
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "cl100k_base")

# -----------------------------------------------------------------

# basic guards
if OVERLAP_TOKENS >= MAX_TOKENS:
    raise ValueError(f"OVERLAP_TOKENS ({OVERLAP_TOKENS}) must be less than MAX_TOKENS ({MAX_TOKENS})")

# ---------- TOKENIZER WRAPPER ----------
class TokenizerWrapper:
    def __init__(self, encoding_name: str = TOKENIZER_NAME):
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids)

    def token_len(self, text: str) -> int:
        return len(self.encode(text))


def token_split_text(text: str, max_tokens: int, overlap_tokens: int, tokenizer: TokenizerWrapper) -> List[str]:
    if not text or not text.strip():
        return []

    ids = tokenizer.encode(text)
    total = len(ids)
    if total <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max(1, max_tokens - overlap_tokens)
    while start < total:
        end = min(start + max_tokens, total)
        token_slice = ids[start:end]
        chunk_text = tokenizer.decode(token_slice).strip()
        if not chunk_text:
            # defensive: advance by step if decode returned empty (rare)
            start += step
            continue
        chunks.append(chunk_text)
        if end == total:
            break
        start += step

    return chunks


# ---------- stable id + metadata helpers ----------
def stable_id(filename: str, index, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(filename))[0]
    return f"{base}::chunk::{index}::{h}"


def extract_pages_from_chunk_meta(chunk_meta) -> str:
    page_nos = set()
    try:
        for doc_item in getattr(chunk_meta, "doc_items", []) or []:
            for prov in getattr(doc_item, "prov", []) or []:
                page_nos.add(int(getattr(prov, "page_no", -1)))
    except Exception:
        pass
    page_nos = sorted([p for p in page_nos if p >= 0])
    return ",".join(map(str, page_nos))


def chunk_metadata(chunk, original_index, sub_index, filename):
    pages = extract_pages_from_chunk_meta(getattr(chunk, "meta", {}))

    headings_val = getattr(getattr(chunk, "meta", None), "headings", None)
    if headings_val is None:
        headings = ""
    elif isinstance(headings_val, (list, tuple, set)):
        headings = ",".join(str(h) for h in headings_val)
    else:
        headings = str(headings_val)

    preview = (chunk.text[:300] + "...") if getattr(chunk, "text", "") and len(chunk.text) > 300 else getattr(chunk, "text", "")
    return {
        "source": os.path.basename(filename),
        "pages": pages,
        "headings": headings,
        "parent_chunk_index": original_index,
        "subchunk_index": sub_index,
        "preview": preview
    }


# ---------- Embedding generator ----------
class EmbeddingGenerator:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE):
        print(f"[INFO] Loading SentenceTransformer '{model_name}' on device='{device}'", flush=True)
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
        if not texts:
            return []
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
        return embs.tolist()


# ---------- Qdrant helpers ----------
def ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int, distance: str = "Cosine"):
    try:
        existing = client.get_collections().collections
        if any(c.name == collection_name for c in existing):
            return
    except Exception:
        # fallback: try to create anyway
        pass

    dist = rest.Distance.COSINE if distance.lower() == "cosine" else (rest.Distance.DOT if distance.lower() == "dot" else rest.Distance.EUCLID)
    print(f"[INFO] Creating collection '{collection_name}' vector_size={vector_size} distance={distance}", flush=True)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=dist),
    )

def qdrant_uuid_from_stable(stable_id_str: str) -> str:
    """
    Deterministically convert stable_id string into a UUID string (UUIDv5).
    This produces a valid Qdrant point id and is repeatable across runs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_id_str))


def upsert_points(client: QdrantClient, collection_name: str, points: List[rest.PointStruct]):
    try:
        client.upsert(collection_name=collection_name, points=points)
    except (httpx.ReadTimeout, httpx.ConnectError, ResponseHandlingException, httpx.TransportError) as e:
        print(f"[ERROR] qdrant upsert transport error: {repr(e)}", flush=True)
        raise
    except Exception as e:
        print(f"[ERROR] unexpected exception during qdrant upsert: {repr(e)}", flush=True)
        raise


def get_existing_ids_in_collection(client: QdrantClient, collection_name: str, limit: int = 10000) -> set:
    """
    Best-effort attempt to gather existing ids via scroll. If scroll is not available or fails, returns empty set.
    """
    ids = set()
    try:
        offset = 0
        while True:
            resp = client.scroll(collection_name=collection_name, limit=limit, offset=offset)
            if not resp or not getattr(resp, "ids", None):
                break
            for _id in resp.ids:
                ids.add(str(_id))
            offset += limit
            if len(resp.ids) < limit:
                break
    except Exception:
        # unable to fetch ids - skip duplicate detection
        pass
    return ids


# ---------- Main ingestion ----------
def main():
    t0 = time.time()

    # find all pdf files in the directory
    pdf_files = []
    if os.path.isdir(PDF_DIR):
        for fname in sorted(os.listdir(PDF_DIR)):
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(PDF_DIR, fname))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}. Nothing to ingest.")
        return

    print(f"Found {len(pdf_files)} PDFs to ingest in {PDF_DIR}.")

    # initialize qdrant client
    prefer_grpc = QDRANT_URL.startswith("grpc://") or QDRANT_URL.startswith("http://localhost")
    client = QdrantClient(url=QDRANT_URL, api_key=(QDRANT_API_KEY.strip() if QDRANT_API_KEY else None), prefer_grpc=prefer_grpc)

    # Prepare local tools
    converter = DocumentConverter()
    chunker = HybridChunker(min_chunk_size=MIN_CHUNK_CHARS, max_chunk_size=MAX_CHUNK_CHARS, overlap=CHUNK_CHAR_OVERLAP)
    tokenizer = TokenizerWrapper(encoding_name=TOKENIZER_NAME)
    emb_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)

    total_new = 0
    existing_ids = get_existing_ids_in_collection(client, QDRANT_COLLECTION)

    for pdf_path in pdf_files:
        print(f"\n--- Processing: {pdf_path} ---", flush=True)
        file_t0 = time.time()
        try:
            dl_doc = converter.convert(pdf_path).document
        except Exception as e:
            print(f"[ERROR] Failed to convert {pdf_path}: {e}", flush=True)
            continue

        initial_chunks = list(chunker.chunk(dl_doc=dl_doc))
        print(f"[INFO] Initial char-chunks: {len(initial_chunks)}", flush=True)

        # Prepare token-subchunks
        sub_texts = []
        sub_metas = []
        sub_ids = []

        # track token counts for instrumentation
        total_tokens = 0
        for orig_idx, chunk in enumerate(initial_chunks):
            text = getattr(chunk, "text", "") or ""
            text = text.strip()
            if not text:
                continue
            ids_len = tokenizer.token_len(text)
            total_tokens += ids_len

            token_subs = token_split_text(text, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, tokenizer=tokenizer)
            for sub_idx, stext in enumerate(token_subs):
                sid = stable_id(pdf_path, f"{orig_idx}.{sub_idx}", stext)
                meta = chunk_metadata(chunk, orig_idx, sub_idx, pdf_path)
                sub_texts.append(stext)
                sub_metas.append(meta)
                sub_ids.append(sid)

            # safety cap
            if len(sub_texts) >= MAX_CHUNKS_PER_PDF:
                print(f"[WARN] reached MAX_CHUNKS_PER_PDF={MAX_CHUNKS_PER_PDF}; stopping further chunking for this file", flush=True)
                break

        est_chunks = len(sub_texts)
        est_batches = math.ceil(est_chunks / QDRANT_UPSERT_BATCH) if QDRANT_UPSERT_BATCH > 0 else 0
        print(f"[INFO] file={os.path.basename(pdf_path)} pages_est_tokens={total_tokens} est_subchunks={est_chunks} est_upsert_batches={est_batches}", flush=True)

        if not sub_texts:
            print("[INFO] no subchunks produced for this file; skipping", flush=True)
            continue

        # embed & upsert in batches
        # ensure collection exists (need vector size) -> compute first embedding batch to know vector size
        first_batch = sub_texts[:EMBED_BATCH_SIZE]
        try:
            first_embs = emb_gen.embed(first_batch, batch_size=EMBED_BATCH_SIZE)
        except Exception as e:
            print(f"[ERROR] Embedding failed for first batch of {pdf_path}: {e}", flush=True)
            continue

        if not first_embs:
            print("[ERROR] First embeddings empty; skipping file", flush=True)
            continue

        emb_dim = len(first_embs[0])
        ensure_qdrant_collection(client, QDRANT_COLLECTION, vector_size=emb_dim, distance="Cosine")

        # We'll upsert using stable ids (so re-run is idempotent: existing ids will be updated)
        # Prepare a generator over indices to process in chunks
        def index_batches(n, batch_size):
            i = 0
            while i < n:
                j = min(n, i + batch_size)
                yield i, j
                i = j

        # If there are many items, process in embed batches and upsert batches
        n = len(sub_texts)
        for i0, i1 in index_batches(n, QDRANT_UPSERT_BATCH):
            chunk_docs = sub_texts[i0:i1]
            chunk_metas = sub_metas[i0:i1]
            chunk_ids = sub_ids[i0:i1]

            # embed in smaller embed-batches to control memory
            embeddings = []
            for e0, e1 in index_batches(len(chunk_docs), EMBED_BATCH_SIZE):
                part = chunk_docs[e0:e1]
                embs = emb_gen.embed(part, batch_size=EMBED_BATCH_SIZE)
                embeddings.extend(embs)

            # create points
            points = []
            for sid, vec, meta, text in zip(chunk_ids, embeddings, chunk_metas, chunk_docs):
                payload = dict(meta)
                payload["filename"] = os.path.basename(pdf_path)
                # keep original stable id in payload for traceability
                payload["_stable_id"] = sid
                # add full raw text to payload
                payload["text"] = text
            
                # convert stable id to a UUID string Qdrant accepts
                point_id = qdrant_uuid_from_stable(sid)
            
                point = rest.PointStruct(id=str(point_id), vector=vec, payload=payload)
                points.append(point)


            # upsert
            try:
                upsert_points(client, QDRANT_COLLECTION, points)
            except Exception as e:
                print(f"[ERROR] upsert failed for batch {i0}-{i1} of {pdf_path}: {e}", flush=True)
                # re-raise if you want to stop on failure; otherwise continue
                raise

            # bookkeeping & memory cleanup
            total_new += len(points)
            del points, embeddings
            gc.collect()
            print(f"[INFO] Upserted items {i0}..{i1} ({len(chunk_docs)} items).", flush=True)

        file_t1 = time.time()
        print(f"[INFO] Finished {os.path.basename(pdf_path)} in {file_t1 - file_t0:.2f}s", flush=True)

    # finished
    t1 = time.time()
    print(f"\nAll done. Total new/updated points processed: {total_new} in {(t1 - t0):.1f}s", flush=True)


if __name__ == "__main__":
    main()