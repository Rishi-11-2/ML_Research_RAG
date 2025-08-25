import os
import hashlib
from typing import List, Tuple
import math
import time
import tiktoken
# embeddings / vectorstore
from sentence_transformers import SentenceTransformer
import chromadb

# your docling tools
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# ---------- CONFIG ----------
STORE_DIR = "./.chromadb_store"
CHROMA_COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"            # "cuda" if available
BATCH_SIZE = 128                    # tune for memory/GPU
# Token-chunk settings (tokens)
MAX_TOKENS = 450                    # target tokens per chunk (good for 512/1024 windows)
OVERLAP_TOKENS = 75                 # token overlap between adjacent chunks
# initial char-based chunker settings (keeps your pipeline behavior)
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 600
CHUNK_CHAR_OVERLAP = 100

# Directory containing PDFs to ingest (new: process all PDFs here)
PDF_DIR = "./papers"
# -----------------------------


import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

# ---------- TOKENIZER UTIL ----------
class TokenizerWrapper:
    def __init__(self, model_for_tokenization: str = "gpt-4o-mini"):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def token_len(self, text: str) -> int:
        return len(self.encode(text))


def token_split_text(text: str, max_tokens: int, overlap_tokens: int, tokenizer: TokenizerWrapper) -> List[str]:
    """
    Split a text string into token-sized chunks with token overlap, returning list of text chunks.
    We operate by encoding entire text to token ids, then slicing token id ranges and decoding back to text.
    If using tiktoken we can decode tokens back to string via enc.decode; for HF fallback we must map token ids to strings:
      - HF's tokenizer.decode works too.
    """
    if not text or not text.strip():
        return []

    # encode to token ids
    ids = tokenizer.encode(text)
    total = len(ids)
    if total <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < total:
        end = min(start + max_tokens, total)
        token_slice = ids[start:end]

        # decode
        chunk_text = tokenizer.enc.decode(token_slice)

        chunks.append(chunk_text)

        if end == total:
            break
        # step forward by (max_tokens - overlap)
        start += (max_tokens - overlap_tokens)

    return chunks

# ---------- STABLE ID / METADATA HELPERS ----------
def stable_id(filename: str, index, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(filename))[0]
    return f"{base}::chunk::{index}::{h}"

def extract_pages_from_chunk_meta(chunk_meta) -> str:
    # preserve earlier approach: chunk.meta.doc_items -> prov.page_no
    page_nos = set()
    try:
        for doc_item in getattr(chunk_meta, "doc_items", []) or []:
            for prov in getattr(doc_item, "prov", []) or []:
                # prov.page_no might be int or string
                page_nos.add(int(getattr(prov, "page_no", -1)))
    except Exception:
        # not guaranteed, return empty
        pass
    page_nos = sorted([p for p in page_nos if p >= 0])
    return ",".join(map(str, page_nos))

def chunk_metadata(chunk, original_index, sub_index, filename):
    pages = extract_pages_from_chunk_meta(getattr(chunk, "meta", {}))

    # Safely handle headings which may be None, a list, a string, or other types
    headings_val = getattr(getattr(chunk, "meta", None), "headings", None)
    if headings_val is None:
        headings = ""
    elif isinstance(headings_val, (list, tuple, set)):
        headings = ",".join(str(h) for h in headings_val)
    else:
        # If it's a single string/int/other, convert to string
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

# ---------- MAIN INGESTION ----------
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

    # Initialize Chroma client and collection once for all files
    print("Initializing Chroma client and collection...")
    client = chromadb.PersistentClient(path=STORE_DIR)
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

    # Best-effort: collect existing ids in the collection to skip duplicates
    existing_ids = set()
    try:
        CHUNK = 500
        for i in range(0, 1):  # one attempt to get all ids if supported
            resp = collection.get()
            existing_batch = resp.get("ids", []) or []
            existing_ids.update(existing_batch)
    except Exception:
        existing_ids = set()

    # Load embedding model once
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device {EMBEDDING_DEVICE}...")
    s2t = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)

    # converter and chunker reused
    converter = DocumentConverter()
    chunker = HybridChunker(min_chunk_size=MIN_CHUNK_CHARS, max_chunk_size=MAX_CHUNK_CHARS, overlap=CHUNK_CHAR_OVERLAP)

    # tokenizer
    tok = TokenizerWrapper(model_for_tokenization="gpt-4o-mini")

    total_new = 0
    for PDF_PATH in pdf_files:
        print(f"\nLoading document with DocumentConverter: {PDF_PATH} ...")
        try:
            dl_doc = converter.convert(PDF_PATH).document
        except Exception as e:
            print(f"Failed to convert {PDF_PATH}: {e}")
            continue

        print("Initial (char-based) chunking with HybridChunker...")
        initial_chunks = list(chunker.chunk(dl_doc=dl_doc))
        print(f"Initial chunk count: {len(initial_chunks)}")

        # prepare list of sub-chunks (token-aware) to ingest for this file
        to_ingest_texts = []
        to_ingest_metadatas = []
        to_ingest_ids = []

        for orig_idx, chunk in enumerate(initial_chunks):
            text = chunk.text.strip()
            if not text:
                continue

            sub_texts = token_split_text(text, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, tokenizer=tok)
            for sub_idx, sub_text in enumerate(sub_texts):
                sid = stable_id(PDF_PATH, f"{orig_idx}.{sub_idx}", sub_text)
                meta = chunk_metadata(chunk, orig_idx, sub_idx, PDF_PATH)
                to_ingest_texts.append(sub_text)
                to_ingest_metadatas.append(meta)
                to_ingest_ids.append(sid)

        print(f"After token-aware splitting for {os.path.basename(PDF_PATH)}, total sub-chunks to consider: {len(to_ingest_texts)}")

        # filter out already-present ids if possible (avoid duplicates)
        filtered_texts, filtered_meta, filtered_ids = [], [], []
        for doc, meta, _id in zip(to_ingest_texts, to_ingest_metadatas, to_ingest_ids):
            if _id in existing_ids:
                continue
            filtered_texts.append(doc)
            filtered_meta.append(meta)
            filtered_ids.append(_id)

        print(f"Ingesting {len(filtered_texts)} new sub-chunks for {os.path.basename(PDF_PATH)} (skipped {len(to_ingest_texts)-len(filtered_texts)} duplicates).")

        if not filtered_texts:
            print("No new items to add for this file. Continuing to next PDF.")
            continue

        # batch-add with local embeddings
        def batched_add(docs, metas, ids, batch_size=BATCH_SIZE):
            for i in range(0, len(docs), batch_size):
                sub_docs = docs[i:i+batch_size]
                sub_metas = metas[i:i+batch_size]
                sub_ids = ids[i:i+batch_size]
                embs = s2t.encode(sub_docs, show_progress_bar=False, convert_to_numpy=True)
                collection.add(documents=sub_docs, metadatas=sub_metas, ids=sub_ids, embeddings=embs.tolist())
                print(f"Added batch {i//batch_size + 1} ({len(sub_docs)} items)")

        batched_add(filtered_texts, filtered_meta, filtered_ids)
        # update existing_ids to avoid re-adding across files
        existing_ids.update(filtered_ids)
        total_new += len(filtered_texts)

    print("Persisting (Chroma should persist automatically when using PersistentClient).")
    try:
        client.persist()
    except Exception:
        pass

    t1 = time.time()
    print(f"\nIngestion completed in {(t1-t0):.1f}s. Total new items added across all PDFs: {total_new}")

if __name__ == "__main__":
    main()
