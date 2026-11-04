# ML Research RAG Chatbot

A premium, intelligent chatbot designed to answer questions about Machine Learning research papers. It leverages a robust **Retrieval-Augmented Generation (RAG)** pipeline to provide accurate, context-aware answers grounded in the latest academic literature.

**🚀 Deployed Application:** [https://mlresearch-rag.streamlit.app/](https://mlresearch-rag.streamlit.app/)

---

## 🏗️ High-Level Architecture

The system is built on a modular architecture separating the UI, orchestration, and core RAG logic.

```mermaid
graph TD
    %% Define Styles
    classDef user fill:#FF6B6B,stroke:#fff,stroke-width:2px,color:#fff,font-weight:bold;
    classDef ui fill:#6C63FF,stroke:#fff,stroke-width:2px,color:#fff,font-weight:bold,rx:10,ry:10;
    classDef logic fill:#1A202C,stroke:#00D4FF,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef db fill:#2D3748,stroke:#F6E05E,stroke-width:2px,color:#fff;
    classDef ext fill:#4A5568,stroke:#CBD5E0,stroke-width:2px,color:#fff,stroke-dasharray: 5 5;
    classDef decision fill:#D53F8C,stroke:#fff,stroke-width:2px,color:#fff,rx:5,ry:5;

    User([User]):::user -->|Query| UI["Streamlit UI (app.py)"]:::ui
    UI -->|Handle Message| CM["Context Manager (context_manager.py)"]:::logic
    
    subgraph Orchestration [LangGraph Workflow]
        style Orchestration fill:#232730,stroke:#00D4FF,stroke-width:1px,stroke-dasharray: 5 5,color:#fff
        CM -->|Check History| Sum[Summarizer Node]:::logic
        Sum -->|Decision| Router{Should Retrieve?}:::decision
        Router -->|Yes| Ret[Retriever Node]:::logic
        Router -->|No| End((End)):::logic
        Ret -->|Context| Gen[Generator Node]:::logic
    end
    
    subgraph RAG_Core ["RAG Engine (RAG.py)"]
        style RAG_Core fill:#232730,stroke:#6C63FF,stroke-width:1px,stroke-dasharray: 5 5,color:#fff
        Ret -->|Query| Embed[Embedding Model]:::logic
        Embed -->|Vector| Qdrant[(Qdrant Vector DB)]:::db
        Qdrant -->|Candidates| Rerank[Cross-Encoder Reranker]:::logic
        Rerank -->|Top-K Docs| Gen
    end
    
    Gen -->|Prompt + Context| LLM["LLM (OpenRouter)"]:::ext
    LLM -->|Response| UI
    
    subgraph Ingestion [Background Services]
        style Ingestion fill:#232730,stroke:#A0AEC0,stroke-width:1px,stroke-dasharray: 5 5,color:#fff
        Cron[Weekly Cron Job]:::ext -->|Fetch Papers| Arxiv[ArXiv API]:::ext
        Arxiv -->|Process & Embed| Qdrant
    end
```

---

## 🧩 Detailed Component Overview

### 1. Frontend & User Interface (`app.py` & `style.css`)
- **Framework:** Built with **Streamlit** for a responsive and interactive web application.
- **Design:** Features a **premium, glassmorphism-inspired UI** with:
    - Custom CSS for dark mode, smooth animations, and translucent elements.
    - Mobile-responsive layout ensuring a great experience on all devices.
    - Real-time streaming of LLM responses for a fluid conversational feel.
- **Functionality:**
    - Manages user sessions and chat history.
    - Displays LaTeX formulas correctly for mathematical explanations.
    - Provides a sidebar for triggering manual file ingestion.

### 2. Orchestration & Memory (`context_manager.py`)
- **LangGraph:** Uses a state machine to manage the conversation flow.
- **Smart Memory:**
    - **Short-term Memory:** Keeps the most recent messages for immediate context.
    - **Summarization:** Automatically condenses older parts of the conversation using an LLM to maintain long-term context without exceeding token limits.
- **Token Management:** Dynamically trims prompts and history to fit within the model's context window (default 3000 tokens).
- **Thread Management:** Assigns unique IDs to conversations for persistent state management.

### 3. Core RAG Logic (`RAG.py`)
- **Retrieval:**
    - **Embedding:** Defaults to `sentence-transformers/all-MiniLM-L6-v2`. After migrating the Qdrant data, it can use Google's local `google/embeddinggemma-2` model.
    - **Vector DB:** Queries **Qdrant** to find the most relevant document chunks.
- **Reranking:**
    - **Cross-Encoder:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score the initial candidates. This step significantly improves accuracy by analyzing the deep semantic relationship between the query and the documents.
- **Generation:**
    - **LLM:** Connects to high-performance models (e.g., via OpenRouter) to generate answers.
    - **Citation System:** The system prompt enforces strict evidence-based answering, requiring inline citations (e.g., `[1]`) and a reference list of source files.

### 4. Data Ingestion
- **Automated Pipeline:** A weekly cron job fetches top trending ML papers from ArXiv.
- **Processing:** Papers are chunked, embedded, and stored in the Qdrant vector database.
- **Manual Trigger:** Users can request ingestion of specific files via the UI, which triggers a remote ingestion service.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- API Keys: OpenRouter, Qdrant

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/ml-research-rag.git
   cd ml-research-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   QDRANT_URL=<your_qdrant_url>
   QDRANT_API_KEY=<your_qdrant_api_key>
   QDRANT_COLLECTION=document_chunks
   # Set these only after the migration is verified:
   # EMBEDDING_MODEL=google/embeddinggemma-2
   # QDRANT_COLLECTION_EG2=document_chunks_embeddinggemma2
   OPENROUTER_API_KEY=<your_openrouter_key>
   INGESTION_SERVICE_URL=<ingestion_service_url>
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Performance & Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrieve_k` | 50 | Number of initial candidates fetched from Vector DB |
| `final_k` | 5 | Number of top documents kept after reranking |
| `token_budget` | 3000 | Maximum tokens allowed in the LLM prompt |
| `summary_threshold` | 8 | Number of messages before summarization triggers |

### EmbeddingGemma 2 migration

EmbeddingGemma 2 uses a different vector space from MiniLM. Existing Qdrant vectors must be regenerated; changing only the query model will make retrieval invalid. The included migration command writes to a separate collection, retains point IDs and payloads, checks the final point count, and never modifies or deletes the source collection.

1. Upgrade the project environment from this directory: `./ML_Research_RAG/bin/python -m pip install -r requirements.txt`.
2. Set `QDRANT_COLLECTION_EG2` to a new collection name if the default `<QDRANT_COLLECTION>_embeddinggemma2` is unsuitable.
3. Run `./ML_Research_RAG/bin/python migrate_embeddinggemma2.py` and wait for the final verified count. The migration requires each source point to have a non-empty `text` payload and an unnamed dense vector collection. If interrupted, rerun with `--resume`; it skips destination points whose IDs, payloads, and vector shape are already valid, then verifies the exact source/destination ID sets. The source remains untouched.
4. After validating retrieval, set `EMBEDDING_MODEL=google/embeddinggemma-2` in the app environment. This selects the new collection. Keep the existing `QDRANT_COLLECTION` unchanged as the rollback path.

The EmbeddingGemma 2 collection has already been migrated and verified. The migration workflow has been removed; the migration script remains available for an explicitly requested future migration.

The app uses the model's text and code encoder and 256-dimensional normalized vectors to reduce Qdrant storage and search work. Queries use the `SearchQuery` prompt and corpus chunks use `Document`, as required for asymmetric retrieval. The original MiniLM collection remains available. The separate ingestion service must also write new documents with EmbeddingGemma 2 before switching the app, or newly ingested papers will not appear in the new collection. The venv needs sentence-transformers 6.1 or later for this model; that release line conflicts with the `docling`/`datasets` dependency versions currently present in the same venv, so check `pip check` before using the upgraded environment for other ingestion work.

### Retrieval evaluation

Precision and recall require relevance labels; vector similarity scores alone cannot provide them. Create a UTF-8 JSONL file with one query per line and the relevant Qdrant `_stable_id` values, then run:

```bash
cat > labeled_queries.jsonl <<'EOF'
{"query":"What is U-Net?","relevant_ids":["exact_stable_chunk_id_1","exact_stable_chunk_id_2"]}
EOF
./ML_Research_RAG/bin/python evaluate_retrieval.py labeled_queries.jsonl --only minilm
```

Replace the example IDs with judged relevant chunk IDs. After migration, run the same dataset with `--only eg2` (or omit `--only` to run both). The evaluator reports candidate Recall@50 and Precision@50, reranked Recall@5 and Precision@5, and mean/median/p95 latency split across embedding, Qdrant search, and reranking. It only reads the collections. For a meaningful comparison, label at least 50 representative questions and pool relevant IDs across both models' top results before scoring.

The **Benchmark EmbeddingGemma 2 vs MiniLM** GitHub Actions workflow prepares deterministic corpus-derived probes and compares both verified collections. It defaults to 1,000 probes, reports candidate/reranked precision and recall plus mean, median, and p95 latency, and uploads the query set and JSON metrics. These synthetic passage-derived probes are useful for repeatable load comparisons, but human-judged questions are needed to measure production QA relevance.

### Hypencoder benchmark

The **Benchmark Hypencoder vs Bi-Encoder** workflow assumes the Hypencoder collection is already built in Zilliz; it does not run migration or graph construction. It supports 10,000–200,000 deterministic passage-derived probes from the original Qdrant document collection, each labeled with its source chunk. The full query set is prepared once, then evaluated in 2,000-query shards (100 jobs for 200,000 probes) so a single GitHub-hosted job does not need to process the full run. These synthetic probes are useful for retrieval/load comparison, not human-authored QA accuracy.

The hosted-runner defaults limit Hypencoder graph traversal to 12 entry points, 10 expansion candidates, and 4 iterations; the workflow lets you adjust these values to trade runtime against retrieval quality. Evaluation batches query encoding and reranking, saves checkpoint metrics every batch, and retries once from the saved checkpoints if a hosted job fails or times out. The aggregation job only publishes metrics after every requested query is covered. The completed report includes candidate and reranked precision/recall, MRR, and batch-amortized mean latency. Large runs consume substantial GitHub Actions runner minutes even when shards run in parallel. These metrics use synthetic corpus-derived probes with one labeled source chunk each; use human-judged query labels for production-quality relevance evaluation.

The **Migrate Qdrant to Hypencoder** workflow is the legacy path: it encodes document text from `document_chunks`, then writes Hypencoder vectors and approximate neighbor lists to `document_chunks_hypencoder` in Qdrant. It has a status check and one automatic continuation job after a failed run. It does not write to Zilliz. The separate, manually dispatched **Build Fresh Hypencoder Collection in Zilliz** workflow reads the original `document_chunks` payloads, computes new passage vectors, builds L2 neighbor lists through Zilliz search, and validates every source ID and graph link before its cleanup job deletes the legacy Qdrant Hypencoder collection. It never reads or migrates the old Hypencoder vectors and preserves `document_chunks`. Graph search requests use at most 10 vectors to satisfy the Zilliz/Milvus `nq` limit. If the build stops after embedding, rerun with the same staging collection and build ID; `--resume` skips stored vectors and continues graph construction. Set repository secrets `QDRANT_URL`, `ZILLIZ_URI`, and `ZILLIZ_TOKEN`; `QDRANT_API_KEY` is only needed when the Qdrant endpoint requires it.

The RAG app can use the Zilliz Hypencoder path by setting `RETRIEVER_BACKEND=hypencoder_zilliz`, `ZILLIZ_URI`, `ZILLIZ_TOKEN`, and `ZILLIZ_COLLECTION_HYPENCODER`. The default `qdrant` backend remains unchanged. The Hypencoder benchmark reads its source probes from Qdrant and evaluates the Hypencoder graph in Zilliz.

---

## 🔄 Data Flow Summary

1. **User Input:** The user asks a question in the Streamlit UI.
2. **Context Check:** `ContextManager` loads the conversation history.
3. **Retrieval:** `RAG.py` converts the query to a vector and searches Qdrant.
4. **Refinement:** The Cross-Encoder reranks the results to find the best matches.
5. **Prompting:** A prompt is built with the system instructions, history, and top context chunks.
6. **Generation:** The LLM generates a streamed response with citations.
7. **Display:** The answer is shown to the user, and the conversation state is updated.
