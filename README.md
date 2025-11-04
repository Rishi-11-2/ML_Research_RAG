# ML Chatbot with RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** chatbot built with Streamlit that combines vector search, cross-encoder reranking, and LLM inference to deliver accurate, context-grounded answers.

## Architecture

### Components

| File | Purpose |
|------|---------|
| **RAG.py** | Core RAG logic: vector DB integration, embedding, reranking, LLM calls |
| **context_manager.py** | LangGraph-based orchestration: conversation memory, context management, message summarization |
| **app.py** | Streamlit UI: chat interface, streaming responses, persistent thread management |

---

## Core Features

### 1. **RAG.py** - Retrieval & Reranking
- **`call_vector_db(query, k)`** → Queries **Qdrant vector DB** using `sentence-transformers/all-MiniLM-L6-v2` embeddings; returns top-k chunks with scores
- **`cross_encoder_rerank(query, docs, top_k)`** → Reranks candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`; improves relevance by semantic similarity
- **`llm_call(prompt)`** → Calls `z-ai/glm-4.5-air:free` via OpenRouter API with system prompt for factual, context-only answers
- **`rag_query(query, final_k, candidate_k)`** → End-to-end pipeline: retrieve candidates → rerank → build context → generate answer

### 2. **context_manager.py** - Memory & Orchestration
- **LangGraph StateGraph** with automatic persistence (MemorySaver)
- **Workflow nodes:**
  - `_summarize_node` → Manages message history; auto-summarizes when threshold (8 messages) is reached
  - `_retrieve_node` → Fetches relevant chunks for the query
  - `_generate_node` → Builds prompt with history + context; calls LLM
- **Smart history trimming:** Keeps recent messages, summarizes old ones; respects token budget (3000 tokens default)
- **Thread management:** Each conversation gets a unique thread_id for persistent multi-turn context
- **Public API:**
  - `handle_user_message(user_text, filter_meta, thread_id)` → Full RAG pipeline in one call
  - `save_messages(user_msg, ai_msg, thread_id)` → Persist user/AI pairs without inference
  - `retrieve(query, k, filter_meta)` → Standalone retrieval
  - `build_prompt(user_query, conversation_history, retrieved)` → Token-aware prompt assembly

### 3. **app.py** - Streamlit UI
- **Chat interface** with Streamlit's native chat components
- **Streaming responses** → Real-time token generation via `st.write_stream()`
- **Session management** → Per-browser session state for messages; persists across reruns
- **Remote ingestion** → Fire-and-forget POST to ingestion service (background doc processing)
- **Error handling** → Graceful failures without breaking the UI
- **Caching** → `@st.cache_resource` for LLM model & ContextManager (efficient resource use)

---

## Setup

### Requirements
```
streamlit
langchain-openai
langgraph
sentence-transformers
qdrant-client
litellm
tiktoken
python-dotenv
```

### Environment Variables
```env
QDRANT_URL=<qdrant_db_url>
QDRANT_API_KEY=<qdrant_api_key>
QDRANT_COLLECTION=document_chunks  # optional, defaults to this
OPENROUTER_API_KEY=<your_openrouter_key>
HF_TOKEN=<huggingface_token>  # optional, for embeddings
INGESTION_SERVICE_URL=https://ingestion-service-zlt7.onrender.com  # optional
```

### Run the App
```bash
streamlit run app.py
```

---

## Data Flow

```
User Query
    ↓
[Streamlit UI] (app.py)
    ↓
[ContextManager] (context_manager.py)
    ├─ Load conversation history from thread
    ├─ Summarize old messages if needed
    ├─ Retrieve (RAG.py → Qdrant)
    ├─ Rerank (cross-encoder)
    └─ Generate (LLM with prompt)
    ↓
[Streaming Response]
    ↓
[Save to Persistent Memory]
```

---

## Key Configurations

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `short_term_max_messages` | 10 | Max messages in short-term history |
| `summary_threshold` | 8 | Trigger summarization at this count |
| `token_budget` | 3000 | Max tokens in final prompt |
| `retrieve_k` | 50 | Number of candidates to fetch |
| `system_prompt` | [see RAG.py] | LLM behavior directive |

---

## Usage Examples

### Via Streamlit UI
Just type queries in the chat box. The system handles everything—retrieval, reranking, context management, streaming.

### Programmatic (Python)
```python
from context_manager import ContextManager
from RAG import call_vector_db, llm_call, cross_encoder_rerank
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", 
                 openai_api_key=..., 
                 model="z-ai/glm-4.5-air:free")

cm = ContextManager(
    call_vector_db=call_vector_db,
    llm_call=llm_call,
    rerank_fn=cross_encoder_rerank,
    llm_model=llm
)

# Start a conversation
thread_id = cm.start_new_thread()
result = cm.handle_user_message("What is U-NET?", thread_id=thread_id)
print(result["answer"])
```

---

## Performance Notes

- **Embeddings** → ~100ms (sentence-transformers, runs locally)
- **Reranking** → ~200-500ms (cross-encoder, k=50)
- **LLM inference** → ~1-5s (depends on model & context size, streamed)
- **Token counting** → Automatic trimming keeps prompts under budget

---


## Qdrant DB CRON JOB

weekly cron jobs are performed in which top-5 ML research papers are ingested into vector database. This ensures that
the latest information is available to the LLM and also enhances the user experience