# context_manager_only.py
import time
import math
from typing import List, Dict, Any, Callable, Optional

import tiktoken

class ContextManager:
    """
    ContextManager coordinates multi-turn conversation, long/short memory,
    retrieval (via adapter.query_by_vector), optional reranking and prompt-building.
    
    Requirements:
      - call_vector_db(query: str, k: int) -> List[str]  (provided externally)
      - llm_call(prompt: str) -> str  (provided externally)
    """
    def __init__(
        self,
        call_vector_db : Callable[[str,int], List[str]],
        llm_call: Callable[[str], str],
        summarizer_fn: Optional[Callable[[str], str]] = None,
        rerank_fn: Optional[Callable[[List[Dict], str], List[Dict]]] = None,
        short_term_max_turns: int = 8,
        memory_max_items: int = 200,
        token_budget: int = 3000,
        retrieve_k: int = 12,
        include_memory_last_n: int = 6,
    ):
        self.call_vector_db = call_vector_db
        self.llm_call = llm_call
        self.summarizer_fn = summarizer_fn or self._simple_summarizer
        self.rerank_fn = rerank_fn
        self.short_history: List[Dict[str, Any]] = []   # [{'role':'user'|'assistant','text':..., 'ts':...}, ...]
        self.long_memory: List[Dict[str, Any]] = []     # [{'summary':..., 'ts':..., 'meta':...}, ...]
        self.short_term_max_turns = short_term_max_turns
        self.memory_max_items = memory_max_items
        self.token_budget = token_budget
        self.retrieve_k = retrieve_k
        self.include_memory_last_n = include_memory_last_n

    # -------------------------
    # public API
    # -------------------------
    def add_turn(self, role: str, text: str):
        """Add a conversation turn and possibly compress old turns into long_memory."""
        self.short_history.append({'role': role, 'text': text, 'ts': time.time()})
        if len(self.short_history) > self.short_term_max_turns:
            self._compress_oldest_turns()

    def handle_user_message(self, user_text: str, filter_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        High-level helper:
          1) add user turn
          2) retrieve relevant chunks (via embed_fn -> adapter.query_by_vector)
          3) (optional) rerank
          4) build prompt (token-aware)
          5) call llm_call(prompt)
          6) add assistant turn and return answer + used chunks + prompt
        """
        # 1) store user turn
        self.add_turn("user", user_text)

        # 2) retrieve
        retrieved = self.retrieve(user_text, k=self.retrieve_k, filter_meta=filter_meta)
        # 3) pick top-N to include in prompt (we will refine ordering via rerank if present)
        top_candidates = retrieved[: max(1, min(len(retrieved), 6))]

        # 4) build prompt
        prompt = self.build_prompt(user_text, top_candidates)

        # 5) call LLM

        answer = self.llm_call(prompt)

        # 6) record assistant turn
        self.add_turn("assistant", answer)

        # used_chunks = [{"id": c.get("id")} for c in top_candidates]
        return {"answer": answer, "used_chunks": None, "prompt": prompt}

    def retrieve(self, query: str, k: Optional[int] = None, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch k candidates from vector DB 
        """
        k = k or self.retrieve_k
        
        candidates  = self.call_vector_db(query,k)

        candidates = self.rerank_fn(query,candidates,top_k=k)
        return candidates

    def build_prompt(self, user_query: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        Order and token-budget-aware prompt builder.
          - System instruction
          - Recent short-term conversation
          - Long-term memory (last N)
          - Retrieved blocks with provenance header
          - User query
        Token trimming strategy (in order of dropping priority):
          1) drop long-term memory
          2) drop retrieved chunks one-by-one (lowest priority last)
          3) truncate retrieved chunk texts
        """
        system = "SYSTEM: You are an expert assistant. Use ONLY the provided context for factual claims and cite sources.But don't say 'based on the context'."

        recent_block = "\n".join(f"{t['role']}: {t['text']}" for t in self.short_history[-self.short_term_max_turns:])
        memory_block = "\n".join(m['summary'] for m in self.long_memory[-self.include_memory_last_n:]) if self.long_memory else ""

        retrieved_parts = []
        for c in retrieved:
            meta = c.get("metadata", {}) or {}
            title =  meta.get("source") or "unknown"
            preview = meta.get("preview", "")
            header = f"[{title}— {preview}]"
            retrieved_parts.append(header + "\n" + (c.get("text") or ""))

        parts = [
            "SYSTEM:\n" + system,
        ]
        if memory_block:
            parts.append("LONG_TERM_MEMORY:\n" + memory_block)
        if recent_block:
            parts.append("RECENT_CONVERSATION:\n" + recent_block)
        if retrieved_parts:
            parts.append("RETRIEVED_DOCUMENTS:\n" + "\n\n".join(retrieved_parts))
        parts.append("USER_QUERY:\n" + user_query)

        prompt = "\n\n".join(parts)

        # Trim according to token budget
        if self._num_tokens(prompt) <= self.token_budget:
            return prompt

        # 1) drop long-term memory
        if memory_block:
            parts = [p for p in parts if not p.startswith("LONG_TERM_MEMORY")]
            prompt = "\n\n".join(parts)
            if self._num_tokens(prompt) <= self.token_budget:
                return prompt

        # 2) iteratively drop lowest-priority retrieved_parts (we assume earlier items are higher priority)
        # find index of RETRIEVED_DOCUMENTS in parts
        def rebuild_with_retrieved(r_parts):
            new_parts = [p for p in parts if not p.startswith("RETRIEVED_DOCUMENTS")]
            if r_parts:
                new_parts.insert(len(new_parts)-1, "RETRIEVED_DOCUMENTS:\n" + "\n\n".join(r_parts))
            return "\n\n".join(new_parts)

        curr_retrieved = list(retrieved_parts)
        while curr_retrieved and self._num_tokens(prompt) > self.token_budget:
            # drop last retrieved (lowest priority)
            curr_retrieved.pop()
            prompt = rebuild_with_retrieved(curr_retrieved)

        # 3) truncate each retrieved document body naively (last resort)
        if self._num_tokens(prompt) > self.token_budget and curr_retrieved:
            truncated = []
            for rp in curr_retrieved:
                words = rp.split()
                truncated_text = " ".join(words[:200]) + (" ..." if len(words) > 200 else "")
                truncated.append(truncated_text)
            prompt = rebuild_with_retrieved(truncated)

        # 4) final fallback: hard trunocate prompt to token_budget words approximation
        if self._num_tokens(prompt) > self.token_budget:
            words = prompt.split()
            approx_words = max(1, int(self.token_budget * 0.75))  # conservative approx
            prompt = " ".join(words[-approx_words:])

        return prompt

    def add_memory(self, summary: str, meta: Optional[Dict[str, Any]] = None):
        """Allow external addition of long-term memory (e.g., from an external summarizer)."""
        self.long_memory.append({'summary': summary, 'ts': time.time(), 'meta': meta or {}})
        self.long_memory = self.long_memory[-self.memory_max_items:]

    def feedback(self, chunk_id: str, positive: bool = True):
        """
        Hook for feedback: implementations can override to reweight adapter / store signals.
        Default: no-op.
        """
        pass

    # -------------------------
    # internal helpers
    # -------------------------
    def _compress_oldest_turns(self):
        """
        Compress oldest turns in short_history into a single long_memory summary using summarizer_fn.
        Strategy: compress about half of history so recent conversation remains raw.
        """
        n = len(self.short_history)
        n_to_compress = max(1, n - (self.short_term_max_turns // 2))
        to_compress = self.short_history[:n_to_compress]
        combined_text = "\n".join(f"{t['role']}: {t['text']}" for t in to_compress)
        try:
            summary = self.summarizer_fn(combined_text)
        except Exception:
            summary = self._simple_summarizer(combined_text)
        self.long_memory.append({'summary': summary, 'ts': time.time(), 'meta': {'compressed_turns': n_to_compress}})
        # drop compressed turns from short_history
        self.short_history = self.short_history[n_to_compress:]
        # prune long_memory to cap
        self.long_memory = self.long_memory[-self.memory_max_items:]

    def _simple_summarizer(self, text: str, max_chars: int = 800) -> str:
        """Fallback cheap summarizer (just truncates nicely)."""
        s = " ".join(text.split())
        if len(s) <= max_chars:
            return s
        return s[:max_chars].rsplit(" ", 1)[0] + " ..."

    def _num_tokens(self, text: str) -> int:
        """Estimate token count: use tiktoken if available, otherwise heuristic."""
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
# context_manager_only.py
import time
import math
from typing import List, Dict, Any, Callable, Optional

import tiktoken

class ContextManager:
    """
    ContextManager coordinates multi-turn conversation, long/short memory,
    retrieval (via adapter.query_by_vector), optional reranking and prompt-building.
    
    Requirements:
      - call_vector_db(query: str, k: int) -> List[str]  (provided externally)
      - llm_call(prompt: str) -> str  (provided externally)
    """
    def __init__(
        self,
        call_vector_db : Callable[[str,int], List[str]],
        llm_call: Callable[[str], str],
        summarizer_fn: Optional[Callable[[str], str]] = None,
        rerank_fn: Optional[Callable[[List[Dict], str], List[Dict]]] = None,
        short_term_max_turns: int = 8,
        memory_max_items: int = 200,
        token_budget: int = 3000,
        retrieve_k: int = 20,
        include_memory_last_n: int = 6,
    ):
        self.call_vector_db = call_vector_db
        self.llm_call = llm_call
        self.summarizer_fn = summarizer_fn or self._simple_summarizer
        self.rerank_fn = rerank_fn
        self.short_history: List[Dict[str, Any]] = []   # [{'role':'user'|'assistant','text':..., 'ts':...}, ...]
        self.long_memory: List[Dict[str, Any]] = []     # [{'summary':..., 'ts':..., 'meta':...}, ...]
        self.short_term_max_turns = short_term_max_turns
        self.memory_max_items = memory_max_items
        self.token_budget = token_budget
        self.retrieve_k = retrieve_k
        self.include_memory_last_n = include_memory_last_n

    # -------------------------
    # public API
    # -------------------------
    def add_turn(self, role: str, text: str):
        """Add a conversation turn and possibly compress old turns into long_memory."""
        self.short_history.append({'role': role, 'text': text, 'ts': time.time()})
        if len(self.short_history) > self.short_term_max_turns:
            self._compress_oldest_turns()

    def handle_user_message(self, user_text: str, filter_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        High-level helper:
          1) add user turn
          2) retrieve relevant chunks (via embed_fn -> adapter.query_by_vector)
          3) (optional) rerank
          4) build prompt (token-aware)
          5) call llm_call(prompt)
          6) add assistant turn and return answer + used chunks + prompt
        """
        # 1) store user turn
        self.add_turn("user", user_text)

        # 2) retrieve
        retrieved = self.retrieve(user_text, k=self.retrieve_k, filter_meta=filter_meta)
        # 3) pick top-N to include in prompt (we will refine ordering via rerank if present)
        top_candidates = retrieved[: max(1, min(len(retrieved), 6))]

        # 4) build prompt
        prompt = self.build_prompt(user_text, top_candidates)

        # 5) call LLM

        answer = self.llm_call(prompt)

        # 6) record assistant turn
        self.add_turn("assistant", answer)

        # used_chunks = [{"id": c.get("id")} for c in top_candidates]
        return {"answer": answer, "used_chunks": None, "prompt": prompt}

    def retrieve(self, query: str, k: int = 50, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch k candidates from vector DB 
        """
        k = k or self.retrieve_k
        
        candidates  = self.call_vector_db(query,k)

        candidates = self.rerank_fn(query,candidates,top_k=k)
        return candidates

    def build_prompt(self, user_query: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        Order and token-budget-aware prompt builder.
          - System instruction
          - Recent short-term conversation
          - Long-term memory (last N)
          - Retrieved blocks with provenance header
          - User query
        Token trimming strategy (in order of dropping priority):
          1) drop long-term memory
          2) drop retrieved chunks one-by-one (lowest priority last)
          3) truncate retrieved chunk texts
        """
        system = "SYSTEM: You are an expert assistant. Use ONLY the provided context for factual claims and cite sources.But don't say 'based on the context'."

        recent_block = "\n".join(f"{t['role']}: {t['text']}" for t in self.short_history[-self.short_term_max_turns:])
        memory_block = "\n".join(m['summary'] for m in self.long_memory[-self.include_memory_last_n:]) if self.long_memory else ""

        retrieved_parts = []
        for c in retrieved:
            meta = c.get("metadata", {}) or {}
            title = meta.get("paper_title") or meta.get("source") or "unknown"
            preview = meta.get("preview", "")
            header = f"[{title} {preview}]"
            retrieved_parts.append(header + "\n" + (c.get("text") or ""))

        parts = [
            "SYSTEM:\n" + system,
        ]
        if memory_block:
            parts.append("LONG_TERM_MEMORY:\n" + memory_block)
        if recent_block:
            parts.append("RECENT_CONVERSATION:\n" + recent_block)
        if retrieved_parts:
            parts.append("RETRIEVED_DOCUMENTS:\n" + "\n\n".join(retrieved_parts))
        parts.append("USER_QUERY:\n" + user_query)

        prompt = "\n\n".join(parts)

        # Trim according to token budget
        if self._num_tokens(prompt) <= self.token_budget:
            return prompt

        # 1) drop long-term memory
        if memory_block:
            parts = [p for p in parts if not p.startswith("LONG_TERM_MEMORY")]
            prompt = "\n\n".join(parts)
            if self._num_tokens(prompt) <= self.token_budget:
                return prompt

        # 2) iteratively drop lowest-priority retrieved_parts (we assume earlier items are higher priority)
        # find index of RETRIEVED_DOCUMENTS in parts
        def rebuild_with_retrieved(r_parts):
            new_parts = [p for p in parts if not p.startswith("RETRIEVED_DOCUMENTS")]
            if r_parts:
                new_parts.insert(len(new_parts)-1, "RETRIEVED_DOCUMENTS:\n" + "\n\n".join(r_parts))
            return "\n\n".join(new_parts)

        curr_retrieved = list(retrieved_parts)
        while curr_retrieved and self._num_tokens(prompt) > self.token_budget:
            # drop last retrieved (lowest priority)
            curr_retrieved.pop()
            prompt = rebuild_with_retrieved(curr_retrieved)

        # 3) truncate each retrieved document body naively (last resort)
        if self._num_tokens(prompt) > self.token_budget and curr_retrieved:
            truncated = []
            for rp in curr_retrieved:
                words = rp.split()
                truncated_text = " ".join(words[:200]) + (" ..." if len(words) > 200 else "")
                truncated.append(truncated_text)
            prompt = rebuild_with_retrieved(truncated)

        # 4) final fallback: hard trunocate prompt to token_budget words approximation
        if self._num_tokens(prompt) > self.token_budget:
            words = prompt.split()
            approx_words = max(1, int(self.token_budget * 0.75))  # conservative approx
            prompt = " ".join(words[-approx_words:])

        return prompt

    def add_memory(self, summary: str, meta: Optional[Dict[str, Any]] = None):
        """Allow external addition of long-term memory (e.g., from an external summarizer)."""
        self.long_memory.append({'summary': summary, 'ts': time.time(), 'meta': meta or {}})
        self.long_memory = self.long_memory[-self.memory_max_items:]

    def feedback(self, chunk_id: str, positive: bool = True):
        """
        Hook for feedback: implementations can override to reweight adapter / store signals.
        Default: no-op.
        """
        pass

    # -------------------------
    # internal helpers
    # -------------------------
    def _compress_oldest_turns(self):
        """
        Compress oldest turns in short_history into a single long_memory summary using summarizer_fn.
        Strategy: compress about half of history so recent conversation remains raw.
        """
        n = len(self.short_history)
        n_to_compress = max(1, n - (self.short_term_max_turns // 2))
        to_compress = self.short_history[:n_to_compress]
        combined_text = "\n".join(f"{t['role']}: {t['text']}" for t in to_compress)
        try:
            summary = self.summarizer_fn(combined_text)
        except Exception:
            summary = self._simple_summarizer(combined_text)
        self.long_memory.append({'summary': summary, 'ts': time.time(), 'meta': {'compressed_turns': n_to_compress}})
        # drop compressed turns from short_history
        self.short_history = self.short_history[n_to_compress:]
        # prune long_memory to cap
        self.long_memory = self.long_memory[-self.memory_max_items:]

    def _simple_summarizer(self, text: str, max_chars: int = 800) -> str:
        """Fallback cheap summarizer (just truncates nicely)."""
        s = " ".join(text.split())
        if len(s) <= max_chars:
            return s
        return s[:max_chars].rsplit(" ", 1)[0] + " ..."

    def _num_tokens(self, text: str) -> int:
        """Estimate token count: use tiktoken if available, otherwise heuristic."""
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))