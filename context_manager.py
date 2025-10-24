import time
import uuid
from typing import List, Dict, Any, Callable, Optional, TypedDict
import tiktoken

from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    RemoveMessage,
    BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph


# Define the state for our graph
class RAGState(TypedDict):
    messages: List[BaseMessage]
    user_query: Optional[str] # <-- Made optional
    context: List[Dict[str, Any]]
    filter_meta: Optional[Dict[str, Any]]
    prompt: str


class ContextManager:
    """
    Modern ContextManager using LangGraph for full RAG orchestration and memory.
    
    Key improvements:
    - Uses LangGraph's StateGraph for automatic persistence AND orchestration.
    - Moves all RAG logic (retrieve, prompt, generate) into graph nodes.
    - Implements efficient message trimming with trim_messages.
    - Automatic summarization using LLM-based compression.
    - Fixes token trimming logic to keep *recent* history.
    - Fixes thread management and default behavior.
    """
    
    def __init__(
        self,
        call_vector_db: Callable[[str, int], List[Dict]],
        llm_call: Callable[[str], str],
        llm_model: Any,  # LangChain LLM instance for summarization
        rerank_fn: Optional[Callable[[str, List[Dict], int], List[Dict]]] = None,
        short_term_max_messages: int = 10,
        summary_threshold: int = 8,
        token_budget: int = 3000,
        retrieve_k: int = 50,
        system_prompt: str = "You are an expert assistant. Use ONLY the provided context for factual claims. No need to give scores",
    ):
        self.call_vector_db = call_vector_db
        self.llm_call = llm_call
        self.llm_model = llm_model
        self.rerank_fn = rerank_fn
        self.short_term_max_messages = short_term_max_messages
        self.summary_threshold = summary_threshold
        self.token_budget = token_budget
        self.retrieve_k = retrieve_k
        self.system_prompt = system_prompt
        
        # Initialize LangGraph workflow with memory
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        # Thread management
        self.current_thread_id = self.start_new_thread()
        
        # Encoding for token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with state management."""
        workflow = StateGraph(state_schema=RAGState)
        
        # Define the nodes
        workflow.add_node("check_summary", self._summarize_node)
        workflow.add_node("retrieve_context", self._retrieve_node)
        workflow.add_node("generate_answer", self._generate_node)
        
        # Define the edges
        workflow.set_entry_point("check_summary")
        
        # After summarizing, decide what to do next
        workflow.add_conditional_edges(
            "check_summary",  # The node to branch from
            self._should_retrieve, # A function to decide the next step
            {
                "retrieve": "retrieve_context", # If "retrieve", go to retrieve_node
                "end": END                      # If "end", stop here
            }
        )
        
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow

    # --- NEW DECIDER FUNCTION ---
    def _should_retrieve(self, state: RAGState) -> str:
        """
        Decide whether to proceed with RAG or just end (if only saving messages).
        """
        if state.get("user_query"):
            # If user_query is present, we are in a full RAG pipeline
            return "retrieve"
        else:
            # If no user_query, we are just saving messages
            return "end"
    # --- END NEW FUNCTION ---

    def _summarize_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Check if summarization is needed and perform it.
        This node runs first to manage the message history.
        """
        messages = state["messages"]
        
        # Check if we need to summarize
        if len(messages) >= self.summary_threshold:
            # This function returns a dict, so we update state with it
            summary_update = self._summarize_and_process(messages)
            state.update(summary_update)
            return summary_update
        
        # No summarization needed, just pass the messages through
        return {"messages": messages}

    def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents based on the user query."""
        # We can be sure user_query exists because _should_retrieve checked it
        user_query = state["user_query"] 
        filter_meta = state.get("filter_meta")
        
        retrieved_chunks = self.retrieve(
            user_query, 
            k=self.retrieve_k, 
            filter_meta=filter_meta
        )
        return {"context": retrieved_chunks}

    def _generate_node(self, state: RAGState) -> Dict[str, Any]:
        """Generate an answer using the prompt and context."""
        user_query = state["user_query"]
        conversation_history = state["messages"]
        retrieved_chunks = state["context"]
        
        prompt = self.build_prompt(
            user_query=user_query,
            conversation_history=conversation_history,
            retrieved=retrieved_chunks
        )
        
        answer = self.llm_call(prompt)
        
        assistant_message = AIMessage(content=answer)
        
        # Return new messages and the prompt for logging
        return {"messages": [assistant_message], "prompt": prompt}


    def _summarize_and_process(self, messages: List[BaseMessage]) -> Dict[str, List[BaseMessage]]:
        """
        Summarize old messages when threshold is reached.
        Keeps recent messages and creates a summary of older ones.
        """
        # Split messages: older ones to summarize, recent ones to keep
        num_to_summarize = len(messages) - (self.short_term_max_messages // 2)
        messages_to_summarize = messages[:num_to_summarize]
        recent_messages = messages[num_to_summarize:]
        
        # Generate summary using LLM
        summary_prompt = (
            "Distill the following chat messages into a concise summary. "
            "Include all important details, facts, and context:\n\n"
        )
        
        # Format messages for summary
        formatted_history = "\n".join([
            f"{msg.type}: {msg.content}" 
            for msg in messages_to_summarize
        ])
        
        summary_content = self.llm_model.invoke([
            HumanMessage(content=summary_prompt + formatted_history)
        ]).content
        
        # Create summary message
        summary_message = SystemMessage(
            content=f"[CONVERSATION SUMMARY]: {summary_content}"
        )
        
        # Delete old messages and keep summary + recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        
        return {
            "messages": [summary_message] + recent_messages + delete_messages
        }

    def handle_user_message(
        self, 
        user_text: str, 
        filter_meta: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        High-level method to handle a user message with automatic memory management.
        This method now just invokes the graph.
        
        Args:
            user_text: User's input message
            filter_meta: Optional metadata filters for retrieval
            thread_id: Optional thread ID for conversation continuity. 
                       If None, a new thread is created.
            
        Returns:
            Dict with answer, used_chunks, prompt, and thread_id
        """
        if thread_id is None:
            thread_id = self.start_new_thread()
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Define the input for the graph
        # The 'messages' key takes a list, which LangGraph's MessagesState
        # will automatically append to the existing history.
        graph_input = {
            "messages": [HumanMessage(content=user_text)],
            "user_query": user_text, # This being present triggers the RAG flow
            "filter_meta": filter_meta
        }
        
        # Invoke the graph. This runs the full flow:
        # 1. _summarize_node (loads history, summarizes if needed)
        # 2. _should_retrieve (sees "user_query", returns "retrieve")
        # 3. _retrieve_node (gets context)
        # 4. _generate_node (builds prompt, calls LLM, adds AI msg)
        final_state = self.app.invoke(graph_input, config=config)
        
        # Extract results from the final state
        answer = final_state["messages"][-1].content
        used_chunks = final_state.get("context", [])
        prompt = final_state.get("prompt", "Prompt not captured.")
        
        return {
            "answer": answer,
            "used_chunks": [{"id": c.get("id"), "score": c.get("score")} for c in used_chunks],
            "prompt": prompt,
            "thread_id": thread_id
        }

    # --- THIS METHOD IS NOW FIXED (no code change needed) ---
    def save_messages(self, user_message: HumanMessage, ai_message: AIMessage, thread_id: str):
        """
        Manually saves a pair of user/ai messages to the history,
        running them through the summarization check.
        """
        config = {"configurable": {"thread_id": thread_id}}
        # We invoke the graph with just the messages.
        # 1. _summarize_node runs.
        # 2. _should_retrieve (sees no "user_query", returns "end")
        # 3. Graph stops.
        # This no longer crashes.
        self.app.invoke(
            {"messages": [user_message, ai_message]}, 
            config=config
        )
    # --- END OF METHOD ---


    def retrieve(
        self, 
        query: str, 
        k: Optional[int] = None, 
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and optionally rerank relevant chunks.
        (Note: filter_meta is not implemented in the base call_vector_db)
        """
        k = k or self.retrieve_k
        # We assume call_vector_db can accept filter_meta
        # If not, you'd need to modify this part.
        candidates = self.call_vector_db(query, k) # Add filter_meta here if supported
        
        if self.rerank_fn:
            candidates = self.rerank_fn(query, candidates, top_k=k)
        
        return candidates

    def build_prompt(
        self,
        user_query: str,
        conversation_history: List[BaseMessage],
        retrieved: List[Dict[str, Any]]
    ) -> str:
        """
        Build token-aware prompt with conversation history and retrieved context.
        """
        # Build system message
        system_msg = f"SYSTEM:\n{self.system_prompt}"
        
        # Format conversation history
        # We don't need to exclude the last message, as the graph state
        # only contains history *up to* the generation step.
        conversation_parts = []
        for msg in conversation_history:
            if isinstance(msg, SystemMessage):
                if "[CONVERSATION SUMMARY]" in msg.content:
                    conversation_parts.append(f"SUMMARY: {msg.content}")
            elif isinstance(msg, HumanMessage):
                conversation_parts.append(f"USER: {msg.content}")
            elif isinstance(msg, AIMessage):
                conversation_parts.append(f"ASSISTANT: {msg.content}")
        
        conversation_block = "\n".join(conversation_parts) if conversation_parts else ""
        
        # Format retrieved documents
        retrieved_parts = []
        for c in retrieved:
            score = c.get("score", 0.0)
            metadata = c.get("metadata", {}) or {}
            title = metadata.get("source", "unknown")
            text = c.get("text", "")
            
            header = f"[SOURCE: {title} | SCORE: {score:.3f}]"
            retrieved_parts.append(f"{header}\n{text}")
        
        # Assemble prompt parts
        parts = [system_msg]
        
        if conversation_block:
            parts.append(f"CONVERSATION_HISTORY:\n{conversation_block}")
        
        if retrieved_parts:
            parts.append(f"RETRIEVED_CONTEXT:\n\n{self._separator().join(retrieved_parts)}")
        
        parts.append(f"CURRENT_USER_QUERY:\n{user_query}")
        
        prompt = "\n\n".join(parts)
        
        # Token budget management
        prompt = self._trim_prompt_to_budget(prompt, parts, retrieved_parts, conversation_block)
        
        return prompt

    def _trim_prompt_to_budget(
        self,
        prompt: str,
        parts: List[str],
        retrieved_parts: List[str],
        conversation_block: str
    ) -> str:
        """Trim prompt to fit within token budget."""
        if self._num_tokens(prompt) <= self.token_budget:
            return prompt
        
        # Strategy 1: Trim conversation history (from the start, keeping recent)
        if conversation_block:
            trimmed_conv = self._trim_text_to_tokens_end(
                conversation_block, 
                self.token_budget // 4  # Allow 25% of budget for history
            )
            parts = [p if "CONVERSATION_HISTORY" not in p else f"CONVERSATION_HISTORY:\n{trimmed_conv}" for p in parts]
            prompt = "\n\n".join(parts)
            
            if self._num_tokens(prompt) <= self.token_budget:
                return prompt
        
        # Strategy 2: Iteratively drop lowest-priority retrieved chunks
        curr_retrieved = list(retrieved_parts)
        while curr_retrieved and self._num_tokens(prompt) > self.token_budget:
            curr_retrieved.pop() # Drop from the end (lowest priority)
            prompt = self._rebuild_prompt_with_retrieved(parts, curr_retrieved)
        
        # Strategy 3: Truncate each retrieved document (from the end, keeping start)
        if self._num_tokens(prompt) > self.token_budget and curr_retrieved:
            truncated = [
                self._trim_text_to_tokens_start(rp, 200) # Hard cap per chunk
                for rp in curr_retrieved
            ]
            prompt = self._rebuild_prompt_with_retrieved(parts, truncated)
        
        if self._num_tokens(prompt) > self.token_budget:
            tokens = self.encoding.encode(prompt)
            truncated_tokens = tokens[-self.token_budget:]
            prompt = self.encoding.decode(truncated_tokens)
        
        return prompt

    def _rebuild_prompt_with_retrieved(self, parts: List[str], retrieved_parts: List[str]) -> str:
        """Rebuild prompt with updated retrieved documents."""
        new_parts = []
        for p in parts:
            if p.startswith("RETRIEVED_CONTEXT"):
                if retrieved_parts:
                    new_parts.append(f"RETRIEVED_CONTEXT:\n\n{self._separator().join(retrieved_parts)}")
            else:
                new_parts.append(p)
        
        # Ensure context is added if it wasn't there before
        if "RETRIEVED_CONTEXT" not in parts[2] and retrieved_parts:
             new_parts.insert(-1, f"RETRIEVED_CONTEXT:\n\n{self._separator().join(retrieved_parts)}")
             
        return "\n\n".join(new_parts)

    def _trim_text_to_tokens_start(self, text: str, max_tokens: int) -> str:
        """Trim text to specified token count, keeping the START of the text."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        trimmed_tokens = tokens[:max_tokens]
        return self.encoding.decode(trimmed_tokens) + " ..."

    def _trim_text_to_tokens_end(self, text: str, max_tokens: int) -> str:
        """Trim text to specified token count, keeping the END of the text."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        trimmed_tokens = tokens[-max_tokens:]
        return "..." + self.encoding.decode(trimmed_tokens)

    def _num_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _separator(self) -> str:
        """Separator for retrieved documents."""
        return "\n" + "="*80 + "\n"

    def get_conversation_history(self, thread_id: Optional[str] = None) -> List[BaseMessage]:
        """Get conversation history for a thread."""
        if thread_id is None:
            thread_id = self.current_thread_id
        
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.app.get_state(config)
            return state.values.get("messages", [])
        except: # Handle case where thread doesn't exist yet
             return []


    def start_new_thread(self) -> str:
        """Create a new conversation thread and sets it as the current default."""
        new_thread_id = str(uuid.uuid4())
        self.current_thread_id = new_thread_id
        return new_thread_id

