import streamlit as st
from langchain_openai import ChatOpenAI
# We still need these imports for the ContextManager
from RAG import call_vector_db, llm_call, cross_encoder_rerank
# Import from our new file
from context_manager import ContextManager
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import requests
INGESTION_SERVICE_URL = os.getenv('INGESTION_SERVICE_URL')
load_dotenv()
def trigger_remote_ingestion(query: str, timeout: float = 10.0) -> None:
    """
    Fire-and-forget POST to ingestion service. Fails silently (logs to stdout).
    Minimal: does NOT update st.session_state or show anything to users.
    """
    url = INGESTION_SERVICE_URL + "/enqueue"
    headers = {}

    try:
        requests.post(url, json={"query": query}, headers=headers, timeout=timeout)
    except Exception as e:
        # don't raise or show to user — just log for operator debugging
        print(f"[WARN] trigger_remote_ingestion failed: {e}", flush=True)

# --- Model and Context Manager Setup ---

@st.cache_resource
def get_llm_model():
    """Cache the LLM model instance."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="z-ai/glm-4.5-air:free", 
        temperature=0
    )

@st.cache_resource
def get_context_manager(_llm_model):
    """Cache the ContextManager instance."""
    return ContextManager(
        call_vector_db=call_vector_db,
        llm_call=llm_call,  # Used for summarization
        rerank_fn=cross_encoder_rerank,
        llm_model=_llm_model  # Pass the model for summarization
    )

llm_model = get_llm_model()
cm = get_context_manager(llm_model)

# --- Streamlit App ---

st.set_page_config(page_title="Machine Learning Chatbot", layout="centered")
st.title("Machine Learning Chatbot 💬")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize thread_id in session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = cm.start_new_thread()
    print(f"New thread created: {st.session_state.thread_id}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter your query"):
    if not query.strip():
        st.error("Please enter a non-empty query.")
    else:
        # Add user message to UI chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        
        # --- New Streaming RAG Logic ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Get conversation history from the manager
                    history = cm.get_conversation_history(st.session_state.thread_id)
                    
                    # 2. Retrieve context
                    # (Assuming default retrieve_k, add filters if needed)
                    context_chunks = cm.retrieve(query) 
                    
                    # 3. Build the prompt
                    prompt = cm.build_prompt(
                        user_query=query,
                        conversation_history=history,
                        retrieved=context_chunks
                    )
                    
                    # 4. Define the streaming generator
                    def stream_generator():
                        for chunk in llm_model.stream(prompt):
                            yield chunk.content
                    
                    # 5. Stream the response to the UI
                    # st.write_stream returns the full, concatenated response
                    full_response = st.write_stream(stream_generator())

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    full_response = "Sorry, I ran into an error."

        # 6. Add assistant response to UI chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 7. Save the messages to the ContextManager's persistent memory
        try:
            cm.save_messages(
                user_message=HumanMessage(content=query),
                ai_message=AIMessage(content=full_response),
                thread_id=st.session_state.thread_id
            )
        except Exception as e:
            st.warning(f"Failed to save history to context manager: {e}")
        

        try:
            trigger_remote_ingestion(query)
        except Exception as e:
            # never show to user; just log for operator debugging
            print(f"[WARN] trigger_remote_ingestion threw: {e}", flush=True)
