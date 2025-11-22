import streamlit as st
from langchain_openai import ChatOpenAI
from RAG import call_vector_db, llm_call, cross_encoder_rerank
from context_manager import ContextManager
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import requests

INGESTION_SERVICE_URL = 'https://ingestion-service-zlt7.onrender.com'
load_dotenv()

def trigger_remote_ingestion(query: str, timeout: float = 10.0) -> None:
    """Fire-and-forget POST to ingestion service."""
    url = INGESTION_SERVICE_URL + "/enqueue"
    headers = {}

    try:
        requests.post(url, json={"query": query}, headers=headers, timeout=timeout)
    except Exception as e:
        print(f"[WARN] trigger_remote_ingestion failed: {e}", flush=True)


def convert_latex_delimiters(text: str) -> str:
    """Convert OpenAI-style LaTeX to Streamlit format."""
    if not isinstance(text, str):
        return text
    
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    return text


@st.cache_resource
def get_llm_model():
    """Cache the LLM model instance."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-oss-20b:free", 
        temperature=0
    )


@st.cache_resource
def get_context_manager(_llm_model):
    """Cache the ContextManager instance."""
    return ContextManager(
        call_vector_db=call_vector_db,
        llm_call=llm_call,
        rerank_fn=cross_encoder_rerank,
        llm_model=_llm_model
    )


llm_model = get_llm_model()
cm = get_context_manager(llm_model)


st.set_page_config(page_title="Machine Learning Chatbot", layout="centered")

# --- Custom CSS Injection ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")
# ----------------------------

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
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # --- Streaming RAG Logic with LaTeX (Single Display) ---
        with st.chat_message("assistant"):
            # Create empty placeholder for streaming output
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # 1. Get conversation history
                    history = cm.get_conversation_history(st.session_state.thread_id)
                    
                    # 2. Retrieve context
                    context_chunks = cm.retrieve(query) 
                    
                    # 3. Build the prompt
                    prompt = cm.build_prompt(
                        user_query=query,
                        conversation_history=history,
                        retrieved=context_chunks
                    )
                    
                    # 4. Stream to placeholder directly
                    def stream_generator():
                        for chunk in llm_model.stream(prompt):
                            yield chunk.content
                    
                    # 5. Stream response to placeholder
                    full_response = message_placeholder.write_stream(stream_generator())

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    full_response = "Sorry, I ran into an error."


        # 6. Convert LaTeX delimiters and update placeholder once
        full_response_converted = convert_latex_delimiters(full_response)
        
        # Only update if conversion changed something
        if full_response_converted != full_response:
            message_placeholder.markdown(full_response_converted)
        
        # 7. Add to UI chat history with converted version
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_converted
        })
        
        # 8. Save to ContextManager
        try:
            cm.save_messages(
                user_message=HumanMessage(content=query),
                ai_message=AIMessage(content=full_response_converted),
                thread_id=st.session_state.thread_id
            )
        except Exception as e:
            st.warning(f"Failed to save history: {e}")


# --- Sidebar for File Ingestion ---
with st.sidebar:
    st.header("File Ingestion")
    st.markdown("Enter the filename you want to ingest.")
    
    filename_input = st.text_input("Filename", placeholder="example.pdf")
    
    if st.button("Ingest File"):
        if filename_input.strip():
            with st.spinner(f"Triggering ingestion for {filename_input}..."):
                trigger_remote_ingestion(filename_input)
            st.success(f"Ingestion triggered for: {filename_input}")
        else:
            st.warning("Please enter a valid filename.")
