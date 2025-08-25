import streamlit as st

from RAG import rag_query, call_vector_db, llm_call, cross_encoder_rerank, summarize_conversation
from context_manager import ContextManager

cm = ContextManager(call_vector_db=call_vector_db,
                    llm_call=llm_call,
                    summarizer_fn=summarize_conversation,
                    rerank_fn=cross_encoder_rerank)

st.set_page_config(page_title="Machine Learning Chatbot", layout="centered")
st.title("Machine Learning Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter your query"):
    if not query.strip():
        st.error("Please enter a non-empty query.")
    else:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = cm.handle_user_message(query)['answer']
                except Exception as e:
                    st.error(f"handle_user_message raised an exception: {e}")
                    result = ""
            
            if not isinstance(result, str):
                st.warning("`handle_user_message` returned a non-string. Converting to string for display.")
                result = str(result)
            
            st.markdown(result)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})