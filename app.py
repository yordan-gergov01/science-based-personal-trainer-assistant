import streamlit as st
import sys
from pathlib import Path
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import create_rag_chain, ask_question
from src.config import MODULE_CATEGORIES

# Page config
st.set_page_config(
    page_title="Personal Fitness Assistant",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("💪 Ask your personal coach (Q&A)")

# Initialize
@st.cache_resource(show_spinner=False)
def initialize_chain():
    """Initialize chain (cached)"""
    return create_rag_chain(verbose=False)

# Load chain
with st.spinner("Loading AI model..."):
    try:
        chain = initialize_chain()
        st.success("✅ Ready to answer questions!")
    except Exception as e:
        st.error(f"❌ Error loading chain: {e}")
        st.code(traceback.format_exc())
        st.stop()

# Sidebar
with st.sidebar:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.header("💡 Example Questions")
    examples = [
        "What is protein?",
        "How much protein per day?",
        "Best exercises for chest?"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example[:20]}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about training, nutrition, etc..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                from src.rag_chain import ask_question
                
                answer, sources = ask_question(chain, prompt, verbose=False)
                
                # Show answer
                st.markdown(answer)

                # Save to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Full Error Details"):
                    st.code(traceback.format_exc())