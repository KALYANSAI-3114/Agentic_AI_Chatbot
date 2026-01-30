import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

#title 
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    layout="wide"
)
#Interface code for chatbot
@st.cache_resource
def load_rag_pipeline():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        st.error("**SARVAM_API_KEY not found in .env file!**")
        st.stop()
    print(f"Sarvam API Key loaded: {api_key[:10]}...")
    return RAGPipeline("Ebook-Agentic-AI.pdf")

def main():
    st.title("🤖 Agentic AI Chatbot")
    
    # API Key check
    if not os.getenv("SARVAM_API_KEY"):
        st.error("""
         **Missing SARVAM_API_KEY!**
        1. Get key: https://dashboard.sarvam.ai/
        2. Add to `.env`: `SARVAM_API_KEY=your_key_here`
        3. Restart app
        """)
        st.stop()
    
    # Load pipeline
    if "rag_pipeline" not in st.session_state:
        with st.spinner("Initializing Sarvam RAG pipeline..."):
            try:
                st.session_state.rag_pipeline = load_rag_pipeline()
                st.success("Sarvam pipeline loaded!")
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Sarvam AI")
        st.success("Using Sarvam-m model")
        st.info("**Features:**\n• Local FAISS storage\n• SentenceTransformer\n• Indian AI focus\n• Generous credits")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface (same as before)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "contexts" in message:
                conf = message["confidence"]
                with st.expander(f"Contexts (Confidence: {conf:.3f})"):
                    for i, ctx in enumerate(message["contexts"], 1):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.metric("Score", f"{ctx['score']:.3f}")
                        with col2:
                            st.caption(ctx['text'][:300] + "..." if len(ctx['text']) > 300 else ctx['text'])
    
    if prompt := st.chat_input("Ask about Agentic AI..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Sarvam is thinking..."):
                response = st.session_state.rag_pipeline.invoke(prompt)
                
                st.markdown("**Answer:**")
                st.markdown(response["answer"])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{response['confidence']:.3f}")
                
                full_response = {
                    "role": "assistant",
                    "content": response["answer"],
                    "contexts": response["contexts"],
                    "confidence": response["confidence"]
                }
                st.session_state.messages.append(full_response)

if __name__ == "__main__":
    main()
