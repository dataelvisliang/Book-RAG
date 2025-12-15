"""
Streamlit Frontend for PDF RAG System
"""

import streamlit as st
from rag_backend import RAGBackend

# Page config
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📄",
    layout="wide"
)

# Initialize backend
@st.cache_resource(show_spinner="Loading embedding model...")
def get_rag_backend():
    """Initialize RAG backend (cached for performance)."""
    return RAGBackend(
        persist_directory="./chroma_db",
        embedding_model_name="BAAI/bge-base-en-v1.5"
    )

backend = get_rag_backend()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    # OpenRouter API
    st.markdown("### 🤖 OpenRouter API")
    openrouter_api_key = st.text_input("API Key", type="password", key="openrouter_key")
    openrouter_model = st.selectbox(
        "Model",
        [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4-turbo",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-70b-instruct"
        ],
        index=0
    )

    st.markdown("---")

    # Document selection
    st.markdown("### 📚 Documents")

    # Get all collections
    collections = backend.get_all_collections()

    if collections:
        selected_collections = []
        for collection in collections:
            col_name = collection.name
            display_name = collection.metadata.get('document_name', col_name)

            if st.checkbox(display_name, key=f"doc_{col_name}"):
                selected_collections.append(col_name)

        st.session_state.selected_collections = selected_collections
    else:
        st.info("No documents uploaded yet")
        st.session_state.selected_collections = []

    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("1. Enter your OpenRouter API key")
    st.markdown("2. Select documents to chat with")
    st.markdown("3. Ask questions!")
    st.markdown("")
    st.markdown("_Note: Use your preprocessing script to add documents to the database_")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_collections" not in st.session_state:
    st.session_state.selected_collections = []

# Main app
st.title("📄 PDF RAG System")
st.caption("Standalone RAG with ChromaDB, Sentence Transformers, and OpenRouter")

# Chat interface
st.subheader("💬 Chat with your documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources
        if message["role"] == "assistant" and "sources" in message:
            sources = message.get("sources", [])
            if sources:
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source #{i+1}** (Distance: {source['distance']:.3f})")
                        st.text(source['text'][:500] + "...")
                        st.markdown(f"*Page: {source['page_number']}*")
                        st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Validation
    if not openrouter_api_key:
        st.error("❌ Please enter your OpenRouter API key in the sidebar")
        st.stop()

    if not st.session_state.selected_collections:
        st.warning("⚠️ Please select at least one document from the sidebar")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Retrieve relevant chunks
            try:
                sources = backend.retrieve_relevant_chunks(
                    prompt,
                    st.session_state.selected_collections
                )

                if not sources:
                    st.error("No relevant information found in the documents")
                    st.stop()

                # Build context
                context = "\n\n".join([
                    f"[Page {source['page_number']}]: {source['text']}"
                    for source in sources
                ])

            except Exception as e:
                st.error(f"Error retrieving context: {e}")
                st.stop()

        with st.spinner("Generating answer..."):
            try:
                # Generate answer
                answer = backend.query_openrouter(
                    prompt,
                    context,
                    openrouter_api_key,
                    openrouter_model
                )

                st.markdown(answer)

                # Show sources
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source #{i+1}** (Distance: {source['distance']:.3f})")
                        st.text(source['text'][:500] + "...")
                        st.markdown(f"*Page: {source['page_number']}*")
                        st.markdown("---")

                # Save message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = str(e)
                st.error(f"Error: {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {error_msg}"
                })

# Styling
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)
