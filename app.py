"""
Streamlit Frontend for Book RAG System
"""

import streamlit as st
from streamlit_lottie import st_lottie
import requests
from rag_backend import RAGBackend

# Load Lottie animations
def load_lottie_url(url: str):
    """Load Lottie animation from URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Page config
st.set_page_config(
    page_title="Book RAG System",
    page_icon="📚",
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

# Load Lottie animations (cached)
@st.cache_data
def get_animations():
    """Load all Lottie animations."""
    return {
        'books': load_lottie_url('https://lottie.host/7f4c3b5e-8f3a-4c4e-9c4a-2b5e3f4c5d6e/4kF8K9mN7Q.json'),  # Book animation
        'empty': load_lottie_url('https://assets9.lottiefiles.com/packages/lf20_2glqz0ia.json'),  # Empty state
    }

animations = get_animations()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    # OpenRouter API
    st.markdown("### 🤖 OpenRouter API")
    openrouter_api_key = st.text_input("API Key", type="password", key="openrouter_key")
    openrouter_model = st.text_input(
        "Model",
        value="nvidia/nemotron-3-nano-30b-a3b:free",
        help="Enter the OpenRouter model name (e.g., nvidia/nemotron-3-nano-30b-a3b:free, openai/gpt-4o-2024-11-20, anthropic/claude-3.5-sonnet)"
    )

    st.markdown("---")

    # Retrieval Settings
    st.markdown("### 🔍 Retrieval Settings")
    rewrite_mode = st.radio(
        "Query Rewriting Mode",
        options=["none", "hyde", "multi_query"],
        index=1,  # Default to HyDE
        format_func=lambda x: {
            "none": "None (Direct search)",
            "hyde": "HyDE (Hypothetical document)",
            "multi_query": "Multi-Query (3 variations)"
        }[x],
        help="Query rewriting improves retrieval quality:\n"
             "• None: Search with original query\n"
             "• HyDE: Generate hypothetical answer, search with that\n"
             "• Multi-Query: Generate 3 query variations, search with all"
    )

    if rewrite_mode == "hyde":
        st.caption("⚡ Generates a hypothetical answer for better retrieval")
    elif rewrite_mode == "multi_query":
        st.caption("⚡ Generates 3 query variations for broader coverage")

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
        if animations['empty']:
            st_lottie(animations['empty'], height=150, key="empty_animation")
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
col1, col2 = st.columns([4, 1])
with col1:
    st.title("📚 Book RAG System")
    st.caption("Standalone RAG with ChromaDB, Sentence Transformers, and OpenRouter")
with col2:
    if animations['books']:
        st_lottie(animations['books'], height=80, key="books_header")

# Chat interface
st.subheader("💬 Chat with your books")

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
                        score_info = f"Distance: {source['distance']:.3f}"
                        if 'rerank_score' in source:
                            score_info += f" | Rerank: {source['rerank_score']:.3f}"
                        st.markdown(f"**Source #{i+1}** ({score_info})")
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
        # Query rewriting step (if enabled)
        if rewrite_mode == "hyde":
            with st.spinner("Generating hypothetical document for better retrieval..."):
                pass  # The actual generation happens in retrieve_relevant_chunks
        elif rewrite_mode == "multi_query":
            with st.spinner("Generating 3 query variations for better retrieval..."):
                pass  # The actual generation happens in retrieve_relevant_chunks

        with st.spinner("Searching documents..."):
            # Retrieve relevant chunks
            try:
                sources = backend.retrieve_relevant_chunks(
                    prompt,
                    st.session_state.selected_collections,
                    top_k=10,
                    rewrite_mode=rewrite_mode,
                    api_key=openrouter_api_key,
                    rewrite_model=openrouter_model,
                    rerank=True
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

        # Generate answer with streaming
        try:
            # Create placeholder for streaming text
            answer_placeholder = st.empty()
            full_answer = ""

            # Stream the response
            for chunk in backend.query_openrouter(
                prompt,
                context,
                openrouter_api_key,
                openrouter_model,
                collection_names=st.session_state.selected_collections,
                stream=True
            ):
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")  # Add cursor effect

            # Remove cursor and show final answer
            answer_placeholder.markdown(full_answer)

            # Show sources
            with st.expander("📎 View Sources"):
                for i, source in enumerate(sources):
                    score_info = f"Distance: {source['distance']:.3f}"
                    if 'rerank_score' in source:
                        score_info += f" | Rerank: {source['rerank_score']:.3f}"
                    st.markdown(f"**Source #{i+1}** ({score_info})")
                    st.text(source['text'][:500] + "...")
                    st.markdown(f"*Page: {source['page_number']}*")
                    st.markdown("---")

            # Save message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
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
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Lottie animation container styling */
    div[data-testid="stLottie"] {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px 0;
    }

    /* Smooth transitions for all elements */
    .stExpander, .stCheckbox, .stRadio {
        transition: all 0.3s ease;
    }

    /* Hover effects */
    .stCheckbox:hover, .stRadio:hover {
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)
