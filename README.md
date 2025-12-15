# 📄 Book RAG - Standalone PDF Question-Answering System

A powerful, standalone Python-based RAG (Retrieval-Augmented Generation) system for chatting with your PDF documents. Built with local embeddings, persistent vector storage, and OpenRouter LLM integration.

## ✨ Features

- 📚 **Local PDF Processing**: Process PDFs with detailed text extraction and chunking
- 🧠 **Local Embeddings**: Generate embeddings locally using BAAI/bge-base-en-v1.5 (GPU-accelerated on Mac)
- 💾 **Persistent Storage**: ChromaDB vector database for efficient retrieval
- 🤖 **Multiple LLM Support**: Use any model from OpenRouter (Claude, GPT-4, Gemini, LLaMA, etc.)
- 📊 **Comprehensive Logging**: Detailed preprocessing logs with statistics
- 🔄 **Standalone Preprocessing**: Separate script for batch PDF processing
- 💬 **Interactive Chat**: Clean Streamlit interface with conversation history
- 🎯 **Source Attribution**: View source chunks with page numbers and relevance scores

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)              │
│  - Chat interface                                   │
│  - Document selection                               │
│  - API configuration                                │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│            RAG Backend (rag_backend.py)             │
│  - PDF text extraction                              │
│  - Text chunking (500 chars, 50 overlap)            │
│  - Embedding generation (BAAI/bge-base-en-v1.5)     │
│  - ChromaDB vector storage & retrieval              │
│  - OpenRouter LLM query                             │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│         Preprocessing (preprocess_pdf.py)           │
│  - Batch PDF processing                             │
│  - Detailed logging                                 │
│  - Statistics tracking                              │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ (recommended)
- OpenRouter API key ([get one here](https://openrouter.ai/))
- 4GB+ RAM for embedding model

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/dataelvisliang/Book-RAG.git
cd Book-RAG/pdf-rag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Preprocess your PDFs**:
```bash
# Process PDFs from default "sample book" folder
python3.12 preprocess_pdf.py

# Or specify a custom folder/file
python3.12 preprocess_pdf.py /path/to/pdfs
python3.12 preprocess_pdf.py /path/to/document.pdf

# With custom database path and model
python3.12 preprocess_pdf.py --db-path ./my_db --model BAAI/bge-base-en-v1.5
```

4. **Run the Streamlit app**:
```bash
# Using the run script
./run.sh

# Or directly
python3.12 -m streamlit run app.py
```

5. **Open your browser** at `http://localhost:8501`

## 📖 Usage

### Preprocessing PDFs

The preprocessing script processes PDFs and stores embeddings in ChromaDB:

```bash
python3.12 preprocess_pdf.py [pdf_path] [--db-path PATH] [--model MODEL]
```

**What it does**:
- Extracts text from PDFs page by page
- Splits text into 500-character chunks with 50-character overlap
- Generates 768-dimensional embeddings using BAAI/bge-base-en-v1.5
- Stores embeddings in ChromaDB with metadata (page numbers, chunk IDs)
- Creates detailed logs in `./logs/` directory

**Example output**:
```
============================================================
📄 Processing: data science for business.pdf
============================================================
📖 Extracting text from PDF...
   ✓ Extracted 395 pages

📊 Chunking Statistics:
   • Chunk size: 500 characters
   • Overlap: 50 characters
   • Total chunks: 2155
   • Avg chunks per page: 5.5
   • Avg characters per page: 2272

🔄 Generating embeddings...
   • Model: BAAI/bge-base-en-v1.5
   • Embedding dimension: 768
   • Processing 2155 chunks in batches...

✅ Successfully processed 395 pages and 2155 chunks
   • Stored in ChromaDB collection: pdf_data_science_for_business_pdf
```

### Using the Streamlit App

1. **Enter your OpenRouter API key** in the sidebar
2. **Select your preferred AI model** (Claude 3.5 Sonnet, GPT-4o, etc.)
3. **Select documents** to query from the sidebar checkboxes
4. **Ask questions** in the chat interface
5. **View sources** by expanding the "View Sources" section under each answer

## 🛠️ Technical Details

### Embedding Model
- **Model**: BAAI/bge-base-en-v1.5
- **Dimensions**: 768
- **Device**: MPS (Mac GPU), CUDA (NVIDIA GPU), or CPU
- **Normalization**: Enabled for BGE models

### Text Processing
- **Chunk Size**: 500 characters
- **Overlap**: 50 characters (10% of chunk size)
- **Cleaning**: Unicode normalization, special character handling

### Vector Database
- **Database**: ChromaDB (local, persistent)
- **Similarity**: Cosine similarity
- **Top-k Retrieval**: 5 most relevant chunks per query

### LLM Integration
- **Provider**: OpenRouter API
- **Supported Models**:
  - anthropic/claude-3.5-sonnet
  - openai/gpt-4o
  - google/gemini-pro-1.5
  - meta-llama/llama-3.1-70b-instruct
- **Context**: Retrieved chunks from vector search
- **Temperature**: 0.7
- **Max Tokens**: 2000

## 📁 Project Structure

```
pdf-rag/
├── app.py                    # Streamlit frontend
├── rag_backend.py           # Core RAG logic
├── preprocess_pdf.py        # PDF preprocessing script
├── requirements.txt         # Python dependencies
├── run.sh                   # Run script for Streamlit
├── .gitignore              # Git ignore rules
├── .env                    # Environment variables (gitignored)
├── chroma_db/              # Vector database (gitignored)
├── logs/                   # Preprocessing logs (gitignored)
└── sample book/            # Sample PDFs (gitignored)
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
OPENROUTER_API_KEY=your_api_key_here  # Optional, can enter in UI
```

### Preprocessing Options
```bash
python3.12 preprocess_pdf.py --help

Arguments:
  pdf_path              Path to PDF file or directory (default: sample book)
  --db-path PATH        Path to ChromaDB storage (default: ./chroma_db)
  --model MODEL         Embedding model name (default: BAAI/bge-base-en-v1.5)
```

## 📊 Performance

- **Embedding Speed**: ~2-5 chunks/second (depends on hardware)
- **Query Latency**: <1 second for retrieval, 2-10 seconds for LLM response
- **Storage**: ~1MB per 100 chunks (embeddings + metadata)
- **Memory Usage**: ~2GB for embedding model + document chunks

## Troubleshooting

### NumPy Version Error
```bash
pip install "numpy<2"
```

### Python 3.14 Compatibility
Use Python 3.12 instead:
```bash
python3.12 -m pip install -r requirements.txt
python3.12 -m streamlit run app.py
```

### Encoding Errors
The system automatically handles Unicode encoding issues. Check logs for details on skipped chunks.

### ChromaDB Errors
Delete and rebuild the database:
```bash
rm -rf chroma_db/
python3.12 preprocess_pdf.py
```

## 🎯 Roadmap

- [ ] Add reranking step for better retrieval accuracy
- [ ] Support for multiple embedding models
- [ ] Hybrid search (dense + sparse)
- [ ] Document versioning and updates
- [ ] Multi-document query support
- [ ] Conversation memory and context
- [ ] Export chat history

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BAAI** for the bge-base-en-v1.5 embedding model
- **ChromaDB** for the vector database
- **Sentence Transformers** for the embedding framework
- **Streamlit** for the web framework
- **OpenRouter** for LLM API access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using Python, ChromaDB, and OpenRouter**
