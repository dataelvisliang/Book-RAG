# Changelog

All notable changes to the Book RAG system are documented in this file.

## [2.0.0] - 2025-12-16

### 🎉 Major Features

#### Multi-Book Support with Book-Specific Intelligence
- **Book-Specific Prompts**: Dynamic prompt adaptation based on selected books
  - Data Science for Business: Optimized for business analytics and decision-making
  - A Simple Guide to RAG: Optimized for RAG architecture and implementation
  - Generic prompts for multi-book scenarios
- **Friendly Display Names**: Automatic transformation of technical filenames to user-friendly titles
  - Example: `data_science_for_business.pdf` → `Data Science For Business`
- **Collection Metadata**: Store and retrieve display names for better UX

#### Streaming LLM Responses
- **Real-time Generation**: Word-by-word streaming with cursor effect (`▌`)
- **Immediate Feedback**: Users see responses as they're generated
- **Better UX**: Reduced perceived latency and improved engagement
- **SSE Protocol**: Server-Sent Events for OpenRouter streaming

#### Advanced Query Rewriting with Canonical Reformulation
- **Book-Specific Multi-Query**:
  - RAG Guide: RAG technical terms, implementation focus, architecture concepts
  - DS4B: Data mining terminology, business decision focus, analytical thinking
- **Book-Specific HyDE**:
  - RAG Guide: Technical guide style with system architecture focus
  - DS4B: Business analytics style with decision-making focus
- **Intent Preservation**: All rewrites maintain original user intent

#### GPU Acceleration
- **Auto-Detection**: Automatically detect and use CUDA (NVIDIA) or MPS (Mac) GPUs
- **Dual GPU Support**: Applied to both embedding and reranking models
- **Fallback**: Graceful CPU fallback when GPU unavailable
- **Performance**: 3-5x faster embedding and reranking on GPU

### 🎨 UI/UX Improvements

#### Lottie Animations
- **Header Animation**: Book animation in main title
- **Empty State**: Friendly animation when no documents uploaded
- **CSS Animations**: Smooth fade-in for messages, hover effects for controls

#### Enhanced Visual Design
- **Dual Scoring**: Display both distance and rerank scores for each source
- **Inline Citations**: [1], [2] format with References section
- **Improved Layout**: Better spacing and visual hierarchy

### 🔧 Technical Improvements

#### Retrieval Enhancements
- **Top-10 Retrieval**: Increased from 5 to 10 chunks for comprehensive context
- **Thorough Synthesis**: Instructions for LLM to synthesize across ALL retrieved chunks
- **Better Coverage**: More context leads to more complete answers

#### Prompt Engineering
- **Canonical Rewrites**: Focus on canonical terminology for robust retrieval
- **Intent Preservation**: Explicit constraints to maintain user intent
- **Book-Aware Context**: Prompts adapted to each book's domain and style

### 📝 Documentation Updates
- Updated README with new architecture diagram
- Added screenshots showing chat interface and citations
- Comprehensive feature documentation
- Updated roadmap with completed features

---

## [1.0.0] - 2024

### Initial Release
- Basic RAG system with ChromaDB
- HyDE and Multi-Query retrieval
- Cross-encoder reranking
- Streamlit chat interface
- OpenRouter LLM integration
- PDF preprocessing with chunking
- Text cleaning and normalization
