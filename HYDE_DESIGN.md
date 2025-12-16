# HyDE (Hypothetical Document Embeddings) Design

## Overview

HyDE is a query rewriting technique that significantly improves retrieval quality by generating a hypothetical document that would answer the user's question, then using that document's embedding to search the vector database instead of the query embedding.

## Why HyDE Works

### The Problem
Traditional semantic search embeds the query and searches for similar documents. However:
- **Queries are short**: "What is overfitting?"
- **Documents are detailed**: Long explanations with specific terminology
- **Style mismatch**: Questions vs. declarative text

This creates a semantic gap that reduces retrieval quality.

### The Solution
HyDE bridges this gap by:
1. **Query → Hypothetical Answer**: Generate what an answer would look like
2. **Embed the Answer**: Create embedding from the hypothetical document
3. **Search with Answer Embedding**: Find documents similar to answers, not questions

### Example

**User Query:**
```
"What causes overfitting in machine learning?"
```

**Traditional Approach:**
- Embeds the question directly
- Searches for documents containing similar question-like text
- May miss documents that explain overfitting without using the word "causes"

**HyDE Approach:**
- Generates hypothetical answer:
  ```
  "Overfitting occurs when a machine learning model learns the training
  data too well, including noise and outliers. This happens when the model
  is too complex relative to the amount of training data, when there is
  insufficient regularization, or when the model is trained for too many
  epochs. Overfitting results in excellent performance on training data
  but poor generalization to unseen test data."
  ```
- Embeds this detailed explanation
- Searches for documents with similar explanatory content
- **Much better retrieval quality!**

## Implementation Details

### 1. Hypothetical Document Generation

**Location:** `rag_backend.py:generate_hypothetical_document()`

**System Prompt:**
```
You are a helpful assistant writing content for a technical book.
Given a question, write a detailed paragraph that would appear in
a book to answer this question. Write in an informative, encyclopedic
style. Do not use phrases like 'the answer is' or 'in conclusion'.
Just write the factual content directly as it would appear in a textbook.
```

**User Prompt:**
```
Write a detailed paragraph from a technical book that would answer
this question: {user_query}
```

**Parameters:**
- **Model**: Configurable (default: nvidia/nemotron-3-nano-30b-a3b:free)
- **Temperature**: 0.3 (lower = more focused, less creative)
- **Max Tokens**: 300 (concise but informative)
- **Timeout**: 30 seconds

### 2. Retrieval Flow

```
User Query
    ↓
[HyDE Enabled?] → No → Embed Query Directly
    ↓ Yes
Generate Hypothetical Document (LLM API Call)
    ↓
[Generation Successful?] → No → Fallback to Query Embedding
    ↓ Yes
Embed Hypothetical Document
    ↓
Search Vector Database
    ↓
Return Top-K Results
```

### 3. Error Handling

**Graceful Degradation:**
- If HyDE generation fails → Falls back to direct query embedding
- If API key missing → Uses query embedding
- If network timeout → Uses query embedding
- Always returns results (never breaks retrieval)

**Logging:**
```python
logging.info("Generating hypothetical document for query...")
logging.info("Generated hypothetical document (256 chars)")
logging.debug("Hypothetical document: [content preview]...")
logging.warning("Failed to generate hypothetical document: [error]")
logging.info("Falling back to direct query embedding")
```

## Usage

### In Streamlit App

**Enable/Disable:**
```python
use_hyde = st.checkbox(
    "Enable HyDE",
    value=True,  # Enabled by default
    help="HyDE improves retrieval by generating a hypothetical answer first"
)
```

**Retrieval Call:**
```python
sources = backend.retrieve_relevant_chunks(
    query,
    collection_names,
    use_hyde=True,
    api_key=api_key,
    hyde_model="nvidia/nemotron-3-nano-30b-a3b:free"
)
```

### Direct API Usage

```python
from rag_backend import RAGBackend

backend = RAGBackend()

# With HyDE
results = backend.retrieve_relevant_chunks(
    query="What is gradient descent?",
    collection_names=["pdf_ml_book_pdf"],
    use_hyde=True,
    api_key="your-openrouter-key",
    hyde_model="nvidia/nemotron-3-nano-30b-a3b:free",
    top_k=5
)

# Without HyDE (traditional search)
results = backend.retrieve_relevant_chunks(
    query="What is gradient descent?",
    collection_names=["pdf_ml_book_pdf"],
    use_hyde=False,
    top_k=5
)
```

## Performance Considerations

### API Costs
- **HyDE adds 1 API call per query**
- Using free model (nemotron-3-nano-30b-a3b:free) = $0.00 per query
- Using paid model (gpt-4o) = ~$0.002 per query

### Latency
- **Hypothetical document generation**: ~1-3 seconds
- **Embedding**: ~0.1 seconds (local)
- **Vector search**: ~0.05 seconds
- **Total overhead**: ~1-3 seconds per query

### Quality vs. Speed Tradeoff
- **With HyDE**: Better results, slower queries
- **Without HyDE**: Faster queries, potentially worse results
- **Recommendation**: Enable by default, let users disable if speed is critical

## Expected Improvements

Based on HyDE research papers and real-world usage:

- **20-40% improvement** in retrieval quality (measured by relevance)
- **Better handling** of:
  - Vague or ambiguous queries
  - Questions with implicit context
  - Multi-hop reasoning queries
  - Domain-specific terminology mismatches

### When HyDE Helps Most
1. **Complex questions** requiring detailed explanations
2. **Conceptual queries** ("What is X?", "How does Y work?")
3. **Comparison questions** ("What's the difference between X and Y?")
4. **Causality questions** ("Why does X happen?", "What causes Y?")

### When HyDE Helps Less
1. **Keyword searches** ("Find mentions of 'gradient descent'")
2. **Fact lookups** ("Who invented X?")
3. **Short, specific queries** already well-matched to document style

## Testing HyDE

### Compare Results

```python
# Test script to compare with/without HyDE
query = "What causes overfitting in machine learning?"

# Without HyDE
results_no_hyde = backend.retrieve_relevant_chunks(
    query, ["pdf_ml_book"], use_hyde=False, top_k=5
)

# With HyDE
results_hyde = backend.retrieve_relevant_chunks(
    query, ["pdf_ml_book"], use_hyde=True,
    api_key="your-key", top_k=5
)

print("WITHOUT HyDE:")
for r in results_no_hyde[:3]:
    print(f"Distance: {r['distance']:.3f}")
    print(f"Text: {r['text'][:200]}...\n")

print("\nWITH HyDE:")
for r in results_hyde[:3]:
    print(f"Distance: {r['distance']:.3f}")
    print(f"Text: {r['text'][:200]}...\n")
```

## Future Enhancements

### 1. Multi-HyDE
Generate multiple hypothetical documents and use all embeddings:
```python
# Generate 3 variations
hyde_docs = [
    generate_hypothetical_document(query, temp=0.3),
    generate_hypothetical_document(query, temp=0.5),
    generate_hypothetical_document(query, temp=0.7),
]
# Average embeddings or retrieve with each
```

### 2. HyDE Caching
Cache hypothetical documents for repeated queries:
```python
hyde_cache = {}
cache_key = hash(query)
if cache_key in hyde_cache:
    return hyde_cache[cache_key]
```

### 3. Adaptive HyDE
Automatically decide when to use HyDE based on query characteristics:
```python
def should_use_hyde(query):
    # Use HyDE for questions, not for keyword searches
    question_words = ['what', 'how', 'why', 'when', 'who']
    return any(query.lower().startswith(w) for w in question_words)
```

### 4. HyDE + Query Expansion
Combine HyDE with traditional query expansion for best of both worlds.

## References

- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE Paper)](https://arxiv.org/abs/2212.10496)
- Research shows 5-20 point improvements in retrieval metrics
- Works across domains: medical, legal, technical, general knowledge

## Configuration

### Recommended Settings

**For Technical Books (Current Use Case):**
```python
use_hyde = True
hyde_model = "nvidia/nemotron-3-nano-30b-a3b:free"  # Free, good quality
temperature = 0.3  # Focused, factual content
max_tokens = 300   # Detailed but concise
```

**For General Knowledge:**
```python
temperature = 0.5
max_tokens = 200
```

**For Creative Content:**
```python
temperature = 0.7
max_tokens = 400
```
