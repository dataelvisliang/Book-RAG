"""
RAG Backend - Handles PDF processing, embeddings, and retrieval
"""

import os
import tempfile
import logging
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests


class RAGBackend:
    """Backend for RAG operations: PDF processing, embedding, and retrieval."""

    def __init__(self, persist_directory="./chroma_db", embedding_model_name="BAAI/bge-base-en-v1.5"):
        """
        Initialize the RAG backend.

        Args:
            persist_directory: Directory to store ChromaDB data
            embedding_model_name: Name of the sentence-transformers model to use
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

        # Initialize ChromaDB client with persistence
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load reranker model (lazy loading)
        self.reranker = None

    def extract_text_from_pdf(self, pdf_file):
        """
        Extract text from PDF file page by page.

        Args:
            pdf_file: Uploaded file object (from Streamlit)

        Returns:
            List of dicts with 'page_number' and 'text'
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name

            # Read PDF
            pdf_reader = PdfReader(tmp_path)
            pages = []

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text.strip()
                    })

            # Clean up temp file
            os.unlink(tmp_path)

            return pages
        except Exception as e:
            raise Exception(f"Error extracting PDF: {e}")

    def clean_text(self, text):
        """Clean and normalize text for processing.

        Args:
            text: Input text to clean

        Returns:
            Cleaned and normalized text
        """
        import re
        import unicodedata

        if not isinstance(text, str):
            text = str(text)

        # Remove surrogate characters (unpaired Unicode surrogates like \ud835)
        # These are often from mathematical symbols or special characters
        text = ''.join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))

        # Normalize unicode characters
        try:
            text = unicodedata.normalize('NFKC', text)
        except ValueError:
            # If normalization fails, try encoding/decoding to clean
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

        # Replace problematic unicode quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2013', '-').replace('\u2014', '-')

        # Remove other problematic characters (control characters)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Normalize whitespace
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Remove spaces at the beginning/end of lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # Remove empty lines
        text = '\n'.join(line for line in text.split('\n') if line)

        return text.strip()

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk (will be converted to string if not already)
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        # Ensure text is a string and clean it
        text = self.clean_text(str(text))
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        # Adjust chunk size if it's larger than the text
        chunk_size = min(chunk_size, text_length)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
                
            # Break if we've reached the end
            if end == text_length:
                break
                
            # Move the start position, accounting for overlap
            start = end - min(overlap, chunk_size - 1)

        return chunks

    def process_and_store_pdf(self, pdf_file, pdf_name):
        """
        Process PDF and store embeddings in ChromaDB.

        Args:
            pdf_file: Uploaded PDF file
            pdf_name: Name of the PDF

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Get or create collection for this document
            collection_name = f"pdf_{pdf_name.replace(' ', '_').replace('.', '_')}"

            # Delete collection if it exists (re-upload scenario)
            try:
                self.client.delete_collection(collection_name)
            except:
                pass

            # Create friendly display name (remove .pdf extension and title case)
            display_name = pdf_name.replace('.pdf', '').replace('_', ' ').strip()
            # Title case each word
            display_name = ' '.join(word.capitalize() for word in display_name.split())

            collection = self.client.create_collection(
                name=collection_name,
                metadata={"document_name": display_name}
            )

            # Extract text from PDF
            pages = self.extract_text_from_pdf(pdf_file)

            if not pages:
                return False, "Could not extract text from PDF"

            # Chunk and embed each page
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []

            chunk_id = 0

            for page in pages:
                page_text = page['text']
                page_number = page['page_number']

                # Chunk the page text
                chunks = self.chunk_text(page_text)

                for chunk in chunks:
                    # Ensure chunk is a proper string and not empty
                    chunk_str = str(chunk).strip()
                    if not chunk_str:
                        continue

                    all_chunks.append(chunk_str)
                    all_metadatas.append({
                        'page_number': page_number,
                        'chunk_id': chunk_id
                    })
                    all_ids.append(f"{pdf_name}_page{page_number}_chunk{chunk_id}")

                    chunk_id += 1

            # Generate embeddings in batch (more efficient)
            if all_chunks:
                try:
                    # Clean and validate chunks
                    logging.info(f"Cleaning {len(all_chunks)} chunks...")
                    cleaned_chunks = []
                    for i, chunk in enumerate(all_chunks):
                        try:
                            # Clean the chunk
                            clean_chunk = self.clean_text(chunk)
                            if clean_chunk:  # Only keep non-empty chunks
                                cleaned_chunks.append(clean_chunk)
                        except Exception as e:
                            logging.warning(f"Error cleaning chunk {i}: {str(e)}")
                            continue

                    logging.info(f"Successfully cleaned {len(cleaned_chunks)} chunks (removed {len(all_chunks) - len(cleaned_chunks)} empty/invalid chunks)")

                    if not cleaned_chunks:
                        raise ValueError("No valid text chunks to process after cleaning")

                    # Process in smaller batches to avoid memory issues
                    batch_size = min(16, len(cleaned_chunks))  # Reduced batch size
                    all_embeddings = []
                    total_batches = (len(cleaned_chunks) + batch_size - 1) // batch_size
                    logging.info(f"Generating embeddings in {total_batches} batches (batch size: {batch_size})...")

                    for batch_idx, i in enumerate(range(0, len(cleaned_chunks), batch_size), 1):
                        batch = cleaned_chunks[i:i + batch_size]
                        logging.info(f"Processing batch {batch_idx}/{total_batches} ({len(batch)} chunks)...")

                        # Extra validation: ensure all chunks are proper strings
                        validated_batch = []
                        for idx, chunk in enumerate(batch):
                            if not isinstance(chunk, str):
                                logging.warning(f"Chunk {i+idx} is not a string, type: {type(chunk)}")
                                chunk = str(chunk)
                            # Ensure it's ASCII-compatible or properly encoded
                            try:
                                # Try encoding/decoding to catch encoding issues
                                chunk.encode('utf-8').decode('utf-8')
                                validated_batch.append(chunk)
                            except Exception as e:
                                logging.debug(f"Chunk {i+idx} has encoding issues: {str(e)[:100]}")
                                # Try to clean it more aggressively
                                cleaned = chunk.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                                if cleaned.strip():
                                    validated_batch.append(cleaned)

                        if not validated_batch:
                            continue

                        try:
                            # BGE models recommend normalization
                            batch_embeddings = self.embedding_model.encode(
                                validated_batch,
                                convert_to_tensor=False,
                                show_progress_bar=False,
                                normalize_embeddings=True,
                                batch_size=8  # Smaller batch size for stability
                            )
                            all_embeddings.extend(batch_embeddings.tolist())
                            logging.info(f"✓ Batch {batch_idx}/{total_batches} complete ({len(all_embeddings)}/{len(cleaned_chunks)} embeddings generated)")
                        except Exception as e:
                            logging.warning(f"Error in batch {batch_idx}: {str(e)}")
                            # Try processing individual chunks in the batch
                            for j, chunk in enumerate(validated_batch):
                                try:
                                    # Verify chunk is valid string
                                    if not chunk or not chunk.strip():
                                        logging.debug(f"Skipping empty chunk {i+j}")
                                        continue

                                    chunk_embedding = self.embedding_model.encode(
                                        chunk,  # Pass string directly, not in a list
                                        convert_to_tensor=False,
                                        show_progress_bar=False,
                                        normalize_embeddings=True
                                    )
                                    all_embeddings.append(chunk_embedding.tolist())
                                except Exception as chunk_error:
                                    logging.warning(f"Error processing chunk {i+j}: {str(chunk_error)}")
                                    logging.debug(f"Chunk preview: {repr(chunk[:100])}")
                    
                    if not all_embeddings:
                        raise ValueError("Failed to generate any embeddings")

                    logging.info(f"Embedding generation complete: {len(all_embeddings)} vectors created")

                    # Update all_chunks to match the successfully processed chunks
                    all_chunks = cleaned_chunks[:len(all_embeddings)]
                except Exception as e:
                    error_msg = f"Error encoding chunks. "
                    if all_chunks:
                        error_msg += f"Num chunks: {len(all_chunks)}, "
                        error_msg += f"First chunk type: {type(all_chunks[0])}, "
                        error_msg += f"First chunk: {str(all_chunks[0])[:100]}. "
                    error_msg += f"Error: {str(e)}"
                    raise Exception(error_msg)

            # Store in ChromaDB
            logging.info(f"Storing {len(all_embeddings)} embeddings in ChromaDB collection '{collection_name}'...")
            collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logging.info(f"✓ Successfully stored all embeddings in ChromaDB")

            return True, f"Successfully processed {len(pages)} pages and {len(all_chunks)} chunks"

        except Exception as e:
            return False, f"Error processing PDF: {e}"

    def generate_query_rewrites(self, query, api_key, model="nvidia/nemotron-3-nano-30b-a3b:free", book_context="Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking by Foster Provost and Tom Fawcett"):
        """
        Generate multiple alternative search queries (Multi-Query approach).

        Args:
            query: User's question
            api_key: OpenRouter API key
            model: Model to use for generation
            book_context: Context about the book being searched

        Returns:
            List of rewritten queries
        """
        try:
            logging.info(f"Generating query rewrites for: {query[:100]}...")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are generating search queries to retrieve passages from '{book_context}'.\n\n"
                        "This book emphasizes data-analytic thinking for business problems. Key topics include:\n"
                        "- Framing business problems as data mining tasks\n"
                        "- Supervised learning: classification, regression, probability estimation\n"
                        "- Evaluation metrics: accuracy, precision, recall, lift, ROI\n"
                        "- Model complexity: overfitting, generalization, training vs test error\n"
                        "- Data issues: leakage, selection bias, missing data\n"
                        "- Decision-making: expected value, costs and benefits\n"
                        "- Specific algorithms: trees, logistic regression, similarity-based methods\n\n"
                        "Rewrite the user question into THREE alternative search queries:\n\n"
                        "1. **Data Mining Terminology** - Use book-specific terms like 'supervised learning', 'target variable', "
                        "'training set', 'holdout data', 'model induction', 'attribute', 'instance'\n\n"
                        "2. **Business Decision Focus** - Frame in terms of business value, ROI, expected value, costs/benefits, "
                        "decision-making, targeting, segmentation, risk assessment\n\n"
                        "3. **Analytical Thinking** - Focus on fundamental concepts: patterns in data, generalization, "
                        "signal vs noise, predictive modeling, data-driven decisions\n\n"
                        "Constraints:\n"
                        "- Do NOT answer the question, only rewrite it\n"
                        "- Do NOT mention: deep learning, neural networks, LLMs, GPT, transformers, or modern frameworks\n"
                        "- Use terminology from traditional machine learning (pre-2015)\n"
                        "- Each rewrite: 10-20 words, search-optimized\n"
                        "- Preserve the core question meaning\n\n"
                        "Output format: Return ONLY a JSON array of 3 strings, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"User question:\n{query}\n\nGenerate 3 search query rewrites:"
                }
            ]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.5,  # Moderate creativity for variations
                "max_tokens": 1500
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            rewrites_text = result["choices"][0]["message"]["content"]

            # Parse JSON array
            import json
            try:
                rewrites = json.loads(rewrites_text)
                if not isinstance(rewrites, list) or len(rewrites) != 3:
                    raise ValueError("Expected array of 3 rewrites")

                logging.info(f"Generated {len(rewrites)} query rewrites")
                for i, rewrite in enumerate(rewrites, 1):
                    logging.debug(f"  Rewrite {i}: {rewrite}")

                return rewrites

            except json.JSONDecodeError:
                logging.warning("Failed to parse rewrites as JSON, attempting line-by-line parsing")
                # Fallback: try to extract lines
                lines = [line.strip().strip('"').strip("'") for line in rewrites_text.split('\n') if line.strip()]
                rewrites = [line for line in lines if line and not line.startswith('[') and not line.startswith(']')][:3]

                if len(rewrites) >= 1:
                    logging.info(f"Extracted {len(rewrites)} rewrites from text")
                    return rewrites
                else:
                    raise ValueError("Could not extract rewrites")

        except Exception as e:
            logging.warning(f"Failed to generate query rewrites: {e}")
            logging.info("Falling back to original query")
            return None

    def generate_hypothetical_document(self, query, api_key, model="nvidia/nemotron-3-nano-30b-a3b:free"):
        """
        Generate a hypothetical document that would answer the query (HyDE).

        Args:
            query: User's question
            api_key: OpenRouter API key
            model: Model to use for generation

        Returns:
            Generated hypothetical document text
        """
        try:
            logging.info(f"Generating hypothetical document for query: {query[:100]}...")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are writing content for 'Data Science for Business' by Foster Provost and Tom Fawcett. "
                        "This book focuses on data-analytic thinking and how data science connects to business decision-making. "
                        "Given a question, write a detailed paragraph in the style of this book that would answer the question. "
                        "Focus on business implications, decision-making frameworks, and practical application. "
                        "Write in an informative, encyclopedic style. Do not use phrases like 'the answer is' or 'in conclusion'. "
                        "Just write the factual content directly as it would appear in the book."
                    )
                },
                {
                    "role": "user",
                    "content": f"Write a detailed paragraph from 'Data Science for Business' that would answer this question: {query}"
                }
            ]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,  # Lower temperature for more focused content
                "max_tokens": 1500
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            hypothetical_doc = result["choices"][0]["message"]["content"]

            logging.info(f"Generated hypothetical document ({len(hypothetical_doc)} chars)")
            logging.debug(f"Hypothetical document: {hypothetical_doc[:200]}...")

            return hypothetical_doc

        except Exception as e:
            logging.warning(f"Failed to generate hypothetical document: {e}")
            logging.info("Falling back to direct query embedding")
            return None

    def _rerank_results(self, query, results, top_k):
        """Rerank results using cross-encoder reranker."""
        if self.reranker is None:
            logging.info("Loading reranker model (BAAI/bge-reranker-v2-m3)...")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
            logging.info("Reranker loaded")

        # Prepare pairs for reranking
        pairs = [[query, result['text']] for result in results]

        # Get reranking scores
        logging.info(f"Reranking {len(results)} results...")
        scores = self.reranker.predict(pairs)

        # Add scores to results
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])

        # Sort by rerank score (higher is better)
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        logging.info(f"Reranking complete, returning top {top_k}")
        return results[:top_k]

    def retrieve_relevant_chunks(self, query, collection_names, top_k=5, rewrite_mode="none", api_key=None, rewrite_model="nvidia/nemotron-3-nano-30b-a3b:free", rerank=True):
        """
        Retrieve relevant chunks from selected documents.

        Args:
            query: User's question
            collection_names: List of collection names to search
            top_k: Number of top results to return
            rewrite_mode: Query rewriting strategy ("none", "hyde", "multi_query")
            api_key: OpenRouter API key (required if rewrite_mode != "none")
            rewrite_model: Model to use for query rewriting

        Returns:
            List of dicts with 'text', 'page_number', 'distance', 'collection'
        """
        try:
            # Determine queries to embed based on rewrite mode
            queries_to_embed = [query]  # Default: just the original query

            if rewrite_mode == "hyde" and api_key:
                # Generate hypothetical document
                hypothetical_doc = self.generate_hypothetical_document(query, api_key, rewrite_model)

                if hypothetical_doc:
                    queries_to_embed = [hypothetical_doc]
                    logging.info("Using HyDE: embedding hypothetical document instead of query")
                else:
                    logging.info("HyDE generation failed, using original query")

            elif rewrite_mode == "multi_query" and api_key:
                # Generate multiple query rewrites
                rewrites = self.generate_query_rewrites(query, api_key, rewrite_model)

                if rewrites:
                    queries_to_embed = rewrites
                    logging.info(f"Using Multi-Query: searching with {len(rewrites)} query variations")
                else:
                    logging.info("Multi-Query generation failed, using original query")

            else:
                logging.info(f"Using direct query embedding (mode: {rewrite_mode})")

            # Generate embeddings for all queries
            all_results = []

            for query_idx, text_to_embed in enumerate(queries_to_embed):
                logging.debug(f"Embedding query {query_idx + 1}/{len(queries_to_embed)}: {text_to_embed[:100]}...")

                # Generate query embedding
                query_embedding = self.embedding_model.encode(
                    str(text_to_embed),
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ).tolist()

                # Search each collection
                for collection_name in collection_names:
                    try:
                        collection = self.client.get_collection(collection_name)
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=top_k
                        )

                        # Format results
                        if results and results['documents']:
                            for i in range(len(results['documents'][0])):
                                all_results.append({
                                    'text': results['documents'][0][i],
                                    'page_number': results['metadatas'][0][i]['page_number'],
                                    'distance': results['distances'][0][i],
                                    'collection': collection_name
                                })
                    except Exception as e:
                        # Skip collections that have errors
                        continue

            # Deduplicate
            seen_texts = {}
            for result in all_results:
                text_key = result['text'][:100]
                if text_key not in seen_texts or result['distance'] < seen_texts[text_key]['distance']:
                    seen_texts[text_key] = result

            deduplicated_results = list(seen_texts.values())

            # Rerank if enabled
            if rerank and len(deduplicated_results) > 0:
                deduplicated_results = self._rerank_results(query, deduplicated_results, top_k)
            else:
                deduplicated_results.sort(key=lambda x: x['distance'])
                deduplicated_results = deduplicated_results[:top_k]

            return deduplicated_results

        except Exception as e:
            raise Exception(f"Error retrieving chunks: {e}")

    def get_all_collections(self):
        """
        Get all document collections.

        Returns:
            List of ChromaDB collections
        """
        return self.client.list_collections()

    def query_openrouter(self, message, context, api_key, model):
        """
        Generate answer using OpenRouter API.

        Args:
            message: User's question
            context: Retrieved context from documents
            api_key: OpenRouter API key
            model: Model name to use

        Returns:
            Generated answer text
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant helping readers understand 'Data Science for Business: What You Need to Know about "
                        "Data Mining and Data-Analytic Thinking' by Foster Provost and Tom Fawcett.\n\n"
                        "This book focuses on data-analytic thinking - reasoning about problems so data science is useful for business decisions. "
                        "Key themes include: framing business problems as data science problems, prediction/classification/ranking/clustering, "
                        "evaluation metrics (accuracy vs precision/recall vs lift), overfitting, causality vs correlation, data leakage, "
                        "and how data science connects to decision-making and value creation.\n\n"
                        "Answer questions based on the provided context from the book. If the context doesn't contain the answer, say so clearly. "
                        "Reference specific pages when citing information (e.g., 'According to page 5...'). "
                        "Focus on explaining concepts in terms of business decision-making and practical application, not just technical details."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context from the book:\n\n{context}\n\n"
                        f"Question: {message}\n\n"
                        "IMPORTANT: When citing information from the context, add inline citations using [1], [2], etc. "
                        "At the end of your answer, include a 'References:' section listing each citation with its page number. "
                        "Example format:\n"
                        "Your answer text with citation[1]. More information here[2].\n\n"
                        "References:\n"
                        "[1] Page 42\n"
                        "[2] Page 87"
                    )
                }
            ]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            raise Exception(f"OpenRouter error: {e}")
