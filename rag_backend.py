"""
RAG Backend - Handles PDF processing, embeddings, and retrieval
"""

import os
import tempfile
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

        # Initialize ChromaDB client
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

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
            
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Replace problematic unicode quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        
        # Remove other problematic characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
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

            collection = self.client.create_collection(
                name=collection_name,
                metadata={"document_name": pdf_name}
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
                    cleaned_chunks = []
                    for i, chunk in enumerate(all_chunks):
                        try:
                            # Clean the chunk
                            clean_chunk = self.clean_text(chunk)
                            if clean_chunk:  # Only keep non-empty chunks
                                cleaned_chunks.append(clean_chunk)
                        except Exception as e:
                            print(f"Warning: Error cleaning chunk {i}: {str(e)}")
                            continue
                    
                    if not cleaned_chunks:
                        raise ValueError("No valid text chunks to process after cleaning")
                    
                    # Process in smaller batches to avoid memory issues
                    batch_size = min(16, len(cleaned_chunks))  # Reduced batch size
                    all_embeddings = []

                    for i in range(0, len(cleaned_chunks), batch_size):
                        batch = cleaned_chunks[i:i + batch_size]

                        # Extra validation: ensure all chunks are proper strings
                        validated_batch = []
                        for idx, chunk in enumerate(batch):
                            if not isinstance(chunk, str):
                                print(f"Warning: chunk {i+idx} is not a string, type: {type(chunk)}")
                                chunk = str(chunk)
                            # Ensure it's ASCII-compatible or properly encoded
                            try:
                                # Try encoding/decoding to catch encoding issues
                                chunk.encode('utf-8').decode('utf-8')
                                validated_batch.append(chunk)
                            except Exception as e:
                                print(f"Warning: chunk {i+idx} has encoding issues: {str(e)[:100]}")
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
                        except Exception as e:
                            print(f"Error in batch {i//batch_size}: {str(e)}")
                            # Try processing individual chunks in the batch
                            for j, chunk in enumerate(validated_batch):
                                try:
                                    # Verify chunk is valid string
                                    if not chunk or not chunk.strip():
                                        print(f"Skipping empty chunk {i+j}")
                                        continue

                                    chunk_embedding = self.embedding_model.encode(
                                        chunk,  # Pass string directly, not in a list
                                        convert_to_tensor=False,
                                        show_progress_bar=False,
                                        normalize_embeddings=True
                                    )
                                    all_embeddings.append(chunk_embedding.tolist())
                                except Exception as chunk_error:
                                    print(f"Error processing chunk {i+j}: {str(chunk_error)}")
                                    print(f"Chunk preview: {repr(chunk[:100])}")
                    
                    if not all_embeddings:
                        raise ValueError("Failed to generate any embeddings")
                        
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
            collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )

            return True, f"Successfully processed {len(pages)} pages and {len(all_chunks)} chunks"

        except Exception as e:
            return False, f"Error processing PDF: {e}"

    def retrieve_relevant_chunks(self, query, collection_names, top_k=5):
        """
        Retrieve relevant chunks from selected documents.

        Args:
            query: User's question
            collection_names: List of collection names to search
            top_k: Number of top results to return

        Returns:
            List of dicts with 'text', 'page_number', 'distance', 'collection'
        """
        try:
            # Generate query embedding
            # BGE models recommend normalization
            # Pass the query string directly (not in a list)
            query_embedding = self.embedding_model.encode(
                str(query),
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            all_results = []

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

            # Sort by distance (lower is better) and return top_k
            all_results.sort(key=lambda x: x['distance'])
            return all_results[:top_k]

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
                    "content": "You are a helpful AI assistant that answers questions based on the provided context from documents. "
                              "If you don't know the answer based on the context, say so instead of making something up. "
                              "When you use information from the context, reference it by mentioning the source (e.g., 'According to page 5...')."
                },
                {
                    "role": "user",
                    "content": f"Context from documents:\n\n{context}\n\nQuestion: {message}"
                }
            ]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
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
