"""
Standalone PDF Preprocessing Script
Processes PDFs and stores embeddings in ChromaDB for the RAG system.
"""

import argparse
import os
import logging
from datetime import datetime
from pathlib import Path
from rag_backend import RAGBackend


def setup_logging(log_dir="./logs"):
    """Setup logging to both file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"preprocess_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


def preprocess_pdf(pdf_path, backend):
    """
    Preprocess a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        backend: RAGBackend instance

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        pdf_name = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"📄 Processing: {pdf_name}")
        print(f"{'='*60}")
        logging.info(f"Processing PDF: {pdf_name}")

        # Read PDF file
        with open(pdf_path, 'rb') as f:
            # Create a simple file-like object that has getvalue()
            class FileWrapper:
                def __init__(self, file_obj):
                    self.content = file_obj.read()

                def getvalue(self):
                    return self.content

            pdf_file = FileWrapper(f)

            # Extract pages first to show stats
            print(f"📖 Extracting text from PDF...")
            logging.info("Extracting text from PDF...")
            pages = backend.extract_text_from_pdf(pdf_file)

            if not pages:
                logging.error("Could not extract text from PDF")
                return False, "Could not extract text from PDF"

            print(f"   ✓ Extracted {len(pages)} pages")
            logging.info(f"Extracted {len(pages)} pages")

            # Count total chunks
            total_chunks = 0
            total_chars = 0
            for page in pages:
                page_chunks = backend.chunk_text(page['text'])
                total_chunks += len(page_chunks)
                total_chars += len(page['text'])

            avg_chars_per_page = total_chars // len(pages) if pages else 0
            avg_chunks_per_page = total_chunks / len(pages) if pages else 0

            print(f"\n📊 Chunking Statistics:")
            print(f"   • Chunk size: 500 characters")
            print(f"   • Overlap: 50 characters")
            print(f"   • Total chunks: {total_chunks}")
            print(f"   • Avg chunks per page: {avg_chunks_per_page:.1f}")
            print(f"   • Avg characters per page: {avg_chars_per_page}")

            logging.info(f"Chunking Statistics:")
            logging.info(f"  - Chunk size: 500 characters")
            logging.info(f"  - Overlap: 50 characters")
            logging.info(f"  - Total chunks: {total_chunks}")
            logging.info(f"  - Avg chunks per page: {avg_chunks_per_page:.1f}")
            logging.info(f"  - Avg characters per page: {avg_chars_per_page}")
            logging.info(f"  - Total characters: {total_chars}")

            # Now process and store
            print(f"\n🔄 Generating embeddings...")
            print(f"   • Model: BAAI/bge-base-en-v1.5")
            print(f"   • Embedding dimension: 768")
            print(f"   • Processing {total_chunks} chunks in batches...")

            logging.info("Generating embeddings...")
            logging.info(f"  - Model: BAAI/bge-base-en-v1.5")
            logging.info(f"  - Embedding dimension: 768")
            logging.info(f"  - Processing {total_chunks} chunks in batches")

            # Reset file pointer
            pdf_file_reset = FileWrapper(open(pdf_path, 'rb'))
            success, message = backend.process_and_store_pdf(pdf_file_reset, pdf_name)

            if success:
                print(f"\n✅ {message}")
                print(f"   • Stored in ChromaDB collection: pdf_{pdf_name.replace(' ', '_').replace('.', '_')}")
                logging.info(f"SUCCESS: {message}")
                logging.info(f"Stored in ChromaDB collection: pdf_{pdf_name.replace(' ', '_').replace('.', '_')}")
            else:
                print(f"\n❌ {message}")
                logging.error(f"FAILED: {message}")

            return success, message

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Error processing {pdf_path}: {e}", exc_info=True)
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description='Preprocess PDFs for RAG system')
    parser.add_argument('pdf_path', type=str, nargs='?', default='sample book',
                       help='Path to PDF file or directory containing PDFs (default: sample book)')
    parser.add_argument('--db-path', type=str, default='./chroma_db',
                       help='Path to ChromaDB storage (default: ./chroma_db)')
    parser.add_argument('--model', type=str, default='BAAI/bge-base-en-v1.5',
                       help='Embedding model name (default: BAAI/bge-base-en-v1.5)')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()
    logging.info("="*60)
    logging.info("PDF Preprocessing Started")
    logging.info("="*60)
    logging.info(f"Log file: {log_file}")

    # Initialize backend
    print(f"\n{'='*60}")
    print(f"🚀 Initializing RAG Backend")
    print(f"{'='*60}")
    print(f"📂 Database path: {args.db_path}")
    print(f"🤖 Embedding model: {args.model}")
    print(f"⏳ Loading model (this may take a moment)...")

    logging.info(f"Database path: {args.db_path}")
    logging.info(f"Embedding model: {args.model}")
    logging.info("Loading embedding model...")

    backend = RAGBackend(
        persist_directory=args.db_path,
        embedding_model_name=args.model
    )
    print(f"✅ Backend initialized!\n")
    logging.info("Backend initialized successfully")

    # Process PDFs
    pdf_path = Path(args.pdf_path)

    if pdf_path.is_file():
        # Single PDF file
        if pdf_path.suffix.lower() == '.pdf':
            logging.info(f"Processing single PDF file: {pdf_path}")
            success, message = preprocess_pdf(str(pdf_path), backend)
            if success:
                print(f"✓ {message}")
                logging.info(f"✓ {message}")
            else:
                print(f"✗ {message}")
                logging.error(f"✗ {message}")
        else:
            error_msg = f"Error: {pdf_path} is not a PDF file"
            print(error_msg)
            logging.error(error_msg)

    elif pdf_path.is_dir():
        # Directory of PDFs
        pdf_files = list(pdf_path.glob('*.pdf'))

        if not pdf_files:
            error_msg = f"No PDF files found in {pdf_path}"
            print(error_msg)
            logging.warning(error_msg)
            return

        print(f"\nFound {len(pdf_files)} PDF files")
        print("-" * 60)
        logging.info(f"Found {len(pdf_files)} PDF files in directory: {pdf_path}")

        success_count = 0
        fail_count = 0

        for pdf_file in pdf_files:
            success, message = preprocess_pdf(str(pdf_file), backend)

            if success:
                print(f"✓ {pdf_file.name}: {message}")
                success_count += 1
            else:
                print(f"✗ {pdf_file.name}: {message}")
                fail_count += 1

        print("-" * 60)
        print(f"\nSummary: {success_count} succeeded, {fail_count} failed")
        logging.info("="*60)
        logging.info(f"Processing Summary: {success_count} succeeded, {fail_count} failed")
        logging.info("="*60)

    else:
        error_msg = f"Error: {pdf_path} does not exist"
        print(error_msg)
        logging.error(error_msg)
        return

    print("\nPreprocessing complete! Documents are ready for querying.")
    logging.info("Preprocessing complete! Documents are ready for querying.")


if __name__ == "__main__":
    main()
