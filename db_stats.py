"""
Database Statistics
Shows size, shape, and detailed statistics of ChromaDB
"""

import chromadb
import os
from pathlib import Path
import argparse


def get_directory_size(path):
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Warning: Could not calculate size: {e}")
    return total


def format_bytes(bytes_size):
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description='Show ChromaDB statistics')
    parser.add_argument('--db-path', type=str, default='./chroma_db',
                       help='Path to ChromaDB storage (default: ./chroma_db)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ChromaDB Size and Shape Analysis")
    print("="*70 + "\n")

    # Database directory info
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database path '{db_path}' does not exist")
        return

    total_size = get_directory_size(db_path)
    print(f"Database Directory: {db_path.absolute()}")
    print(f"Total Disk Size: {format_bytes(total_size)}")
    print()

    # Connect to database
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collections = client.list_collections()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    print(f"Total Collections: {len(collections)}")
    print()

    if not collections:
        print("No collections found in database.")
        return

    # Analyze each collection
    total_items = 0
    total_embedding_size = 0

    for coll in collections:
        print("-" * 70)
        print(f"Collection: {coll.name}")
        print("-" * 70)

        count = coll.count()
        total_items += count
        print(f"  Total Items: {count:,}")

        # Get embedding dimension
        if count > 0:
            try:
                sample = coll.peek(1)

                # Get embedding dimension
                emb_dim = 0
                try:
                    if 'embeddings' in sample and sample['embeddings'] is not None:
                        if len(sample['embeddings']) > 0:
                            first_emb = sample['embeddings'][0]
                            if first_emb is not None:
                                emb_dim = len(first_emb)
                except (TypeError, ValueError, IndexError):
                    pass

                print(f"  Embedding Dimension: {emb_dim}")

                # Estimate embedding storage size (float32 = 4 bytes per value)
                embedding_size = count * emb_dim * 4
                total_embedding_size += embedding_size
                print(f"  Estimated Embedding Size: {format_bytes(embedding_size)}")

                # Average text length
                if 'documents' in sample and sample['documents']:
                    total_text_len = sum(len(doc) for doc in sample['documents'])
                    avg_text_len = total_text_len / len(sample['documents'])
                    est_total_text = count * avg_text_len
                    print(f"  Average Chunk Size: {avg_text_len:.0f} characters")
                    print(f"  Estimated Total Text Size: {format_bytes(est_total_text)}")

                # Metadata info
                if 'metadatas' in sample and sample['metadatas']:
                    print(f"  Metadata Fields: {list(sample['metadatas'][0].keys())}")

            except Exception as e:
                print(f"  Warning: Could not analyze collection: {e}")

        # Collection metadata
        if coll.metadata:
            print(f"  Collection Metadata: {coll.metadata}")

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total Collections: {len(collections)}")
    print(f"  Total Items: {total_items:,}")
    print(f"  Estimated Embedding Storage: {format_bytes(total_embedding_size)}")
    print(f"  Actual Disk Usage: {format_bytes(total_size)}")
    if total_embedding_size > 0:
        overhead = total_size - total_embedding_size
        overhead_pct = (overhead / total_size) * 100
        print(f"  Storage Overhead: {format_bytes(overhead)} ({overhead_pct:.1f}%)")
    print()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
