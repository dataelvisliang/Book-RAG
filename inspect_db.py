"""
Inspect ChromaDB contents
Utility script to view what's stored in the vector database
"""

import chromadb
import argparse


def main():
    parser = argparse.ArgumentParser(description='Inspect ChromaDB contents')
    parser.add_argument('--db-path', type=str, default='./chroma_db',
                       help='Path to ChromaDB storage (default: ./chroma_db)')
    parser.add_argument('--collection', type=str, default=None,
                       help='Specific collection to inspect (default: show all)')
    parser.add_argument('--sample', type=int, default=3,
                       help='Number of sample items to show (default: 3)')

    args = parser.parse_args()

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=args.db_path)
    collections = client.list_collections()

    print(f"\n{'='*70}")
    print(f"ChromaDB Inspection: {args.db_path}")
    print(f"{'='*70}\n")

    if not collections:
        print("No collections found in database.")
        return

    print(f"Total Collections: {len(collections)}\n")

    for coll in collections:
        # Skip if specific collection requested and this isn't it
        if args.collection and coll.name != args.collection:
            continue

        print(f"\n{'-'*70}")
        print(f"Collection: {coll.name}")
        print(f"{'-'*70}")
        print(f"  Total Items: {coll.count()}")
        print(f"  Metadata: {coll.metadata}")

        # Get sample data
        if coll.count() > 0:
            sample_size = min(args.sample, coll.count())
            sample = coll.peek(sample_size)

            print(f"\n  Sample Items ({sample_size} of {coll.count()}):\n")

            for i in range(len(sample['ids'])):
                print(f"  [{i+1}] ID: {sample['ids'][i]}")
                print(f"      Page: {sample['metadatas'][i].get('page_number', 'N/A')}")
                print(f"      Chunk ID: {sample['metadatas'][i].get('chunk_id', 'N/A')}")

                # Show text preview
                text = sample['documents'][i]
                preview_length = 500
                if len(text) > preview_length:
                    preview = text[:preview_length] + "..."
                else:
                    preview = text

                print(f"      Text: {preview}")

                # Show embedding info
                try:
                    if 'embeddings' in sample and sample['embeddings'] is not None:
                        if i < len(sample['embeddings']) and sample['embeddings'][i] is not None:
                            emb_dim = len(sample['embeddings'][i])
                            print(f"      Embedding: {emb_dim}-dimensional vector")
                except (TypeError, ValueError, IndexError):
                    # Skip embedding info if there's any issue
                    pass

                print()

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
