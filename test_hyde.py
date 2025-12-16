"""
Test HyDE (Hypothetical Document Embeddings) vs Traditional Retrieval
Compare retrieval quality with and without HyDE
"""

import os
from rag_backend import RAGBackend
import argparse


def test_hyde_comparison(query, api_key, top_k=5):
    """
    Compare retrieval results with and without HyDE.

    Args:
        query: Test query
        api_key: OpenRouter API key
        top_k: Number of results to retrieve
    """
    print("="*80)
    print(f"Testing HyDE with query: {query}")
    print("="*80)

    # Initialize backend
    backend = RAGBackend(persist_directory="./chroma_db")

    # Get all collections
    collections = backend.get_all_collections()
    if not collections:
        print("\nNo collections found. Please run preprocessing first.")
        return

    collection_names = [c.name for c in collections]
    print(f"\nSearching in {len(collection_names)} collection(s)")

    # Test WITHOUT HyDE
    print("\n" + "-"*80)
    print("TRADITIONAL RETRIEVAL (Direct Query Embedding)")
    print("-"*80)

    results_no_hyde = backend.retrieve_relevant_chunks(
        query=query,
        collection_names=collection_names,
        top_k=top_k,
        use_hyde=False
    )

    for i, result in enumerate(results_no_hyde, 1):
        print(f"\n[Result {i}] Distance: {result['distance']:.4f} | Page: {result['page_number']}")
        print(f"Text: {result['text'][:300]}...")
        print()

    # Test WITH HyDE
    print("\n" + "-"*80)
    print("HyDE RETRIEVAL (Hypothetical Document Embedding)")
    print("-"*80)

    results_hyde = backend.retrieve_relevant_chunks(
        query=query,
        collection_names=collection_names,
        top_k=top_k,
        use_hyde=True,
        api_key=api_key,
        hyde_model="nvidia/nemotron-3-nano-30b-a3b:free"
    )

    for i, result in enumerate(results_hyde, 1):
        print(f"\n[Result {i}] Distance: {result['distance']:.4f} | Page: {result['page_number']}")
        print(f"Text: {result['text'][:300]}...")
        print()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    avg_dist_no_hyde = sum(r['distance'] for r in results_no_hyde) / len(results_no_hyde)
    avg_dist_hyde = sum(r['distance'] for r in results_hyde) / len(results_hyde)

    print(f"\nAverage Distance (Lower is Better):")
    print(f"  Traditional: {avg_dist_no_hyde:.4f}")
    print(f"  HyDE:        {avg_dist_hyde:.4f}")

    if avg_dist_hyde < avg_dist_no_hyde:
        improvement = ((avg_dist_no_hyde - avg_dist_hyde) / avg_dist_no_hyde) * 100
        print(f"\n✓ HyDE improved retrieval by {improvement:.1f}%")
    else:
        print(f"\n  Traditional retrieval performed better for this query")

    # Check overlap
    hyde_pages = set(r['page_number'] for r in results_hyde)
    no_hyde_pages = set(r['page_number'] for r in results_no_hyde)
    overlap = len(hyde_pages & no_hyde_pages)

    print(f"\nPage Overlap: {overlap}/{top_k} results from same pages")
    if overlap < top_k:
        print(f"  → HyDE found {top_k - overlap} different pages")


def main():
    parser = argparse.ArgumentParser(description='Test HyDE vs Traditional Retrieval')
    parser.add_argument('--query', type=str, help='Test query')
    parser.add_argument('--api-key', type=str, help='OpenRouter API key')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OpenRouter API key required")
        print("Provide via --api-key argument or OPENROUTER_API_KEY environment variable")
        return

    # Default test queries if none provided
    if args.query:
        queries = [args.query]
    else:
        print("No query provided. Testing with sample queries...\n")
        queries = [
            "What is overfitting in machine learning?",
            "How does gradient descent work?",
            "What are the differences between supervised and unsupervised learning?",
        ]

    # Run tests
    for query in queries:
        test_hyde_comparison(query, api_key, args.top_k)
        if len(queries) > 1:
            print("\n\n")


if __name__ == "__main__":
    main()
