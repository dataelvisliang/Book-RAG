"""
Delete all collections from ChromaDB
This is safer than deleting the directory
"""

import chromadb
from chromadb.config import Settings

def delete_all_collections(db_path="./chroma_db"):
    """Delete all collections from ChromaDB."""

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    collections = client.list_collections()

    if not collections:
        print("No collections found - database is already empty")
        return

    print(f"\nFound {len(collections)} collection(s):")
    for coll in collections:
        print(f"  - {coll.name} ({coll.count()} items)")

    print("\nDeleting all collections...")
    for coll in collections:
        try:
            client.delete_collection(coll.name)
            print(f"[OK] Deleted: {coll.name}")
        except Exception as e:
            print(f"[FAIL] Failed to delete {coll.name}: {e}")

    # Verify
    remaining = client.list_collections()
    if not remaining:
        print("\n" + "="*60)
        print("All collections deleted successfully!")
        print("Database is now empty and ready for new embeddings.")
        print("="*60)
    else:
        print(f"\nWarning: {len(remaining)} collection(s) still remain")


if __name__ == "__main__":
    delete_all_collections()
