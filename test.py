import time
from src.query_engine import CRS
from src.embedders import MultimodalEmbedder


def run_demo():
    print("--- 🧠 Loading CRS (Stage 1) ---")
    start_load = time.time()

    # 1. Load the "Library"
    crs = CRS(root="data")

    # 2. Load the "Translator" (Embedder)
    # This must match the model you used to build it (MiniLM)
    embedder = MultimodalEmbedder()

    print(f"System loaded in {time.time() - start_load:.2f}s")
    print("-" * 50)

    while True:
        query = input("\n🔎 Enter a concept to search (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        # --- A. Vector Search (The "Vibe" Check) ---
        print(f"   Computing vector for '{query}'...")
        query_vec = embedder.embed_text(query)

        # Search FAISS for nearest neighbors
        start_search = time.time()
        results_ids = crs.search_vector(query_vec, k=3)  # Get top 3
        duration = time.time() - start_search

        print(f"   Found {len(results_ids)} results in {duration:.4f}s:")

        # --- B. Retrieve & Display (Zero-Copy) ---
        for i, cid in enumerate(results_ids):
            # Read directly from Flatbuffer file
            concept = crs.get_concept(cid)

            if concept:
                label = concept.Label().decode('utf-8')
                definition = concept.Definition().decode('utf-8')
                print(f"   {i + 1}. [{label}] -> {definition[:100]}...")

                # --- C. Graph Traversal (The "Logic" Check) ---
                # Check what this concept is connected to
                related_ids = crs.get_relations(cid)
                if related_ids:
                    # Just fetch the first 2 relations to show it works
                    print(f"      🔗 Connected to {len(related_ids)} other concepts.")
                    # Let's peek at the first parent/child
                    first_rel_id = related_ids[0]
                    rel_concept = crs.get_concept(first_rel_id)
                    if rel_concept:
                        rel_label = rel_concept.Label().decode('utf-8')
                        print(f"      🔗 Example connection: [{rel_label}]")


if __name__ == "__main__":
    run_demo()