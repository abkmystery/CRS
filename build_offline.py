import os
import time
import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# Import our modules
from src.builder import ConceptBuilder
from src.embedders import MultimodalEmbedder
from src.indexer import CRSIndexer

# Disable Symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuration for Intel Iris Xe (CPU)
BATCH_SIZE = 32


def process_batch_offline(batch_synsets, embedder, builder):
    """
    Processes a batch strictly using local CPU and local Disk.
    No Network calls.
    """
    batch_data = []
    texts_to_embed = []

    # 1. Prepare Metadata (Instant - from local NLTK DB)
    for syn in batch_synsets:
        # FIX: Use underscore instead of colon for Windows filename safety
        cid = f"wn_{syn.offset()}{syn.pos()}"
        label = syn.lemmas()[0].name().replace('_', ' ')
        definition = syn.definition()

        # Extract Relations (Local Graph Traversal)
        relations = []

        # Hypernyms (Parents)
        for hyper in syn.hypernyms():
            relations.append({
                'type': 'is_a',
                # FIX: Ensure target IDs also use underscore
                'target_id': f"wn_{hyper.offset()}{hyper.pos()}",
                'source': 'wordnet'
            })

        # Hyponyms (Children)
        for hypo in syn.hyponyms():
            relations.append({
                'type': 'parent_of',
                'target_id': f"wn_{hypo.offset()}{hypo.pos()}",
                'source': 'wordnet'
            })

        # Holonyms (Part of)
        for holo in syn.part_holonyms():
            relations.append({
                'type': 'part_of',
                'target_id': f"wn_{holo.offset()}{holo.pos()}",
                'source': 'wordnet'
            })

        # Local Properties
        props = [
            {'key': 'pos', 'value': syn.pos()},
            {'key': 'lex_domain', 'value': syn.lexname()},
            {'key': 'lemmas', 'value': ",".join([l.name() for l in syn.lemmas()])}
        ]

        item = {
            'id': cid,
            'label': label,
            'definition': definition,
            'relations': relations,
            'properties': props
        }

        batch_data.append(item)
        # Context for embedding: "Label: Definition"
        texts_to_embed.append(f"{label}: {definition}")

    # 2. Batch Embedding (CPU Vectorized)
    # This uses your Intel CPU's math co-processor efficiently
    embeddings = embedder.embed_text_batch(texts_to_embed)

    # 3. Write to Flatbuffers
    meta_results = []
    for i, item in enumerate(batch_data):
        item['text_embedding'] = embeddings[i]
        builder.build_concept(item)
        meta_results.append(item)

    return meta_results


def build_offline_crs(limit=None):
    print("--- ⚡ Offline CRS Builder (Intel Optimized) ---")

    # Initialize
    builder = ConceptBuilder()
    embedder = MultimodalEmbedder()  # Auto-detects CPU/GPU
    indexer = CRSIndexer()

    # Load Data from Disk
    print("Loading WordNet Database...")
    all_synsets = list(wn.all_synsets())
    total_count = len(all_synsets)

    if limit:
        all_synsets = all_synsets[:limit]
        total_count = limit

    print(f"Loaded {total_count} concepts.")
    print(f"Processing in batches of {BATCH_SIZE} on CPU...")

    meta_for_indexing = []

    # Main Loop
    start_time = time.time()

    for i in tqdm(range(0, total_count, BATCH_SIZE)):
        batch = all_synsets[i: i + BATCH_SIZE]
        results = process_batch_offline(batch, embedder, builder)
        meta_for_indexing.extend(results)

    duration = time.time() - start_time
    print(f"Processed {total_count} items in {duration:.2f}s")

    # Indexing
    print("Building Search Indexes...")
    indexer.build_indexes(meta_for_indexing)
    print("✅ Offline Build Complete.")


if __name__ == "__main__":
    # None = Run ALL 117,000 items
    build_offline_crs(limit=None)