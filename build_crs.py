import sys
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import time

# Import our modules
from src.builder import ConceptBuilder
from src.embedders import MultimodalEmbedder
from src.indexer import CRSIndexer
from src.wikidata import WikidataFetcher

def generate_seed_crs(limit=100):
    print("--- Stage 1 CRS Builder (WordNet + Wikidata) ---")
    
    # 1. Initialize Components
    builder = ConceptBuilder()
    embedder = MultimodalEmbedder()
    indexer = CRSIndexer()
    wiki = WikidataFetcher()
    
    meta_for_indexing = []
    
    # Get Synsets
    synsets = list(wn.all_synsets())[:limit]
    print(f"Processing {len(synsets)} concepts...")
    
    for syn in tqdm(synsets):
        # --- A. Basic Data (WordNet) ---
        label = syn.lemmas()[0].name().replace('_', ' ')
        cid = f"wn:{syn.offset()}{syn.pos()}" # Stable ID
        definition = syn.definition()
        
        # --- B. Relations (WordNet) ---
        relations = []
        for hyper in syn.hypernyms():
            relations.append({
                'type': 'is_a', 
                'target_id': f"wn:{hyper.offset()}{hyper.pos()}", 
                'source': 'wordnet'
            })
            
        # --- C. Enrichment (Wikidata) ---
        # Note: Real ingestion pipelines run async or batch this.
        # We do it synchronously here for the demo logic.
        properties = [{'key': 'pos', 'value': syn.pos()}]
        
        # 1. Find Wikidata ID
        qid = wiki.search_entity(label)
        if qid:
            # 2. Fetch Deep Properties
            w_props, w_rels = wiki.get_properties(qid)
            
            # Merge
            properties.extend(w_props)
            relations.extend(w_rels)
            
            # Add QID as a property alias
            properties.append({'key': 'wikidata_id', 'value': qid})
            
            # Be nice to the API
            time.sleep(0.1) 

        # --- D. Embeddings ---
        text_context = f"{label}: {definition}"
        text_emb = embedder.embed_text(text_context)
        
        # --- E. Build Artifact ---
        concept_data = {
            'id': cid,
            'label': label,
            'definition': definition,
            'text_embedding': text_emb,
            'relations': relations,
            'properties': properties
        }
        
        builder.build_concept(concept_data)
        meta_for_indexing.append(concept_data)
        
    # --- F. Indexing ---
    print("Building Indexes...")
    indexer.build_indexes(meta_for_indexing)
    print("CRS Build Complete.")

if __name__ == "__main__":
    # Lower limit for testing because API calls take time
    generate_seed_crs(limit=50)