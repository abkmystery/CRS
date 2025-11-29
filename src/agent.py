import time
import os
import json
import numpy as np
import faiss
import wikipedia
import re
import trafilatura
import glob
import lmdb
from duckduckgo_search import DDGS
from scipy import sparse
from src.query_engine import CRS
from src.embedders import MultimodalEmbedder
from src.wikidata import WikidataFetcher
from src.builder import ConceptBuilder


class LearningAgent:
    def __init__(self, root="data"):
        print("🤖 Agent waking up... (Loading Memory)")
        self.root = root
        self.crs = CRS(root)
        self.embedder = MultimodalEmbedder()
        self.wiki = WikidataFetcher()
        self.builder = ConceptBuilder(output_dir=f"{root}/concepts")

        self.UNKNOWN_THRESHOLD = 0.85
        self.items_learned_session = 0
        self.MAINTENANCE_TRIGGER = 5
        self.label_index_path = f"{root}/metadata/label_index.json"
        self.label_index = self._load_label_index()

    def _load_label_index(self):
        if os.path.exists(self.label_index_path):
            with open(self.label_index_path, 'r') as f: return json.load(f)
        return {}

    def _save_label_index(self):
        with open(self.label_index_path, 'w') as f: json.dump(self.label_index, f)

    def extract_subject(self, sentence):
        patterns = [
            r"^what's happening with\s+", r"^what is\s+", r"^who is\s+",
            r"^tell me about\s+", r"^latest news on\s+"
        ]
        clean_q = sentence
        for p in patterns: clean_q = re.sub(p, "", clean_q, flags=re.IGNORECASE)
        return clean_q.replace('?', '').replace('!', '').strip()

    def get_web_data(self, query):
        try:
            results = DDGS().text(query, max_results=1)
            if not results: return None, None
            top_hit = results[0]
            url = top_hit['href']
            print(f"   🔗 Crawling URL: {url}")
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text: return text[:600] + "...", url
            return top_hit['body'], url
        except Exception as e:
            return None, None

    def ask(self, user_query):
        is_news = any(w in user_query.lower() for w in ['news', 'latest', 'today', 'price'])
        search_term = self.extract_subject(user_query)
        clean_key = search_term.lower().strip()
        print(f"\n🤔 Query: '{user_query}' -> Subject: '{search_term}'")

        # 1. Symbolic Search
        if not is_news and clean_key in self.label_index:
            cid = self.label_index[clean_key]
            print(f"   📖 Found in Symbolic Index (ID: {cid}).")
            concept = self.crs.get_concept(cid)
            if concept: return self.format_concept(concept)

        # 2. Vector Search
        query_vec = self.embedder.embed_text(search_term)
        if not query_vec: return "Error."
        D, I = self.crs.index.search(np.array([query_vec]).astype('float32'), k=1)

        best_idx = I[0][0]
        distance = D[0][0]
        print(f"   Internal Memory Distance: {distance:.4f}")

        # 3. Decision
        if is_news or best_idx == -1 or distance > self.UNKNOWN_THRESHOLD:
            print(f"   🌑 Unknown or News requested.")
            new_cid = self.learn_concept(search_term, force_web=is_news)
            if new_cid:
                self.check_maintenance()
                return self.format_concept(self.crs.get_concept(new_cid))
            return "❌ Could not find info."

        return self.format_concept(self.crs.get_concept(self.crs.faiss_map[best_idx]))

    def learn_concept(self, query, force_web=False):
        qid = self.wiki.search_entity(query)
        props, rels, aliases = [], [], []

        if qid:
            print(f"   🔗 Linking to Wikidata ID: {qid}")
            _, _, wiki_props, wiki_rels = self.wiki.get_details(qid)
            props.extend(wiki_props)
            rels.extend(wiki_rels)
            aliases.append(query)

        evidence = []
        final_text = ""

        if not force_web:
            try:
                final_text = wikipedia.summary(query, sentences=3)
                evidence.append({'source_type': 'wikipedia', 'url': 'wikipedia.org', 'snippet': final_text[:100]})
            except:
                pass

        if not final_text:
            web_text, url = self.get_web_data(query)
            if web_text:
                final_text = web_text
                evidence.append({'source_type': 'web', 'url': url, 'snippet': web_text[:100]})

        if not final_text and qid:
            wiki_desc, _, _, _ = self.wiki.get_details(qid)
            final_text = wiki_desc
            evidence.append({'source_type': 'wikidata', 'url': f"wikidata.org/wiki/{qid}", 'snippet': 'Description'})

        if not final_text: return None

        concept_id = f"wiki_{qid}" if qid else f"web_{hash(query)}"
        text_emb = self.embedder.embed_text(f"{query}: {final_text}")

        concept_data = {
            'id': concept_id,
            'label': query,
            'aliases': aliases,
            'definition': final_text,
            'text_embedding': text_emb,
            'relations': rels,
            'properties': props,
            'evidence': evidence
        }

        self.builder.build_concept(concept_data)

        self.crs.index.add(np.array([text_emb]).astype('float32'))
        self.crs.faiss_map[self.crs.index.ntotal - 1] = concept_data['id']

        self.label_index[query.lower()] = concept_id
        for alias in aliases: self.label_index[alias.lower()] = concept_id
        self._save_label_index()

        faiss.write_index(self.crs.index, f"{self.root}/vectors/text.faiss")
        with open(f"{self.root}/metadata/faiss_id_map.json", 'w') as f:
            json.dump(self.crs.faiss_map, f)

        return concept_data['id']

    def format_concept(self, concept):
        label = concept.Label().decode('utf-8')
        definition = concept.Definition().decode('utf-8')
        ev_str = ""
        e_len = concept.EvidenceLength()
        if e_len > 0:
            e = concept.Evidence(0)
            ev_str = f"\n   [Source]: {e.Url().decode('utf-8')}"
        return f"[{label}]\n   {definition}{ev_str}"

    def check_maintenance(self):
        self.items_learned_session += 1
        if self.items_learned_session >= self.MAINTENANCE_TRIGGER:
            print(f"\n   🛠️ MAINTENANCE: Rebuilding Graph & Packing LMDB...")
            self.rebuild_graph()
            self.pack_memory()
            self.items_learned_session = 0

    def rebuild_graph(self):
        all_ids = list(self.crs.faiss_map.values())
        id_to_int = {cid: i for i, cid in enumerate(all_ids)}
        row_ind, col_ind, data_val = [], [], []

        for cid in all_ids:
            c = self.crs.get_concept(cid)
            if not c: continue
            u = id_to_int[cid]
            rels_len = c.RelationsLength()
            for i in range(rels_len):
                r = c.Relations(i)
                target = r.TargetId().decode('utf-8')
                if target in id_to_int:
                    v = id_to_int[target]
                    row_ind.append(u)
                    col_ind.append(v)
                    data_val.append(1)

        size = len(all_ids)
        adj_matrix = sparse.csr_matrix((data_val, (row_ind, col_ind)), shape=(size, size))
        np.savez(f"{self.root}/graph/csr_arrays.npz", indptr=adj_matrix.indptr, indices=adj_matrix.indices,
                 data=adj_matrix.data)
        with open(f"{self.root}/graph/node_map.json", 'w') as f:
            json.dump(id_to_int, f)
        self.crs.graph = adj_matrix
        self.crs.node_map = id_to_int
        self.crs.rev_node_map = {v: k for k, v in id_to_int.items()}

    def pack_memory(self):
        # Open with 2GB limit, but file will only grow as needed on Linux/Mac.
        # On Windows it pre-allocates, so we might want to compact later.
        env = lmdb.open(f"{self.root}/storage", map_size=2 * 1024 * 1024 * 1024)
        files = glob.glob(f"{self.root}/concepts/*.bin")

        if not files: return

        with env.begin(write=True) as txn:
            for filepath in files:
                cid = os.path.basename(filepath).replace('.bin', '')
                with open(filepath, 'rb') as f: data = f.read()
                txn.put(cid.encode('ascii'), data)

        # Cleanup loose files after packing
        for filepath in files:
            try:
                os.remove(filepath)
            except:
                pass  # File might be locked on Windows

        print("   📦 Memory Packed.")


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    agent = LearningAgent()
    print("Welcome to Stage 2.5 (Evidence + Packing).")
    while True:
        q = input("\nUser> ")
        if q.lower() == 'q': break
        print(f"Agent> {agent.ask(q)}")