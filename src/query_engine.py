import faiss
import duckdb
import numpy as np
import json
import flatbuffers
import sys
import os
import lmdb  # New dependency
from scipy import sparse

sys.path.append('./generated')
import crs.Concept as C


class CRS:
    def __init__(self, root="data"):
        self.root = root

        # 1. Open LMDB (The Packed DB)
        # map_size=0 means use existing size. readonly=True for speed.
        self.env = None
        db_path = f"{root}/storage"
        if os.path.exists(f"{db_path}/data.mdb"):
            try:
                self.env = lmdb.open(db_path, readonly=True, lock=False)
            except Exception as e:
                print(f"⚠️ Warning: Could not open LMDB: {e}")

        # 2. Load FAISS (Vectors)
        self.index = faiss.read_index(f"{root}/vectors/text.faiss")
        with open(f"{root}/metadata/faiss_id_map.json") as f:
            self.faiss_map = {int(k): v for k, v in json.load(f).items()}

        # 3. Load DuckDB (Properties)
        self.db = duckdb.connect(f"{root}/properties/properties.duckdb", read_only=True)

        # 4. Load Graph (Relations)
        loader = np.load(f"{root}/graph/csr_arrays.npz")
        self.graph = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']))
        with open(f"{root}/graph/node_map.json") as f:
            self.node_map = json.load(f)
            self.rev_node_map = {v: k for k, v in self.node_map.items()}

    def get_concept(self, cid):
        """Reads Flatbuffer from LMDB (Fast) or Disk (Fallback)"""
        # Strategy A: Check LMDB
        if self.env:
            with self.env.begin() as txn:
                # LMDB keys are bytes, so encode the string ID
                buf = txn.get(cid.encode('ascii'))
                if buf:
                    # Zero-copy read directly from memory
                    return C.Concept.GetRootAsConcept(buf, 0)

        # Strategy B: Check Loose File (Fallback for new/unpacked items)
        path = f"{self.root}/concepts/{cid}.bin"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                buf = f.read()
                return C.Concept.GetRootAsConcept(buf, 0)

        return None

    def search_vector(self, embedding, k=5):
        vec = np.array([embedding]).astype('float32')
        D, I = self.index.search(vec, k)
        results = []
        for idx in I[0]:
            if idx != -1:
                cid = self.faiss_map[idx]
                results.append(cid)
        return results

    def filter_properties(self, key, val):
        res = self.db.execute("SELECT id FROM props WHERE key=? AND val_str=?", [key, val]).fetchall()
        return [r[0] for r in res]

    def get_relations(self, cid):
        if cid not in self.node_map: return []
        idx = self.node_map[cid]
        row = self.graph[idx]
        targets = row.indices
        return [self.rev_node_map[t] for t in targets]