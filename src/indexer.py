import faiss
import duckdb
import numpy as np
import os
import json
from scipy import sparse


class CRSIndexer:
    def __init__(self, data_root="data"):
        self.root = data_root
        os.makedirs(f"{self.root}/vectors", exist_ok=True)
        os.makedirs(f"{self.root}/properties", exist_ok=True)
        os.makedirs(f"{self.root}/graph", exist_ok=True)
        os.makedirs(f"{self.root}/metadata", exist_ok=True)

    def build_indexes(self, concepts_metadata):
        """
        concepts_metadata: List of dicts {id, embedding, relations, properties}
        """
        print("Building FAISS Index...")
        self._build_faiss(concepts_metadata)

        print("Building DuckDB Store...")
        self._build_duckdb(concepts_metadata)

        print("Building CSR Graph...")
        self._build_csr(concepts_metadata)

    # def _build_faiss(self, data):
    #     # Extract embeddings
    #     ids = []
    #     vecs = []
    #     for i, item in enumerate(data):
    #         if item.get('text_embedding'):
    #             ids.append(i)  # Use integer ID mapping
    #             vecs.append(item['text_embedding'])
    #
    #     if not vecs: return
    #
    #     d = len(vecs[0])
    #     vecs_np = np.array(vecs).astype('float32')
    #
    #     # HNSW for speed (Commercial Safe MIT)
    #     index = faiss.IndexHNSWFlat(d, 32)
    #     index.add(vecs_np)
    #
    #     faiss.write_index(index, f"{self.root}/vectors/text.faiss")
    #
    #     # Save ID mapping
    #     mapping = {i: item['id'] for i, item in zip(ids, data)}
    #     with open(f"{self.root}/metadata/faiss_id_map.json", 'w') as f:
    #         json.dump(mapping, f)

    def _build_faiss(self, data):
        # 1. Filter only items that HAVE embeddings
        valid_items = [item for item in data if item.get('text_embedding')]

        if not valid_items:
            return

        # 2. Prepare data for FAISS
        vecs = [item['text_embedding'] for item in valid_items]
        d = len(vecs[0])
        vecs_np = np.array(vecs).astype('float32')

        # 3. Build Index
        index = faiss.IndexHNSWFlat(d, 32)
        index.add(vecs_np)

        faiss.write_index(index, f"{self.root}/vectors/text.faiss")

        # 4. Save ID mapping (FAISS sequential ID -> Concept ID)
        # FAISS assigns IDs 0, 1, 2... automatically. We map 0 to the first valid item, 1 to the second, etc.
        mapping = {i: item['id'] for i, item in enumerate(valid_items)}

        with open(f"{self.root}/metadata/faiss_id_map.json", 'w') as f:
            json.dump(mapping, f)



    def _build_duckdb(self, data):
        con = duckdb.connect(f"{self.root}/properties/properties.duckdb")
        con.execute("CREATE TABLE IF NOT EXISTS props (id VARCHAR, key VARCHAR, val_str VARCHAR, val_num DOUBLE)")

        # Batch insert
        rows = []
        for item in data:
            cid = item['id']
            for p in item.get('properties', []):
                val = p['value']
                # Try parsing float
                try:
                    val_num = float(val)
                except:
                    val_num = None
                rows.append((cid, p['key'], str(val), val_num))

        con.executemany("INSERT INTO props VALUES (?, ?, ?, ?)", rows)
        con.close()

    def _build_csr(self, data):
        # Create integer ID map
        id_to_int = {item['id']: i for i, item in enumerate(data)}

        row_ind = []
        col_ind = []
        data_val = []

        for item in data:
            u = id_to_int[item['id']]
            for r in item.get('relations', []):
                target = r['target_id']
                if target in id_to_int:
                    v = id_to_int[target]
                    row_ind.append(u)
                    col_ind.append(v)
                    data_val.append(1)  # Unweighted for now, or use r['confidence']

        # Build Matrix
        size = len(data)
        adj_matrix = sparse.csr_matrix((data_val, (row_ind, col_ind)), shape=(size, size))

        # Save buffers
        np.savez(f"{self.root}/graph/csr_arrays.npz",
                 indptr=adj_matrix.indptr,
                 indices=adj_matrix.indices,
                 data=adj_matrix.data)

        with open(f"{self.root}/graph/node_map.json", 'w') as f:
            json.dump(id_to_int, f)