# Personal AGI Kernel (CRS Core – Stage 1 & 2)

A local-first **Neuro-Symbolic Concept Representation Store (CRS)** combining:

- **Vector Search (FAISS)**
- **Knowledge Graphs (CSR)**
- **Zero-Copy Binary Storage (Flatbuffers + LMDB)**
- **Self-Expanding Knowledge (Stage 2 Agent)**

CRS is a unified concept memory substrate that supports:
- 117k+ WordNet concepts (Stage 1)
- Automatic learning from Wikipedia / Web / Wikidata (Stage 2)
- Ultra-fast retrieval using FAISS + LMDB
- Structured relations and properties using CSR + DuckDB

---

## 🚀 Features
- **Hybrid Storage:**  
  WordNet concepts pre-built using Flatbuffers → packed into LMDB for instant loading.

- **Self-Learning Memory:**  
  The Agent identifies unknown topics, fetches live knowledge, and expands CRS.

- **Neuro-Symbolic Reasoning Substrate:**  
  Uses vector search for fuzziness + symbolic indexes for precision.

---

## 📦 Installation

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
````

### 2. Flatbuffers code
All Flatbuffer Python bindings are pre-generated and included in `generated/`.
You do NOT need the flatc compiler.


---

### 3. Building the CRS Dataset (Required Before Running Agent)

> **Important:**
> CRS does not ship with pre-built data.
> You must generate the dataset once on your machine (≈1.5–2 GB).

Run the offline builder:

```bash
python build_offline.py
```

This will:

* Ingest **WordNet** (≈117k concepts)
* Generate **Flatbuffer concept objects**
* Build the **FAISS vector index**
* Build the **DuckDB property store**
* Build the **CSR relation graph**

Then **pack everything into LMDB**:

```bash
python pack_crs.py
```

After this step, your full CRS memory is ready at:

```
data/storage/
data/vectors/
data/graph/
data/properties/
```

---

### 4. Running the Self-Learning Agent

Once the data is built:

```bash
python -m src.agent
```

The agent supports:

* **Learning new concepts** from Web / Wikipedia / Wikidata
* **Adding evidence to concepts**
* **Updating embeddings + FAISS index**
* **Rebuilding the graph periodically**
* **Packing memory as it grows**

### Example:

Ask:

```
Who is Sam Altman?
```

→ Agent will fetch from Wikipedia/Web and store it.

Ask again:

```
Sam Altman
```

→ Retrieved instantly from your CRS memory.

---

## 📂 Repository Structure

```
src/
  agent.py            # Stage 2 self-learning agent
  builder.py          # Flatbuffer concept builder
  embedders.py        # Text + image embedding
  indexer.py          # FAISS, DuckDB, CSR generation
  query_engine.py     # CRS runtime reader
  wikidata.py         # Wikidata API integration

schema/
  concept.fbs         # Flatbuffer schema

generated/
  crs/*.py            # Auto-generated Flatbuffer code (run 'flatc' first)

build_offline.py      # Build entire CRS dataset (Stage 1)
pack_crs.py           # Pack Flatbuffers → LMDB
```

---

## 📌 Notes

* Multimodal (image/audio/video) learning is planned for future stages.
* Conflict resolution & property typing deferred until Stage 3 (LLM-assisted reasoning).
* For now, CRS stores raw values to maintain robustness across noisy Web data.

---

## 📄 License

Apache 2.0 

---

