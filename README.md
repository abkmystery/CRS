# Personal AGI Kernel (Stage 1 & 2)

A Neuro-Symbolic Concept Representation Store (CRS) that runs locally.
It combines Vector Search (FAISS), Knowledge Graphs (CSR), and Zero-Copy Storage (Flatbuffers/LMDB).

## Features
- **Hybrid Storage:** 117k Concepts pre-packed in LMDB + Dynamic expansion via Flatbuffers.
- **Self-Learning:** The Agent detects unknown concepts, fetches them from Wikipedia/Web, and integrates them into the graph.
- **Neuro-Symbolic:** Uses Vectors for fuzzy search and Symbolic Index for exact recall.

## Installation
1. Install dependencies:
   `pip install -r requirements.txt`
2. Compile Schema:
   `flatc --python -o generated schema/concept.fbs`

## Usage
**Run the Learning Agent:**
`python -m src.agent`

- Ask: "Who is Sam Altman?" (It will learn from the web)
- Ask: "Sam Altman" (It will recall from memory)