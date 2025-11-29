import lmdb
import glob
import os
import struct
from tqdm import tqdm


def pack_to_lmdb():
    print("--- 📦 Packing CRS to LMDB ---")

    # 1. Setup LMDB (Map size = 1GB is enough for WordNet, using 2GB to be safe)
    os.makedirs("data/storage", exist_ok=True)
    env = lmdb.open("data/storage", map_size=2 * 1024 * 1024 * 1024)

    # 2. Get all files
    files = glob.glob("data/concepts/*.bin")
    print(f"Found {len(files)} files to pack.")

    # 3. Write to DB
    with env.begin(write=True) as txn:
        for filepath in tqdm(files):
            # Extract ID from filename (e.g., "data/concepts\wn_12345n.bin" -> "wn_12345n")
            filename = os.path.basename(filepath)
            cid = os.path.splitext(filename)[0]

            with open(filepath, 'rb') as f:
                data = f.read()

            # Store: Key=ID, Value=BinaryData
            txn.put(cid.encode('ascii'), data)

    print("✅ Packing Complete. You can now delete 'data/concepts/' folder.")


if __name__ == "__main__":
    pack_to_lmdb()