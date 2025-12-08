#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------------
# Model and ChromaDB configuration
# -------------------------------
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
STATE_FILE = "embeddings/state.json"
COLLECTION_NAME = "tf_docs"
DATA_PATH = "data/"

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="embeddings/chroma")
collection = client.get_or_create_collection(COLLECTION_NAME)

# -------------------------------
# Helper functions
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_files(path, extensions={'.tf', '.tfvars', '.md'}):
    docs = []
    for p in Path(path).rglob("*"):
        if p.suffix in extensions:
            try:
                docs.append((str(p), p.read_text()))
            except Exception as e:
                print(f"Cannot read file {p}: {e}")
    return docs

def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def load_state():
    if Path(STATE_FILE).exists():
        return json.loads(Path(STATE_FILE).read_text())
    return {}

def save_state(state):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))

# -------------------------------
# Update and rebuild handling
# -------------------------------
def get_changed_files():
    state = load_state()
    changed = []
    removed = []
    current_files = {}

    for path, content in load_files(DATA_PATH):
        h = file_hash(content)
        current_files[path] = h
        if path not in state or state[path] != h:
            changed.append((path, content))

    for old_path in state:
        if old_path not in current_files:
            removed.append(old_path)

    return changed, removed, current_files

def update_index():
    changed, removed, current_files = get_changed_files()
    print(f"Found {len(changed)} changed/new files")
    print(f"Found {len(removed)} removed files")

    for r in removed:
        print(f"Removing from index: {r}")
        # find ids for this source and delete by ids (safe for Chroma)
        try:
            res = collection.get(where={"source": {"$eq": r}})
            ids = res.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        except Exception as e:
            print(f"Error deleting {r}: {e}")

    for path, content in changed:
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": path}],
                ids=[f"{path}-{i}"]
            )

    save_state(current_files)
    print("Index update completed.")

def ingest_documents():
    print("Creating full index...")
    # delete all documents safely by collecting all ids first
    try:
        all_items = collection.get()
        all_ids = all_items.get("ids", [])
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception as e:
        print(f"Error clearing collection: {e}")
    for path, content in load_files(DATA_PATH):
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": path}],
                ids=[f"{path}-{i}"]
            )
    state = {path: file_hash(content) for path, content in load_files(DATA_PATH)}
    save_state(state)
    print("Index rebuilt.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Update only changed or new files")
    parser.add_argument("--rebuild", action="store_true", help="Clear the index and rebuild it")
    args = parser.parse_args()

    if args.rebuild:
        ingest_documents()
    elif args.update:
        update_index()
    else:
        ingest_documents()
