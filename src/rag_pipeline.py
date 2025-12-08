#!/usr/bin/env python3
import argparse
import chromadb
from chromadb.config import Settings
import requests
import json
from pathlib import Path

CHROMA_PATH = "embeddings/chroma"
COLLECTION_NAME = "tf_docs"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama"

# ---------------------------------------
# Ollama communication functions
# ---------------------------------------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()

    data = resp.json()
    return data.get("response", "")

# ---------------------------------------
# RAG - fetch documents from ChromaDB
# ---------------------------------------
def get_relevant_docs(query, n_results=4):
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return list(zip(docs, metas))

# ---------------------------------------
# Build prompt with context
# ---------------------------------------
def build_prompt(user_query: str, context_docs: list) -> str:
    context_str = ""

    for doc, meta in context_docs:
        context_str += f"\n# Source: {meta.get('source')}\n{doc}\n"

    final_prompt = f"""
You are an expert in Terraform and clean code for IaC.

Answer the user's question using the provided context.
If something is missing from the context, state it clearly, but try to suggest a best practice.

### CONTEXT
{context_str}

### QUESTION
{user_query}

### ANSWER
"""
    return final_prompt

# ---------------------------------------
# Interactive mode
# ---------------------------------------
def interactive_chat():
    print("Interactive RAG chat. Type 'exit' to quit.\n")

    while True:
        query = input("> ")

        if query.strip().lower() in ["exit", "quit"]:
            break

        docs = get_relevant_docs(query)
        prompt = build_prompt(query, docs)
        answer = call_ollama(prompt)

        print("\n--- Answer ---\n")
        print(answer)
        print("\n-----------------\n")

# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG pipeline for Terraform + documentation")
    parser.add_argument("--query", type=str, help="One-time query to the model")
    parser.add_argument("--n", type=int, default=4, help="Number of returned documents")
    args = parser.parse_args()

    if args.query:
        docs = get_relevant_docs(args.query, n_results=args.n)
        prompt = build_prompt(args.query, docs)
        answer = call_ollama(prompt)
        print(answer)
    else:
        interactive_chat()
