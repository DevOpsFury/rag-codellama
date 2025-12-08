#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# Constants
CHROMA_PATH = "embeddings/chroma"
COLLECTION_NAME = "tf_docs"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama"

# Load embedding model and ChromaDB database
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
# ---------------------------------------------------------
# Function: generate embedding and search for context
# ---------------------------------------------------------
def search_context(query: str, top_k: int = 5) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    if not results["documents"]:
        return ""

    documents = results["documents"][0]
    return "\n\n".join(documents)
    )

    if not results["documents"]:
        return ""

    documents = results["documents"][0]
    return "\n\n".join(documents)


# -----------------------------------------
# 3. Function: query LLM model with context
# -----------------------------------------
# Function: query LLM model with context via Ollama
# -----------------------------------------
def ask_model(question: str, context: str) -> str:
    prompt = f"""
You are an expert in Terraform and clean code for IaC.

Answer the user's question using the provided context.
If something is missing from the context, state it clearly, but try to suggest a best practice.

### CONTEXT
{context}

### QUESTION
{question}

Respond precisely and technically.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()

# -------------------------------------------------------
# Usage example — Terraform analysis and refactoring
# -------------------------------------------------------
if __name__ == "__main__":

    tf_snippet = """
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  name   = "example"
  cidr   = "10.0.0.0/16"

  azs             = ["eu-central-1a", "eu-central-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway = true
}
"""

    question = (
        "Should this VPC module support public subnets "
        "and how to add it according to our standard?"
    )

    context = search_context(question + "\n" + tf_snippet)
    response = ask_model(question + "\n\nCode:\n" + tf_snippet, context)

    print("\n=== MODEL RESPONSE ===\n")
    print(response)
    print("\n=== MODEL RESPONSE ===\n")
    print(response)
