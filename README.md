
# Source scripts

## connection_test.py
Test the connection to an Ollama LLM model.

**Usage:**
```bash
python connection_test.py
```

---

## ingest.py
Indexes and updates documents in ChromaDB for retrieval-augmented generation (RAG). Use `--rebuild` to clear and rebuild the index, or `--update` to only update changed files.

**Usage:**
```bash
# Rebuild the index from scratch
python ingest.py --rebuild

# Update only changed or new files
python ingest.py --update
```

---

## rag_pipeline.py
Main RAG pipeline: retrieves context from ChromaDB and queries an LLM (Ollama) with user questions. Supports both one-off queries and interactive chat.

**Usage:**
```bash
# One-time query
python rag_pipeline.py --query "What's the purpose of terraform-aws-atlantis module?"

# Interactive chat mode
python rag_pipeline.py
```

---

## query.py
Minimal RAG pipeline using Ollama and ChromaDB, with a sample Terraform code analysis and best-practice suggestion.

**Usage:**
```bash
python quiery.py
```
This script runs a hardcoded example and prints the model's answer. You can update it to your liking.
