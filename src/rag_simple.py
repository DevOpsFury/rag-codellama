from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI


# ---------------------------------------------
# 1. Load embedding model and ChromaDB database
# ---------------------------------------------
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

client = chromadb.PersistentClient(path="embeddings/chroma")
collection = client.get_or_create_collection("terraform_docs")


# ---------------------------------------------------------
# 2. Function: generate embedding and search for context
# ---------------------------------------------------------
def search_context(query: str, top_k: int = 5) -> str:
    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    if not results["documents"]:
        return ""

    documents = results["documents"][0]
    return "\n\n".join(documents)


# -----------------------------------------
# 3. Function: query LLM model with context
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

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message["content"]


# -------------------------------------------------------
# 4. Usage example — Terraform analysis and refactoring
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
    response = ask_model(question + "\n\nKod:\n" + tf_snippet, context)

    print("\n=== MODEL RESPONSE ===\n")
    print(response)
