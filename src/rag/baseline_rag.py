"""
Phase 1 — Baseline RAG Pipeline
---------------------------------
Naive RAG: embed answers → store in ChromaDB → retrieve top-k → generate with LLM

Run:
    python src/rag/baseline_rag.py

First run will index all 16k documents (~5-15 min depending on CPU).
After that, ChromaDB loads from disk instantly.
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()

# Config
CORPUS_PATH = Path("data/processed/corpus.json")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "medquad_baseline"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
BATCH_SIZE = 200
LLM_MODEL = "gemma-4-31b-it"  # Verify exact name in AI Studio

# Embedding function (runs locally, no API cost)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


# Indexing


def get_collection(chroma_client):
    """Load existing collection or create and index from scratch."""
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
        count = collection.count()
        if count > 0:
            print(f"Collection loaded from disk — {count} documents ready.")
            return collection
    except Exception:
        pass

    return _index_corpus(chroma_client)


def _index_corpus(chroma_client):
    """Embed all documents and store in ChromaDB."""
    print("Loading corpus ...")
    corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
    print(f"{len(corpus)} documents found.\n")

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print("Indexing — this runs once and then persists to disk.")
    print("Estimated time: 5–15 minutes on CPU.\n")

    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Embedding batches"):
        batch = corpus[i : i + BATCH_SIZE]

        collection.add(
            ids=[f"doc_{i + j}" for j in range(len(batch))],
            documents=[doc["answer"] for doc in batch],
            metadatas=[
                {
                    "question": doc["question"],
                    "focus": doc["focus"],
                    "qtype": doc["qtype"],
                    "source": doc["source"],
                    "source_url": doc["source_url"],
                }
                for doc in batch
            ],
        )

    print(f"\nIndexing complete — {collection.count()} documents stored.\n")
    return collection


# Retrieval


def retrieve(collection, query: str, top_k: int = TOP_K) -> tuple[list, list]:
    """Embed query and return top-k (documents, metadatas)."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    return results["documents"][0], results["metadatas"][0]


# Generation


def generate(query: str, docs: list[str], metas: list[dict]) -> str:
    """Send retrieved context + question to LLM and return answer."""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Build numbered context block
    context = ""
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        context += f"[Source {i} | {meta['focus']} | {meta['source']}]\n{doc}\n\n"

    prompt = f"""You are a medical information assistant that answers questions using ONLY the provided context.

Rules:
- Answer using ONLY the context below. Do not add outside knowledge.
- Cite sources like [Source 1], [Source 2] when referencing information.
- If the context does not contain enough information, say:
  "I don't have enough information to answer this question."
- Always end with the disclaimer below.

⚠️ Disclaimer: This information is for educational and research purposes only.
Always consult a qualified physician or specialist for medical decisions.

---
Context:
{context}
---

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )

    return response.text


# Full Pipeline


def ask(collection, query: str) -> str:
    """End-to-end: question → retrieve → generate → print answer."""
    print(f"\n{'=' * 65}")
    print(f"Question: {query}")
    print(f"{'=' * 65}")

    docs, metas = retrieve(collection, query)

    print("\nRetrieved sources:")
    for i, meta in enumerate(metas, start=1):
        print(f"  {i}. {meta['focus']}  [{meta['source']}]  ({meta['qtype']})")

    answer = generate(query, docs, metas)
    print(f"\nAnswer:\n{answer}\n")
    return answer


# Main


def main():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_collection(chroma_client)

    test_questions = [
        "What is polycystic kidney disease?",
        "How is Noonan syndrome inherited?",
        "What are the symptoms of celiac disease?",
    ]

    for question in test_questions:
        ask(collection, question)


if __name__ == "__main__":
    main()
