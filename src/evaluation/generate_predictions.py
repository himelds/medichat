"""Run baseline RAG on eval set, save predictions for RAGAS evaluation."""

import json
import time
from pathlib import Path
from tqdm import tqdm

import chromadb
from src.rag.baseline_rag import (
    get_collection,
    retrieve,
    generate,
    CHROMA_DIR,
)

# Smoke test mode
LIMIT = None


def main():
    eval_set = json.loads(Path("data/eval/eval_set.json").read_text(encoding="utf-8"))

    if LIMIT is not None:
        eval_set = eval_set[:LIMIT]
        print(f"SMOKE TEST MODE: {LIMIT} samples\n")

    # Initialize ChromaDB
    print("Loading ChromaDB collection...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_collection(chroma_client)
    print()

    predictions = []
    output_path = Path("data/eval/predictions_baseline.json")

    for item in tqdm(eval_set, desc="Generating"):
        try:
            # Retrieve top-k documents
            docs, metas = retrieve(collection, item["question"])

            # Generate answer
            answer = generate(item["question"], docs, metas)

            predictions.append(
                {
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "answer": answer,
                    "contexts": docs,  # already list of strings ✓
                    "qtype": item.get("qtype"),
                    "focus": item.get("focus"),
                }
            )

            # Save after each — crash safety
            output_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
            time.sleep(0.5)

        except Exception as e:
            print(f"\nFailed: {item['question'][:60]}... -> {e}")
            time.sleep(5)

    print(f"\nSaved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
