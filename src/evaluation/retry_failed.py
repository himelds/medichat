"""Retry generation for samples missing from predictions_baseline.json."""

import json
import time
from pathlib import Path
from tqdm import tqdm

import chromadb
from src.rag.baseline_rag import get_collection, retrieve, generate, CHROMA_DIR


def main():
    eval_set = json.loads(Path("data/eval/eval_set.json").read_text(encoding="utf-8"))
    predictions = json.loads(
        Path("data/eval/predictions_baseline.json").read_text(encoding="utf-8")
    )

    done_questions = {p["question"] for p in predictions}
    missing = [item for item in eval_set if item["question"] not in done_questions]

    if not missing:
        print("All 50 samples already generated!")
        return

    print(f"Retrying {len(missing)} previously failed samples...")
    print(f"Currently have: {len(predictions)}/{len(eval_set)} predictions\n")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_collection(chroma_client)
    output_path = Path("data/eval/predictions_baseline.json")

    new_successes = 0

    for item in tqdm(missing, desc="Retrying"):
        try:
            docs, metas = retrieve(collection, item["question"])
            answer = generate(item["question"], docs, metas)

            predictions.append(
                {
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "answer": answer,
                    "contexts": docs,
                    "qtype": item.get("qtype"),
                    "focus": item.get("focus"),
                }
            )

            # Save after each — crash safety
            output_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
            new_successes += 1
            time.sleep(3)  # extra cushion — Google API stress কম রাখো

        except Exception as e:
            print(f"\nStill failed: {item['question'][:60]}... -> {type(e).__name__}")
            time.sleep(5)

    print("\n=== Retry Results ===")
    print(f"New successes: {new_successes}/{len(missing)}")
    print(f"Total predictions: {len(predictions)}/{len(eval_set)}")


if __name__ == "__main__":
    main()
