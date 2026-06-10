#!/usr/bin/env python3
"""
MedQuAD Ingestion Script — Phase 0
------------------------------------
Parses all MedQuAD XML files and produces:
  1. data/processed/corpus.json   → documents for RAG indexing
  2. data/eval/eval_set.json      → held-out QA pairs for RAGAS evaluation

Usage:
    python src/ingestion/ingest_medquad.py

Edit MEDQUAD_DIR below to match where you cloned MedQuAD.
"""

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MEDQUAD_DIR = Path("data/raw/MedQuAD")
OUTPUT_DIR = Path("data/processed")
EVAL_DIR = Path("data/eval")
EVAL_SIZE = 50  # held-out pairs for evaluation
MIN_ANSWER_LEN = 80  # skip very short / empty answers
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def parse_xml_file(xml_path: Path) -> list[dict]:
    """Parse one MedQuAD XML file → list of QA records."""
    records = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [SKIP] {xml_path.name} — parse error: {e}")
        return records

    doc_id = root.get("id", "")
    source = root.get("source", xml_path.parent.name)
    url = root.get("url", "")

    focus_elem = root.find("Focus")
    focus = focus_elem.text.strip() if (focus_elem is not None and focus_elem.text) else ""

    qa_pairs_elem = root.find("QAPairs")
    if qa_pairs_elem is None:
        return records

    for qa_pair in qa_pairs_elem.findall("QAPair"):
        pid = qa_pair.get("pid", "")

        q_elem = qa_pair.find("Question")
        a_elem = qa_pair.find("Answer")

        if q_elem is None or a_elem is None:
            continue

        question = (q_elem.text or "").strip()
        answer = (a_elem.text or "").strip()
        qtype = q_elem.get("qtype", "")
        qid = q_elem.get("qid", "")

        # Skip pairs with missing or very short answers
        if not question or len(answer) < MIN_ANSWER_LEN:
            continue

        records.append(
            {
                "id": f"{doc_id}_{pid}",
                "qid": qid,
                "question": question,
                "answer": answer,
                "qtype": qtype,
                "focus": focus,
                "source": source,
                "source_url": url,
                "source_file": xml_path.name,
            }
        )

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    if not MEDQUAD_DIR.exists():
        print(f"ERROR: MedQuAD directory not found at '{MEDQUAD_DIR.resolve()}'")
        print("Edit the MEDQUAD_DIR variable at the top of this script.")
        return

    print(f"Scanning: {MEDQUAD_DIR.resolve()}\n")

    all_records: list[dict] = []
    folder_stats: dict[str, int] = {}

    for xml_file in sorted(MEDQUAD_DIR.rglob("*.xml")):
        records = parse_xml_file(xml_file)
        if records:
            folder = xml_file.parent.name
            folder_stats[folder] = folder_stats.get(folder, 0) + len(records)
            all_records.extend(records)

    # Summary
    print("QA pairs found per collection:")
    print("-" * 52)
    for folder, count in sorted(folder_stats.items()):
        print(f"  {folder:<42} {count:>5}")
    print("-" * 52)
    print(f"  {'TOTAL':<42} {len(all_records):>5}\n")

    if not all_records:
        print("No records found. Double-check MEDQUAD_DIR path.")
        return

    # Split: eval vs corpus
    random.shuffle(all_records)
    eval_set = all_records[:EVAL_SIZE]
    corpus = all_records[EVAL_SIZE:]

    # Save
    corpus_path = OUTPUT_DIR / "corpus.json"
    eval_path = EVAL_DIR / "eval_set.json"

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"Corpus saved   → {corpus_path}   ({len(corpus)} documents)")
    print(f"Eval set saved → {eval_path}  ({len(eval_set)} QA pairs)")
    print("\nPhase 0 complete. Ready for Phase 1 — baseline RAG.")


if __name__ == "__main__":
    main()
