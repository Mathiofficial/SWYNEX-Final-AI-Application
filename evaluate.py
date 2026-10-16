"""Evaluate the triage model on a small labelled set.

Usage:  python evaluate.py
Reads  examples/eval_set.csv  (text, expected_category, type)
Writes EVALUATION_RESULTS.md and examples/eval_results.json
"""
import csv
import json
import sys
from collections import defaultdict

from app import load_models, safe_triage

EVAL_FILE = "examples/eval_set.csv"
REPORT_FILE = "EVALUATION_RESULTS.md"
JSON_FILE = "examples/eval_results.json"


def load_eval(path):
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"Error: evaluation file not found: {path}")
    if not rows:
        sys.exit("Error: evaluation file is empty.")
    return rows


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main():
    rows = load_eval(EVAL_FILE)
    try:
        sentiment, classifier = load_models()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")

    results = []
    for row in rows:
        r = safe_triage(row["text"], sentiment, classifier)
        predicted = r.get("category")
        results.append({
            "text": row["text"],
            "type": row["type"],
            "expected": row["expected_category"],
            "predicted": predicted,
            "confidence": r.get("category_score"),
            "correct": predicted == row["expected_category"],
            "error": r.get("error"),
        })

    total = len(results)
    correct = sum(x["correct"] for x in results)
    groups = defaultdict(lambda: [0, 0])  # group -> [correct, total]
    for x in results:
        g = "standard" if x["type"] == "standard" else "hard"
        groups[g][1] += 1
        groups[g][0] += x["correct"]

    lines = ["# Evaluation Results", ""]
    lines.append(f"**Overall accuracy: {pct(correct, total)}** ({correct}/{total})")
    lines.append("")
    for g in ("standard", "hard"):
        c, t = groups[g]
        lines.append(f"- {g} cases: {pct(c, t)} ({c}/{t})")
    lines += ["", "## Misclassified tickets", ""]
    wrong = [x for x in results if not x["correct"]]
    if wrong:
        lines.append("| Ticket | Type | Expected | Predicted | Confidence |")
        lines.append("|---|---|---|---|---|")
        for x in wrong:
            lines.append(f"| {x['text']} | {x['type']} | {x['expected']} | "
                         f"{x['predicted'] or x['error']} | {x['confidence']} |")
    else:
        lines.append("None - every ticket was classified correctly.")
    report = "\n".join(lines) + "\n"

    print(report)
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {REPORT_FILE} and {JSON_FILE}")
    except OSError as exc:
        sys.exit(f"Error: could not write results: {exc}")


if __name__ == "__main__":
    main()
