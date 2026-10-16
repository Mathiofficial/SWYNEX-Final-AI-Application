"""Support Ticket Triage - SWYNEX Task 3 (Intelligent Feature).

Builds on Task 2 and adds:
  * input validation and error handling (empty / invalid / very long / very short text)
  * safe_triage(): never crashes, returns an "error" entry instead
  * clear messages for missing files and model-loading failures

Models (public Hugging Face Hub, no API key needed):
  1. Sentiment analysis       -> how upset is the customer?
  2. Zero-shot classification -> which team should handle the ticket?
"""
import argparse
import json
import sys

from transformers import pipeline

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
CATEGORIES = ["billing", "technical issue", "account access", "delivery", "general inquiry"]
MAX_CHARS = 2000
MIN_RELIABLE_CHARS = 10


class InvalidTicketError(ValueError):
    """Raised when a ticket cannot be processed (empty, wrong type...)."""


def load_models():
    """Load both models. Raises RuntimeError with a readable message on failure."""
    print("Loading models (first run downloads them)...", file=sys.stderr)
    try:
        sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
        classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    except Exception as exc:  # network issues, disk space, missing libs...
        raise RuntimeError(
            f"Could not load the models ({exc.__class__.__name__}: {exc}). "
            "Check your internet connection and free disk space."
        ) from exc
    return sentiment, classifier


def validate_text(text):
    """Return (clean_text, warnings). Raise InvalidTicketError for unusable input."""
    warnings = []
    if not isinstance(text, str):
        raise InvalidTicketError("Ticket must be text (got %s)." % type(text).__name__)
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        raise InvalidTicketError("Ticket is empty.")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        warnings.append(f"Text was longer than {MAX_CHARS} characters and was truncated.")
    if len(text) < MIN_RELIABLE_CHARS:
        warnings.append("Very short text; prediction may be unreliable.")
    return text, warnings


def priority_from(label, score):
    if label == "NEGATIVE" and score >= 0.90:
        return "High"
    if label == "NEGATIVE":
        return "Medium"
    return "Low"


def triage(text, sentiment, classifier):
    """Core logic (assumes text is already validated)."""
    sent = sentiment(text, truncation=True, max_length=512)[0]
    cls = classifier(text, candidate_labels=CATEGORIES)
    return {
        "input": text,
        "sentiment": sent["label"],
        "sentiment_score": round(sent["score"], 3),
        "category": cls["labels"][0],
        "category_score": round(cls["scores"][0], 3),
        "priority": priority_from(sent["label"], sent["score"]),
    }


def safe_triage(text, sentiment, classifier):
    """Validate + triage. Never raises: failures come back as {'input', 'error'}."""
    shown = str(text)[:80]
    try:
        clean, warnings = validate_text(text)
    except InvalidTicketError as exc:
        return {"input": shown, "error": str(exc)}
    try:
        result = triage(clean, sentiment, classifier)
    except Exception as exc:
        return {"input": shown, "error": f"Model failed: {exc.__class__.__name__}: {exc}"}
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


def print_result(r):
    print(f"\nTicket   : {r['input']}")
    if "error" in r:
        print(f"ERROR    : {r['error']}")
        return
    print(f"Sentiment: {r['sentiment']} ({r['sentiment_score']})")
    print(f"Category : {r['category']} ({r['category_score']})")
    print(f"Priority : {r['priority']}")
    if "warning" in r:
        print(f"Warning  : {r['warning']}")


def main():
    parser = argparse.ArgumentParser(description="AI support ticket triage")
    parser.add_argument("--text", help="A single ticket text")
    parser.add_argument("--file", help="Text file with one ticket per line")
    parser.add_argument("--out", help="Write results to this JSON file")
    args = parser.parse_args()

    if args.text is not None:
        tickets = [args.text]
    elif args.file:
        try:
            with open(args.file, encoding="utf-8") as f:
                tickets = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            sys.exit(f"Error: file not found: {args.file}")
        except UnicodeDecodeError:
            sys.exit(f"Error: could not read {args.file} as UTF-8 text.")
        if not tickets:
            sys.exit(f"Error: {args.file} contains no tickets.")
    else:
        parser.error("Provide --text or --file")

    try:
        sentiment, classifier = load_models()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")

    results = [safe_triage(t, sentiment, classifier) for t in tickets]
    for r in results:
        print_result(r)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {args.out}", file=sys.stderr)
        except OSError as exc:
            sys.exit(f"Error: could not write {args.out}: {exc}")


if __name__ == "__main__":
    main()
