# SWYNEX – Final AI Application: Support Ticket Triage

A beginner-friendly AI application that reads a customer support ticket and
automatically tells you **how upset the customer is**, **which team should
handle it**, and **how urgent it is**. It uses pretrained Hugging Face models,
runs locally, and needs **no API keys**.

Built in four steps during the SWYNEX internship:
Task 2 (model integration) → Task 3 (error handling, evaluation, failure cases) → Task 4 (this final application).

---

## 1. Problem
Support teams receive many tickets and must quickly decide *who* should handle
each one and *how urgent* it is. Doing this by hand is slow and inconsistent.
This application automates that first pass so people can focus on solving the
problems.

**Example:** "I was charged twice for my subscription this month and nobody is replying to me." → category **billing**, priority **High**.

## 2. Method
Two pretrained models are combined with a simple rule:

| Step | Model / rule | Output |
|---|---|---|
| 1. Sentiment analysis | `distilbert-base-uncased-finetuned-sst-2-english` | POSITIVE or NEGATIVE, with a score |
| 2. Category (zero-shot classification) | `facebook/bart-large-mnli` | One of: billing, technical issue, account access, delivery, general inquiry |
| 3. Priority rule | NEGATIVE with score ≥ 0.90 → High, other NEGATIVE → Medium, otherwise Low | High / Medium / Low |

Zero-shot classification means no training data was needed: the model matches the
ticket text against the category names directly.

Input validation and error handling (empty, invalid, very long or very short text,
missing files, model download failures) are built in, so the app returns a clear
message instead of crashing.

## 3. Demo
You can use the app in three ways.

**A. Command line**
```bash
python app.py --text "I was charged twice and nobody replies to me."
python app.py --file examples/sample_inputs.txt --out examples/sample_outputs.json
```

**B. Web interface (Gradio)**
```bash
python gradio_app.py
```
Open the local link shown in the terminal (usually http://127.0.0.1:7860), paste a ticket, and read the result.

**C. Google Colab notebook**
Open `demo.ipynb` in Colab (File → Open notebook → GitHub tab) and click Runtime → Run all. The last cell creates a public link for the web interface, so no powerful laptop is needed.

**Sample output**
```
Ticket   : I was charged twice for my subscription this month and nobody is replying to me.
Sentiment: NEGATIVE (1.0)
Category : billing (0.661)
Priority : High
```
More examples: `examples/sample_outputs.json`.

## 4. Evaluation
The model was tested on 20 labelled tickets (`examples/eval_set.csv`): 15 standard
and 5 hard cases (mixed sentiment, sarcasm, very short text, ambiguous category).

| Group | Accuracy |
|---|---|
| Overall | **95.0%** (19/20) |
| Standard cases | **93.3%** (14/15) |
| Hard cases | **100%** (5/5) |

Run `python evaluate.py` to reproduce this. Details are in `EVALUATION_RESULTS.md`,
and the one misclassified ticket is analysed in [FAILURE_CASES.md](FAILURE_CASES.md).

## 5. Limitations
- **Small test set.** 20 tickets give a rough estimate only; real-world accuracy may be lower.
- **Low confidence on some correct answers** (for example 0.37 to 0.50), so those predictions are less trustworthy.
- **Fixed categories.** Only five categories are supported, and tickets that fit two categories get just one label.
- **Sentiment is not urgency.** Priority is based on how negative the text sounds, not on real urgency, customer value, or words like "urgent". A calm but critical ticket could be rated Low.
- **English only.** The models are not designed for other languages.
- **Sarcasm and mixed sentiment** can mislead the sentiment model, even though they passed in this small test.
- **Speed and size.** The first run downloads about 1.5 GB of models, and runs on a CPU are slow for large batches.
- **Text length.** Input over 2000 characters is truncated.

## 6. Ethics and responsible use
- **Decision support, not a decision maker.** A human should review results, especially low-confidence ones, before any ticket is deprioritised.
- **Privacy.** The models run locally and no ticket text is sent to an external service. Even so, avoid using real personal data (names, card numbers, addresses) in tests, and follow your organisation's data rules.
- **Bias and fairness.** The pretrained models learned from public text and may handle some writing styles, dialects, or non-native English worse than others. This could unfairly lower the priority of some customers, so check results across different customer groups before real use.
- **Transparency.** Confidence scores are shown so users can see how certain the model is. Users should be told when an automated system is helping to prioritise their ticket.
- **No sensitive decisions.** Do not use this application for anything that affects people's safety, finances, or legal rights without proper testing and human oversight.
- **Licences.** Both models are openly available on the Hugging Face Hub; check their model cards for licence terms before commercial use.

## 7. Setup
```bash
git clone https://github.com/Mathiofficial/SWYNEX-Final-AI-Application.git
cd SWYNEX-Final-AI-Application
python -m venv venv
venv\Scripts\activate          # Windows (on Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Requires Python 3.9 or newer.

## 8. Project structure
```
app.py                       # triage logic, validation, error handling, CLI
gradio_app.py                # web interface
evaluate.py                  # evaluation script
demo.ipynb                   # Colab notebook demo
FAILURE_CASES.md             # failure case analysis
EVALUATION_RESULTS.md        # generated by evaluate.py
requirements.txt
examples/
  sample_inputs.txt
  sample_outputs.json
  eval_set.csv
  eval_results.json          # generated by evaluate.py
```

## 9. Future improvements
- Fine-tune on real labelled tickets instead of zero-shot classification.
- Allow multiple categories per ticket.
- Make priority consider urgency words and not only sentiment.
- Support more languages.
- Grow the evaluation set and score priority as well as category.
- Automatically send low-confidence tickets to a human agent.

## Tech
Python, Hugging Face Transformers, PyTorch, Gradio
