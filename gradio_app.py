"""Simple web interface for the Support Ticket Triage app (SWYNEX Task 4).

Run locally:   python gradio_app.py
Run in Colab:  use the last cells of demo.ipynb (creates a public share link)
"""
import gradio as gr

from app import load_models, safe_triage

sentiment, classifier = load_models()

EXAMPLES = [
    "I was charged twice for my subscription this month and nobody is replying to me.",
    "The app crashes every time I try to upload a photo.",
    "I forgot my password and the reset email never arrives.",
    "My order was supposed to arrive last week and the tracking hasn't updated.",
    "Thanks for the quick help yesterday, could you tell me what plans you offer?",
]


def run(ticket):
    r = safe_triage(ticket, sentiment, classifier)
    if "error" in r:
        return "-", "-", "-", f"Error: {r['error']}"
    return (
        f"{r['sentiment']} ({r['sentiment_score']})",
        f"{r['category']} ({r['category_score']})",
        r["priority"],
        r.get("warning", "OK. Low confidence scores should be reviewed by a human."),
    )


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=4, label="Customer ticket", placeholder="Paste a support ticket here..."),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Category"),
        gr.Textbox(label="Priority"),
        gr.Textbox(label="Notes"),
    ],
    examples=EXAMPLES,
    title="AI Support Ticket Triage",
    description="Detects sentiment, routes the ticket to a category, and suggests a priority. "
                "Decision support only - a human should review low-confidence results.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
