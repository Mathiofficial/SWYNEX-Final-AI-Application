# Failure Cases

The evaluation set (`examples/eval_set.csv`) has 20 tickets, including 5 **hard
cases** designed to challenge the models. Results: **19/20 correct (95.0%)**.

## Misclassified tickets

| Ticket | Expected | Predicted | Confidence | Why it failed |
|---|---|---|---|---|
| Please refund the extra amount deducted from my card. | billing | general inquiry | 0.384 | No explicit billing words such as "charged" or "invoice". "Refund" and "deducted" are indirect clues, and the zero-shot model read the message as a generic request. Confidence was low. |

## Hard case types tested

| Type | Example | Result | Why it is difficult |
|---|---|---|---|
| Mixed sentiment | "Thanks for the help earlier but I am still being charged twice." | Correct (0.653) | Polite words can hide an unresolved complaint, and the sentiment model gives one label for the whole text, so priority may be rated too low. |
| Sarcasm | "Great another app crash. Just what I needed today." | Correct (0.663) | Positive words such as "Great" can mislead the sentiment model. |
| Too short | "help" | Correct (0.453) | Almost no context. The correct label may be partly luck, and confidence is low. |
| Ambiguous category | "I paid for express delivery but the package is late and I want my money back." | Correct (0.811) | Fits both *billing* and *delivery*; only one label is chosen. |
| Ambiguous category | "I cannot log in because the app keeps crashing on the login screen." | Correct (0.673) | Fits both *account access* and *technical issue*. |

## Other observations
- The hard cases all passed on category, but with lower confidence (0.45 to 0.81), so the model is less certain there.
- Some correct predictions had low confidence: "What plans do you offer and how much do they cost?" (0.373), "My order has not arrived and the tracking is not updating." (0.497) and "help" (0.453). These are the tickets a human should review.
- The evaluation scores category only. Priority for mixed-sentiment and sarcastic tickets was not scored; the sentiment model may under-rate urgency there.
- With only 20 tickets (5 hard), 95% is a rough estimate and not a guarantee of real-world accuracy.

## Mitigations in the prototype
- Empty or invalid input returns a clear error instead of crashing.
- Very short or very long input is flagged with a warning.
- Confidence scores are returned so low-confidence predictions can be reviewed by a human.

## Possible improvements
- Fine-tune on real labelled tickets instead of using zero-shot classification.
- Allow multi-label output for tickets that span two categories.
- Use a sentiment model that handles mixed or sarcastic text better.
- Route low-confidence predictions (for example below 0.5) to a human agent.
- Expand the evaluation set and score priority as well as category.
