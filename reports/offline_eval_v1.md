# Offline Evaluation Report -- v1

Generated 2026-07-20 | Run dated 2026-07-20T05:48:00.134459+00:00

Golden dataset: `data/golden_dataset/golden_v1.json` (42 items). Judge: openai (gpt-4o-mini).

## Rubric scores

| Dimension | Average (0-3) |
|---|---|
| faithfulness | 2.76 |
| context_precision | 2.79 |
| context_recall | 2.93 |
| answer_relevancy | 2.67 |
| citation_accuracy | 2.86 |
| **Overall** | **2.80** |

## Retrieval x answer correctness (2x2)

| Quadrant | Count | % of items |
|---|---|---|
| True quality (retrieval correct, answer correct) | 35 | 83% |
| Shipped luck (retrieval wrong, answer correct) | 1 | 2% |
| Doom loop (retrieval correct, answer wrong) | 5 | 12% |
| Honest failure (retrieval wrong, answer wrong) | 1 | 2% |

## Scores by category

| Category | Average rubric score (0-3) |
|---|---|
| Employee | 2.83 |
| Engineering | 2.80 |
| Finance | 2.77 |
| HR | 2.86 |
| Product | 2.69 |
| Support | 2.86 |

## Judge calibration

Hand-labeled 18 items (blind to judge scores) and compared against the judge.

| Dimension | Cohen's kappa |
|---|---|
| faithfulness | 0.34 |
| context_precision | 0.00 |
| context_recall | 0.65 |
| answer_relevancy | 0.00 |
| citation_accuracy | 0.47 |
| retrieval_correct | 1.00 |
| answer_correct | 0.77 |

**Top 5 judge/human disagreements**, ranked by total score gap:

- `G035` (gap 7): human said answer_correct=True, judge said False -- *SHIPPED LUCK candidate: golden needs PM-005 + PM-009, but only PM-009 was retrieved (PM-005 missing). Answer still substantively covers both invariants -- PM-009 itself references "user research finding 3" so the claim is a reasonable synthesis, not pure fabrication, but the required PM-005 citation is missing entirely. Good candidate for the PRD-requested shipped-luck example.*
- `G022` (gap 4): human said answer_correct=False, judge said False -- *Missing the SSO invariant entirely, even though SUP-001 (retrieved) contains it -- doom_loop, matches automated judge.*
- `G006` (gap 3): human said answer_correct=False, judge said False -- *Missing the 30-day mutual-agreement extension invariant -- doom_loop.*
- `G039` (gap 3): human said answer_correct=True, judge said True -- *no note left*
- `G010` (gap 3): human said answer_correct=True, judge said True -- *DISAGREEMENT: automated judge scored this doom_loop, but the answer explicitly covers both invariants (Tier 4/CFO approval, correctly states Board approval not required under $250k). Worth investigating why the judge marked it incorrect.*

## Failure taxonomy

6 / 42 items failed (14%), bottom-up clustered into 6 group(s):

| Rank | Failure pattern | Count | Example items |
|---|---|---|---|
| 1 | Incomplete or inaccurate information | 1 | G004 |
| 2 | Missing critical detail | 1 | G006 |
| 3 | Missing required information | 1 | G022 |
| 4 | Missing critical information | 1 | G033 |
| 5 | Lack of specific supporting details | 1 | G035 |
| 6 | Lack of critical detail | 1 | G042 |
