"""
Scoring rubric for the RAG evaluation framework (PRD Section 6).

Five 0-3 scored dimensions (faithfulness, context_precision, context_recall,
answer_relevancy, citation_accuracy) plus the retrieval-correct x
answer-correct 2x2 classification. The level definitions here are the
source of truth for judge.py's LLM-as-judge prompts and for bucketing
RAGAS's native 0-1 scores onto this scale.
"""

from dataclasses import dataclass
from enum import Enum


class Dimension(str, Enum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_RELEVANCY = "answer_relevancy"
    CITATION_ACCURACY = "citation_accuracy"


class Quadrant(str, Enum):
    TRUE_QUALITY = "true_quality"
    SHIPPED_LUCK = "shipped_luck"
    DOOM_LOOP = "doom_loop"
    HONEST_FAILURE = "honest_failure"


# Each dimension is defined independently -- the meaning of "2" on
# faithfulness is not the meaning of "2" on citation_accuracy.
RUBRIC: dict[Dimension, dict[int, str]] = {
    Dimension.FAITHFULNESS: {
        0: "Answer contradicts the retrieved context, or states claims with no support in it.",
        1: "Answer is mostly unsupported: at least one major claim has no basis in the retrieved context.",
        2: "Answer is mostly grounded, but includes a minor unsupported detail or an inference that goes slightly beyond the context.",
        3: "Every claim in the answer is directly supported by the retrieved context.",
    },
    Dimension.CONTEXT_PRECISION: {
        0: "None of the retrieved chunks are relevant to the question.",
        1: "A minority of retrieved chunks are relevant; most are noise.",
        2: "Most retrieved chunks are relevant, with one or two irrelevant chunks mixed in.",
        3: "All (or nearly all) retrieved chunks are relevant to the question.",
    },
    Dimension.CONTEXT_RECALL: {
        0: "None of the golden invariants' source chunks were retrieved.",
        1: "A minority of the golden invariants' source chunks were retrieved -- key information is missing.",
        2: "Most of the golden invariants' source chunks were retrieved; a minor supporting chunk was missed.",
        3: "All of the golden invariants' source chunks were retrieved.",
    },
    Dimension.ANSWER_RELEVANCY: {
        0: "Answer does not address the question asked.",
        1: "Answer is tangentially related but misses the core of the question.",
        2: "Answer addresses the question but includes irrelevant padding or hedges more than necessary.",
        3: "Answer directly and completely addresses the question asked, with no irrelevant content.",
    },
    Dimension.CITATION_ACCURACY: {
        0: "No sources cited, or cited sources do not correspond to any retrieved chunk.",
        1: "Sources are cited but are frequently wrong or attribute claims to chunks that don't support them.",
        2: "Sources are mostly correct, with one citation misattributed or a claim left uncited.",
        3: "Every claim requiring a citation is correctly attributed to the retrieved chunk that supports it.",
    },
}


def level_definitions(dimension: Dimension) -> dict[int, str]:
    return RUBRIC[dimension]


def bucket_continuous_score(score: float) -> int:
    """Map a RAGAS-style continuous 0-1 score onto the 0-3 rubric scale."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    if score < 0.25:
        return 0
    if score < 0.5:
        return 1
    if score < 0.85:
        return 2
    return 3


@dataclass
class RubricScore:
    faithfulness: int
    context_precision: int
    context_recall: int
    answer_relevancy: int
    citation_accuracy: int

    def __post_init__(self):
        for dim in Dimension:
            value = getattr(self, dim.value)
            if value not in (0, 1, 2, 3):
                raise ValueError(f"{dim.value} must be 0-3, got {value}")

    def average(self) -> float:
        return sum(getattr(self, d.value) for d in Dimension) / len(Dimension)

    def as_dict(self) -> dict:
        return {d.value: getattr(self, d.value) for d in Dimension}


def classify_quadrant(retrieval_correct: bool, answer_correct: bool) -> Quadrant:
    """
    2x2 classification of retrieval correctness x answer correctness.

    true_quality:   retrieval correct, answer correct -- the desired case.
    shipped_luck:   retrieval wrong,   answer correct -- right answer for the
                    wrong reason; not reproducible once the question gets harder.
    doom_loop:      retrieval correct, answer wrong   -- the context had the
                    answer and generation still got it wrong; improving
                    retrieval further won't fix this.
    honest_failure: retrieval wrong,   answer wrong   -- a consistent failure;
                    the system correctly failed to answer from context that
                    didn't contain the answer.
    """
    if retrieval_correct and answer_correct:
        return Quadrant.TRUE_QUALITY
    if not retrieval_correct and answer_correct:
        return Quadrant.SHIPPED_LUCK
    if retrieval_correct and not answer_correct:
        return Quadrant.DOOM_LOOP
    return Quadrant.HONEST_FAILURE
