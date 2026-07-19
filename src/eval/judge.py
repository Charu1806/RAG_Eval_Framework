"""
LLM-as-judge for the RAG evaluation framework (PRD Section 6).

faithfulness, context_precision, context_recall, and answer_relevancy are
scored via RAGAS, which maps cleanly onto them:
  - faithfulness           -> ragas.metrics.Faithfulness
  - context_precision      -> ragas.metrics.LLMContextPrecisionWithoutReference
                               (no fixed reference answer exists in this
                               dataset -- see note below)
  - context_recall         -> ragas.metrics.LLMContextRecall, using the
                               golden item's invariants joined into a single
                               string as the "reference", since context
                               recall needs something to check chunk coverage
                               against and invariants ARE the required content
  - answer_relevancy       -> ragas.metrics.ResponseRelevancy
Each RAGAS metric returns a continuous 0-1 score, bucketed onto the shared
0-3 rubric scale via rubric.bucket_continuous_score().

citation_accuracy and answer_correct (the "answer" half of the 2x2
classification) are scored via a custom LLM-as-judge prompt, since no RAGAS
metric covers citation attribution or invariant-based correctness.

retrieval_correct (the other half of the 2x2 classification) is computed
deterministically by checking whether every golden source_chunk_id appears
in the retrieved chunks' embedded "Document ID:" text, rather than asking
the LLM to guess -- the golden dataset carries objective ground truth for
this, so a deterministic check is strictly more reliable than an LLM
judgment call here.

Judge model is swappable. Default fallback order is Mistral -> OpenAI ->
Anthropic (Mistral first since it's already used by the RAG pipeline under
test and needs no extra credential). Whichever provider's probe call
succeeds first is used for the entire run, and is printed clearly so it's
never silently swapped without you knowing.
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.eval.rubric import RUBRIC, Dimension, Quadrant, RubricScore, bucket_continuous_score, classify_quadrant

# ── Rate-limit handling ──────────────────────────────────────────────────────
# offline_eval.py paces its own generation calls, but a single item also fires
# 4 RAGAS metric calls + 1 custom judge call, all unpaced -- and when the
# judge is Mistral (the default), those share the same free-tier 1 req/sec
# limit as the generator. langchain-mistralai's own built-in retry isn't
# always enough to absorb a 429, so every judge-facing call below is paced
# and retried at this level too.

JUDGE_CALL_PACE_SECONDS = 1.5  # matches rag_pipeline's own Mistral free-tier pacing


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limited" in text


def _call_with_backoff(fn, *, max_attempts: int = 4, base_delay: float = 5.0):
    """Call fn() with exponential backoff on rate-limit-shaped errors. Any
    other exception (auth failure, malformed response, etc.) is re-raised
    immediately -- only rate limits are worth waiting out."""
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            time.sleep(JUDGE_CALL_PACE_SECONDS)  # pace even on success, to avoid tripping the next call
            return result
        except Exception as e:  # noqa: BLE001 -- inspected immediately below, not swallowed
            if not _is_rate_limit_error(e) or attempt == max_attempts:
                raise
            delay = base_delay * attempt
            print(f"[judge] rate limited (attempt {attempt}/{max_attempts}), waiting {delay:.0f}s before retry")
            time.sleep(delay)


# ── Judge provider selection ─────────────────────────────────────────────────

JUDGE_PROVIDERS = ("mistral", "openai", "anthropic")

_PROVIDER_ENV_KEYS = {
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_PROVIDER_DEFAULT_MODELS = {
    "mistral": "mistral-large-latest",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
}


def _build_chat_model(provider: str):
    api_key = os.environ.get(_PROVIDER_ENV_KEYS[provider])
    if not api_key:
        raise EnvironmentError(f"{_PROVIDER_ENV_KEYS[provider]} not set")
    model_name = os.environ.get("JUDGE_MODEL_NAME", _PROVIDER_DEFAULT_MODELS[provider])

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model_name, api_key=api_key, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0)
    raise ValueError(f"unknown provider: {provider!r}")


@dataclass
class Judge:
    """A chat model selected from JUDGE_PROVIDERS, wrapped for RAGAS use."""

    provider: str
    model_name: str
    llm: object  # langchain chat model, used directly for the custom prompt
    ragas_llm: object = field(init=False)  # LangchainLLMWrapper, used by RAGAS metrics

    def __post_init__(self):
        self.ragas_llm = LangchainLLMWrapper(self.llm)


def get_judge(preferred_provider: Optional[str] = None) -> Judge:
    """
    Try providers in fallback order (default Mistral -> OpenAI -> Anthropic),
    picking the first one that constructs AND answers a cheap probe call.
    That provider is used for the entire run -- never swapped mid-run, so
    scores stay comparable across items.
    """
    order = list(JUDGE_PROVIDERS)
    if preferred_provider:
        if preferred_provider not in JUDGE_PROVIDERS:
            raise ValueError(f"preferred_provider must be one of {JUDGE_PROVIDERS}, got {preferred_provider!r}")
        order = [preferred_provider] + [p for p in JUDGE_PROVIDERS if p != preferred_provider]

    errors = {}
    for provider in order:
        try:
            llm = _build_chat_model(provider)
            llm.invoke("Reply with the single word: ok")  # cheap connectivity probe
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any provider failure should fall through
            errors[provider] = str(e)
            print(f"[judge] {provider} unavailable ({e}); trying next provider")
            continue
        model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or _PROVIDER_DEFAULT_MODELS[provider]
        print(f"[judge] using {provider} ({model_name}) for this run")
        return Judge(provider=provider, model_name=str(model_name), llm=llm)

    raise RuntimeError(f"No judge provider available. Tried: {errors}")


def get_raw_embeddings():
    """Same embedding model as the RAG pipeline (all-MiniLM-L6-v2). Runs
    locally, no API key required. Returns the plain langchain embeddings
    object (not RAGAS-wrapped) so other modules (e.g. failure_taxonomy.py's
    clustering) can reuse it without a RAGAS dependency."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings():
    """RAGAS-wrapped version of get_raw_embeddings(), for use with RAGAS
    metrics like answer_relevancy."""
    return LangchainEmbeddingsWrapper(get_raw_embeddings())


# ── Deterministic retrieval-correctness check ────────────────────────────────

_DOC_ID_RE = re.compile(r"Document ID:\s*([A-Z]+-\d+)")


def extract_chunk_ids(retrieved_contexts: list) -> set:
    """Pull 'Document ID: XXX-000' tokens out of retrieved chunk text."""
    ids = set()
    for chunk in retrieved_contexts:
        ids.update(_DOC_ID_RE.findall(chunk))
    return ids


def compute_retrieval_correct(retrieved_contexts: list, golden_source_chunk_ids: list) -> bool:
    """True only if every golden source chunk was retrieved (not just some)."""
    return set(golden_source_chunk_ids).issubset(extract_chunk_ids(retrieved_contexts))


# ── Custom LLM-as-judge: citation_accuracy + answer_correct ─────────────────

_JUDGE_PROMPT_TEMPLATE = """You are a strict grading assistant for a RAG (retrieval-augmented generation) system evaluation.

Question:
{question}

Required facts (invariants) a correct answer must include, in any phrasing:
{invariants}

Retrieved context chunks (the generator's source material):
{contexts}

Generated answer:
{answer}

Score the answer on two things:

1. answer_correct (true/false): true only if the answer covers ALL of the
   required facts listed above, in substance. Paraphrasing is fine; a
   missing or contradicted fact makes this false.

2. citation_accuracy (integer 0-3), using this scale:
{citation_levels}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"answer_correct": true, "answer_correct_reasoning": "...", "citation_accuracy": 0, "citation_reasoning": "..."}}
"""


def _parse_judge_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    required = {"answer_correct", "answer_correct_reasoning", "citation_accuracy", "citation_reasoning"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"judge response missing fields {missing}. Raw response: {raw!r}")
    if data["citation_accuracy"] not in (0, 1, 2, 3):
        raise ValueError(f"citation_accuracy out of range: {data['citation_accuracy']!r}. Raw response: {raw!r}")
    return data


def judge_citation_and_correctness(judge: Judge, question: str, invariants: list, retrieved_contexts: list, answer: str) -> dict:
    citation_levels = "\n".join(f"  {lvl}: {desc}" for lvl, desc in RUBRIC[Dimension.CITATION_ACCURACY].items())
    contexts_block = "\n\n".join(f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(retrieved_contexts))
    invariants_block = "\n".join(f"- {inv}" for inv in invariants)

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        invariants=invariants_block,
        contexts=contexts_block,
        answer=answer,
        citation_levels=citation_levels,
    )
    raw = _call_with_backoff(lambda: judge.llm.invoke(prompt).content)
    return _parse_judge_json(raw)


def _score_answer_relevancy(judge: Judge, embeddings, sample, max_nan_attempts: int = 2) -> float:
    """
    ResponseRelevancy's default strictness=3 asks the LLM for 3 separate
    completions in one batched call so it can average them for a more robust
    score. langchain-mistralai==0.2.12's _combine_llm_outputs crashes
    (TypeError: unsupported operand type(s) for +=: 'dict' and 'dict') when
    merging usage stats across more than one completion, because Mistral's
    response now includes a nested token-usage-detail dict its naive `+=`
    doesn't expect -- confirmed by direct repro against the installed
    package. That bug is specific to langchain-mistralai, so strictness is
    only dropped to 1 when Mistral is the judge; other providers keep the
    default 3, which also makes the next issue less likely to occur:

    RAGAS returns NaN (rather than 0.0) when *every* generated reverse-
    question comes back unparseable (an LLM response-parsing hiccup, not a
    property of the answer itself) -- with strictness=1 there's only one
    attempt, so a single malformed generation directly causes NaN with
    nothing to average against. Retrying the whole scoring call a couple of
    times handles the rare case where it still happens; a persistent NaN
    falls back to 0.0, the same score RAGAS itself gives an all-noncommittal
    answer.
    """
    strictness = 1 if judge.provider == "mistral" else 3
    for attempt in range(1, max_nan_attempts + 1):
        score = _call_with_backoff(
            lambda: ResponseRelevancy(llm=judge.ragas_llm, embeddings=embeddings, strictness=strictness).single_turn_score(sample)
        )
        if not (isinstance(score, float) and math.isnan(score)):
            return score
        print(f"[judge] answer_relevancy returned NaN (attempt {attempt}/{max_nan_attempts}), retrying")
    print("[judge] answer_relevancy still NaN after retries, scoring as 0.0")
    return 0.0


# ── Full item scoring ────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    rubric: RubricScore
    retrieval_correct: bool
    answer_correct: bool
    quadrant: Quadrant
    citation_reasoning: str
    answer_correct_reasoning: str
    judge_provider: str
    judge_model: str

    def as_dict(self) -> dict:
        return {
            "rubric": self.rubric.as_dict(),
            "rubric_average": self.rubric.average(),
            "retrieval_correct": self.retrieval_correct,
            "answer_correct": self.answer_correct,
            "quadrant": self.quadrant.value,
            "citation_reasoning": self.citation_reasoning,
            "answer_correct_reasoning": self.answer_correct_reasoning,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
        }


def score_item(
    judge: Judge,
    embeddings,
    *,
    question: str,
    answer: str,
    retrieved_contexts: list,
    invariants: list,
    golden_source_chunk_ids: list,
) -> JudgeResult:
    reference = "; ".join(invariants)
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    faithfulness = _call_with_backoff(lambda: Faithfulness(llm=judge.ragas_llm).single_turn_score(sample))
    context_precision = _call_with_backoff(lambda: LLMContextPrecisionWithoutReference(llm=judge.ragas_llm).single_turn_score(sample))
    context_recall = _call_with_backoff(lambda: LLMContextRecall(llm=judge.ragas_llm).single_turn_score(sample))
    answer_relevancy = _score_answer_relevancy(judge, embeddings, sample)

    custom = judge_citation_and_correctness(judge, question, invariants, retrieved_contexts, answer)

    rubric_score = RubricScore(
        faithfulness=bucket_continuous_score(faithfulness),
        context_precision=bucket_continuous_score(context_precision),
        context_recall=bucket_continuous_score(context_recall),
        answer_relevancy=bucket_continuous_score(answer_relevancy),
        citation_accuracy=custom["citation_accuracy"],
    )

    retrieval_correct = compute_retrieval_correct(retrieved_contexts, golden_source_chunk_ids)
    answer_correct = custom["answer_correct"]

    return JudgeResult(
        rubric=rubric_score,
        retrieval_correct=retrieval_correct,
        answer_correct=answer_correct,
        quadrant=classify_quadrant(retrieval_correct, answer_correct),
        citation_reasoning=custom["citation_reasoning"],
        answer_correct_reasoning=custom["answer_correct_reasoning"],
        judge_provider=judge.provider,
        judge_model=judge.model_name,
    )


if __name__ == "__main__":
    # Connectivity check only -- confirms which judge provider is active
    # without scoring anything. Run: python -m src.eval.judge
    j = get_judge()
    print(f"Judge ready: provider={j.provider} model={j.model_name}")
