"""
Part 1 offline evaluation (PRD Section 6).

Runs every golden dataset item through rag_pipeline/'s retriever and
generator, scores each with judge.py, and writes per-item + aggregate
results to reports/offline_eval_v1.json.

Imports the retriever/generator from the rag_pipeline/ submodule rather
than reimplementing them -- this script never duplicates or modifies that
code, and never writes into rag_pipeline/.

Run from the repo root:
    python -m src.eval.offline_eval
    python -m src.eval.offline_eval --limit 5              # quick smoke test
    python -m src.eval.offline_eval --judge-provider openai
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.scripts.step13_rag_query import load_rag_pipeline

from src.eval.judge import extract_chunk_ids, get_embeddings, get_judge, score_item
from src.eval.rubric import Dimension, Quadrant

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = REPO_ROOT / "data" / "golden_dataset" / "golden_v1.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "offline_eval_v1.json"

# Mirrors rag_pipeline's own Mistral free-tier pacing (1 req/sec) for the
# generation calls this script makes. RAGAS's own LLM calls inside judge.py
# are not paced here; ChatMistralAI/ChatOpenAI/ChatAnthropic apply their own
# default retry behavior on rate-limit errors.
RATE_LIMIT_PAUSE = 1.5  # seconds


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_item(rag_chain, judge, embeddings, item: dict) -> dict:
    response = rag_chain.invoke({"input": item["question"]})
    answer = response["answer"]
    retrieved_contexts = [doc.page_content for doc in response["context"]]

    result = score_item(
        judge,
        embeddings,
        question=item["question"],
        answer=answer,
        retrieved_contexts=retrieved_contexts,
        invariants=item["invariants"],
        golden_source_chunk_ids=item["source_chunk_ids"],
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "question": item["question"],
        "answer": answer,
        "retrieved_chunk_ids": sorted(extract_chunk_ids(retrieved_contexts)),
        "golden_source_chunk_ids": item["source_chunk_ids"],
        **result.as_dict(),
    }


def aggregate(results: list) -> dict:
    n = len(results)
    if n == 0:
        raise ValueError("cannot aggregate zero results")

    dims = [d.value for d in Dimension]
    rubric_averages = {dim: sum(r["rubric"][dim] for r in results) / n for dim in dims}
    overall_average = sum(r["rubric_average"] for r in results) / n

    quadrant_counts = {q.value: 0 for q in Quadrant}
    for r in results:
        quadrant_counts[r["quadrant"]] += 1

    category_totals: dict = {}
    for r in results:
        category_totals.setdefault(r["category"], []).append(r["rubric_average"])
    category_averages = {cat: sum(vals) / len(vals) for cat, vals in category_totals.items()}

    difficulty_totals: dict = {}
    for r in results:
        difficulty_totals.setdefault(r["difficulty"], []).append(r["rubric_average"])
    difficulty_averages = {diff: sum(vals) / len(vals) for diff, vals in difficulty_totals.items()}

    return {
        "n_items": n,
        "rubric_averages": rubric_averages,
        "overall_average": overall_average,
        "quadrant_counts": quadrant_counts,
        "category_averages": category_averages,
        "difficulty_averages": difficulty_averages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden items (smoke test)")
    parser.add_argument("--judge-provider", choices=["mistral", "openai", "anthropic"], default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    golden = load_golden_dataset()
    if args.limit:
        golden = golden[: args.limit]

    print("Loading RAG pipeline from rag_pipeline/ ...")
    rag_chain, retriever, db = load_rag_pipeline()

    judge = get_judge(preferred_provider=args.judge_provider)
    embeddings = get_embeddings()

    results = []
    last_call = 0.0
    for i, item in enumerate(golden, start=1):
        elapsed = time.time() - last_call
        if last_call and elapsed < RATE_LIMIT_PAUSE:
            time.sleep(RATE_LIMIT_PAUSE - elapsed)
        last_call = time.time()

        print(f"[{i}/{len(golden)}] {item['id']}: {item['question'][:70]}")
        record = run_item(rag_chain, judge, embeddings, item)
        results.append(record)
        print(f"    -> quadrant={record['quadrant']} rubric_avg={record['rubric_average']:.2f}")

    output = {
        "run_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "golden_dataset": str(GOLDEN_DATASET_PATH.relative_to(REPO_ROOT)),
            "n_items": len(results),
            "judge_provider": judge.provider,
            "judge_model": judge.model_name,
        },
        "items": results,
        "aggregate": aggregate(results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(results)} scored items to {args.output}")
    print(f"Overall rubric average: {output['aggregate']['overall_average']:.2f} / 3.0")
    print(f"Quadrant counts: {output['aggregate']['quadrant_counts']}")


if __name__ == "__main__":
    main()
