"""
Judge calibration workflow (PRD Section 6).

Two-step process:

  1. `python -m src.eval.calibration template`
     Samples a stratified subset of golden items (default 18, ~3 per
     category) from an existing offline_eval_v1.json run and writes a BLIND
     hand-labeling CSV -- it includes the question, the generated answer,
     and the actual retrieved context, but deliberately NOT the judge's
     scores, so you label without anchoring on what the judge said. Also
     writes a rubric cheat-sheet (calibration_rubric_reference.md) using the
     exact level definitions the judge was given.

     Fill in the human_* columns by hand, save as reports/calibration_labels.csv.

  2. `python -m src.eval.calibration compare`
     Joins your hand labels back against the judge's scores for the same
     items and reports Cohen's kappa per rubric dimension (linear-weighted,
     since the 0-3 scale is ordinal) and for the two boolean classification
     fields (unweighted). Surfaces the items where you and the judge
     disagreed most -- ranked by total score gap across all 5 rubric
     dimensions, with a boolean-field flip counted as a max-size gap (3),
     since disagreeing on correctness is a bigger deal than a one-point
     rubric wobble. This is a proxy for "most confident disagreement" --
     the judge doesn't currently emit a separate confidence score.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from src.eval.offline_eval import load_golden_dataset
from src.eval.rubric import RUBRIC, Dimension

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFLINE_EVAL = REPO_ROOT / "reports" / "offline_eval_v1.json"
DEFAULT_TEMPLATE_OUT = REPO_ROOT / "reports" / "calibration_template.csv"
DEFAULT_RUBRIC_REFERENCE_OUT = REPO_ROOT / "reports" / "calibration_rubric_reference.md"
DEFAULT_LABELS_IN = REPO_ROOT / "reports" / "calibration_labels.csv"
DEFAULT_REPORT_OUT = REPO_ROOT / "reports" / "calibration_report.json"

RUBRIC_FIELDS = [d.value for d in Dimension]
BOOLEAN_FIELDS = ["retrieval_correct", "answer_correct"]
TEMPLATE_FIELDS = ["id", "category", "difficulty", "question", "answer", "retrieved_contexts", "golden_invariants"]
HUMAN_FIELDS = [f"human_{f}" for f in RUBRIC_FIELDS + BOOLEAN_FIELDS] + ["human_notes"]


def _load_offline_eval(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def write_rubric_reference(path: Path = DEFAULT_RUBRIC_REFERENCE_OUT) -> None:
    lines = [
        "# Calibration Rubric Reference",
        "",
        "Use these exact level definitions when hand-labeling -- they're the same ones given to the judge.",
        "",
    ]
    for dim in Dimension:
        lines.append(f"## {dim.value}")
        for level, desc in RUBRIC[dim].items():
            lines.append(f"- **{level}**: {desc}")
        lines.append("")
    lines.append("## retrieval_correct")
    lines.append(
        "- true if every chunk needed to answer the question (the golden item's "
        "`source_chunk_ids`) was actually retrieved; false otherwise."
    )
    lines.append("")
    lines.append("## answer_correct")
    lines.append(
        "- true only if the answer covers ALL of the golden item's required facts "
        "(invariants) in substance -- paraphrasing is fine, a missing or "
        "contradicted fact makes it false."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def sample_items_for_calibration(offline_eval: dict, n: int = 18, seed: int = 42) -> list:
    """Stratified sample, ~n / num_categories per category, so calibration
    isn't skewed toward whichever category happened to be scored first."""
    items = offline_eval["items"]
    by_category: dict = {}
    for item in items:
        by_category.setdefault(item["category"], []).append(item)

    rng = random.Random(seed)
    categories = sorted(by_category)
    per_category = max(1, n // len(categories))

    sampled = []
    for cat in categories:
        pool = by_category[cat][:]
        rng.shuffle(pool)
        sampled.extend(pool[:per_category])

    rng.shuffle(sampled)
    return sampled[:n]


def generate_template(
    offline_eval_path: Path = DEFAULT_OFFLINE_EVAL,
    output_csv: Path = DEFAULT_TEMPLATE_OUT,
    n: int = 18,
    seed: int = 42,
) -> Path:
    offline_eval = _load_offline_eval(offline_eval_path)
    golden = load_golden_dataset()
    invariants_by_id = {g["id"]: g["invariants"] for g in golden}

    sampled = sample_items_for_calibration(offline_eval, n=n, seed=seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEMPLATE_FIELDS + HUMAN_FIELDS)
        writer.writeheader()
        for item in sampled:
            writer.writerow(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "difficulty": item["difficulty"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "retrieved_contexts": " ||| ".join(item["retrieved_contexts"]),
                    "golden_invariants": "; ".join(invariants_by_id.get(item["id"], [])),
                    **{f"human_{f}": "" for f in RUBRIC_FIELDS + BOOLEAN_FIELDS},
                    "human_notes": "",
                }
            )

    write_rubric_reference()
    return output_csv


def _kappa(human_vals: list, judge_vals: list, weights=None):
    if len(set(human_vals)) == 1 and len(set(judge_vals)) == 1 and human_vals[0] == judge_vals[0]:
        return 1.0  # trivial perfect agreement -- sklearn returns NaN here (zero expected variance)
    kappa = cohen_kappa_score(human_vals, judge_vals, weights=weights)
    return None if isinstance(kappa, float) and math.isnan(kappa) else kappa


def compute_calibration(
    labels_csv: Path = DEFAULT_LABELS_IN,
    offline_eval_path: Path = DEFAULT_OFFLINE_EVAL,
    top_n_disagreements: int = 5,
) -> dict:
    offline_eval = _load_offline_eval(offline_eval_path)
    judge_by_id = {item["id"]: item for item in offline_eval["items"]}

    with open(labels_csv, encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))

    incomplete = [
        row["id"]
        for row in human_rows
        if not all(str(row.get(f"human_{f}", "")).strip() for f in RUBRIC_FIELDS + BOOLEAN_FIELDS)
    ]
    if incomplete:
        raise ValueError(f"These items have incomplete hand labels, fill them in before comparing: {incomplete}")

    human_by_dim = {d: [] for d in RUBRIC_FIELDS}
    judge_by_dim = {d: [] for d in RUBRIC_FIELDS}
    human_by_bool = {b: [] for b in BOOLEAN_FIELDS}
    judge_by_bool = {b: [] for b in BOOLEAN_FIELDS}
    disagreements = []

    for row in human_rows:
        item_id = row["id"]
        judge_item = judge_by_id.get(item_id)
        if judge_item is None:
            raise ValueError(f"{item_id} from hand-labels not found in {offline_eval_path}")

        gaps = {}
        total_gap = 0
        for dim in RUBRIC_FIELDS:
            h = int(row[f"human_{dim}"])
            j = judge_item["rubric"][dim]
            human_by_dim[dim].append(h)
            judge_by_dim[dim].append(j)
            gap = abs(h - j)
            gaps[dim] = gap
            total_gap += gap

        for b in BOOLEAN_FIELDS:
            h = _parse_bool(row[f"human_{b}"])
            j = bool(judge_item[b])
            human_by_bool[b].append(int(h))
            judge_by_bool[b].append(int(j))
            if h != j:
                total_gap += 3  # a correctness flip counts as much as a max rubric gap

        disagreements.append(
            {
                "id": item_id,
                "question": judge_item["question"],
                "total_gap": total_gap,
                "per_dimension_gap": gaps,
                "human_answer_correct": _parse_bool(row["human_answer_correct"]),
                "judge_answer_correct": judge_item["answer_correct"],
                "judge_answer_correct_reasoning": judge_item["answer_correct_reasoning"],
                "human_notes": row.get("human_notes", ""),
            }
        )

    kappa = {dim: _kappa(human_by_dim[dim], judge_by_dim[dim], weights="linear") for dim in RUBRIC_FIELDS}
    kappa.update({b: _kappa(human_by_bool[b], judge_by_bool[b]) for b in BOOLEAN_FIELDS})

    disagreements.sort(key=lambda d: d["total_gap"], reverse=True)
    top_disagreements = [d for d in disagreements if d["total_gap"] > 0][:top_n_disagreements]

    return {
        "n_labeled": len(human_rows),
        "kappa": kappa,
        "top_disagreements": top_disagreements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    tmpl = sub.add_parser("template", help="Generate a blind hand-labeling CSV template")
    tmpl.add_argument("--n", type=int, default=18)
    tmpl.add_argument("--seed", type=int, default=42)
    tmpl.add_argument("--offline-eval", type=Path, default=DEFAULT_OFFLINE_EVAL)
    tmpl.add_argument("--output", type=Path, default=DEFAULT_TEMPLATE_OUT)

    comp = sub.add_parser("compare", help="Compare filled-in hand labels against judge scores")
    comp.add_argument("--labels", type=Path, default=DEFAULT_LABELS_IN)
    comp.add_argument("--offline-eval", type=Path, default=DEFAULT_OFFLINE_EVAL)
    comp.add_argument("--top-n", type=int, default=5)
    comp.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUT)

    args = parser.parse_args()

    if args.command == "template":
        path = generate_template(args.offline_eval, args.output, n=args.n, seed=args.seed)
        print(f"Wrote {args.n}-item blind hand-labeling template to {path}")
        print(f"Wrote rubric reference to {DEFAULT_RUBRIC_REFERENCE_OUT}")
        print("Fill in the human_* columns, save as reports/calibration_labels.csv, then run:")
        print("  python -m src.eval.calibration compare")
    elif args.command == "compare":
        result = compute_calibration(args.labels, args.offline_eval, top_n_disagreements=args.top_n)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Labeled items: {result['n_labeled']}")
        print("Cohen's kappa by dimension:")
        for dim, k in result["kappa"].items():
            print(f"  {dim}: {k}")
        print(f"\nTop {len(result['top_disagreements'])} disagreements written to {args.output}")


if __name__ == "__main__":
    main()
