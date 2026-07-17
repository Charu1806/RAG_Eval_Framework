"""
Renders reports/offline_eval_v1.md from the three JSON artifacts produced by
the rest of the pipeline, and updates the README's Results section.

This script only formats numbers that already exist in those JSON files --
it does not call any LLM or run any evaluation itself. Run the full
pipeline first, in order:

    python -m src.eval.offline_eval
    python -m src.eval.calibration template      # then hand-label reports/calibration_labels.csv
    python -m src.eval.calibration compare
    python -m src.eval.failure_taxonomy

Then:
    python -m src.eval.generate_report

Calibration and failure-taxonomy sections degrade gracefully with a "not
run yet" note if those JSON files don't exist yet -- the report can be
regenerated after each stage without waiting for the whole pipeline.
"""

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_EVAL_PATH = REPO_ROOT / "reports" / "offline_eval_v1.json"
CALIBRATION_REPORT_PATH = REPO_ROOT / "reports" / "calibration_report.json"
FAILURE_TAXONOMY_PATH = REPO_ROOT / "reports" / "failure_taxonomy.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "offline_eval_v1.md"
README_PATH = REPO_ROOT / "README.md"

_QUADRANT_LABELS = {
    "true_quality": "True quality (retrieval correct, answer correct)",
    "shipped_luck": "Shipped luck (retrieval wrong, answer correct)",
    "doom_loop": "Doom loop (retrieval correct, answer wrong)",
    "honest_failure": "Honest failure (retrieval wrong, answer wrong)",
}


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, digits=2) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)


def render_rubric_table(aggregate: dict) -> str:
    lines = ["| Dimension | Average (0-3) |", "|---|---|"]
    for dim, avg in aggregate["rubric_averages"].items():
        lines.append(f"| {dim} | {_fmt(avg)} |")
    lines.append(f"| **Overall** | **{_fmt(aggregate['overall_average'])}** |")
    return "\n".join(lines)


def render_quadrant_table(aggregate: dict) -> str:
    counts = aggregate["quadrant_counts"]
    n = sum(counts.values())
    lines = ["| Quadrant | Count | % of items |", "|---|---|---|"]
    for key, label in _QUADRANT_LABELS.items():
        c = counts.get(key, 0)
        pct = (c / n * 100) if n else 0
        lines.append(f"| {label} | {c} | {pct:.0f}% |")
    return "\n".join(lines)


def render_category_table(aggregate: dict) -> str:
    lines = ["| Category | Average rubric score (0-3) |", "|---|---|"]
    for cat, avg in sorted(aggregate["category_averages"].items()):
        lines.append(f"| {cat} | {_fmt(avg)} |")
    return "\n".join(lines)


def render_calibration_section(calibration) -> str:
    if calibration is None:
        return (
            "_Calibration has not been run yet. Run `python -m src.eval.calibration template`, "
            "hand-label `reports/calibration_labels.csv`, then `python -m src.eval.calibration compare`._"
        )

    lines = [
        f"Hand-labeled {calibration['n_labeled']} items (blind to judge scores) and compared against the judge.",
        "",
        "| Dimension | Cohen's kappa |",
        "|---|---|",
    ]
    for dim, k in calibration["kappa"].items():
        k_str = "n/a (no score variance to measure agreement against)" if k is None else f"{k:.2f}"
        lines.append(f"| {dim} | {k_str} |")

    top = calibration.get("top_disagreements", [])
    if top:
        lines.append("")
        lines.append(f"**Top {len(top)} judge/human disagreements**, ranked by total score gap:")
        lines.append("")
        for d in top:
            note = d["human_notes"] or "no note left"
            lines.append(
                f"- `{d['id']}` (gap {d['total_gap']}): human said answer_correct="
                f"{d['human_answer_correct']}, judge said {d['judge_answer_correct']} -- *{note}*"
            )
    else:
        lines.append("")
        lines.append("No disagreements -- human and judge labels matched on every hand-labeled item.")

    return "\n".join(lines)


def render_failure_taxonomy_section(taxonomy) -> str:
    if taxonomy is None:
        return "_Failure taxonomy has not been run yet. Run `python -m src.eval.failure_taxonomy`._"
    if taxonomy["n_failed"] == 0:
        return "No failed items in this run -- every golden item was scored `answer_correct=True`."

    lines = [
        f"{taxonomy['n_failed']} / {taxonomy['n_total_items']} items failed "
        f"({taxonomy['failure_rate']:.0%}), bottom-up clustered into {len(taxonomy['clusters'])} group(s):",
        "",
        "| Rank | Failure pattern | Count | Example items |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(taxonomy["clusters"], start=1):
        examples = ", ".join(c["item_ids"][:4])
        lines.append(f"| {i} | {c['label']} | {c['size']} | {examples} |")
    return "\n".join(lines)


def render_report(offline_eval: dict, calibration, taxonomy) -> str:
    meta = offline_eval["run_metadata"]
    aggregate = offline_eval["aggregate"]

    parts = [
        "# Offline Evaluation Report -- v1",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d')} | Run dated {meta.get('generated_at', 'unknown')}",
        "",
        f"Golden dataset: `{meta.get('golden_dataset', 'unknown')}` ({meta['n_items']} items). "
        f"Judge: {meta.get('judge_provider', 'unknown')} ({meta.get('judge_model', 'unknown')}).",
        "",
        "## Rubric scores",
        "",
        render_rubric_table(aggregate),
        "",
        "## Retrieval x answer correctness (2x2)",
        "",
        render_quadrant_table(aggregate),
        "",
        "## Scores by category",
        "",
        render_category_table(aggregate),
        "",
        "## Judge calibration",
        "",
        render_calibration_section(calibration),
        "",
        "## Failure taxonomy",
        "",
        render_failure_taxonomy_section(taxonomy),
        "",
    ]
    return "\n".join(parts)


def render_readme_summary(offline_eval: dict) -> str:
    aggregate = offline_eval["aggregate"]
    meta = offline_eval["run_metadata"]
    counts = aggregate["quadrant_counts"]
    run_date = meta.get("generated_at", "unknown")[:10]
    return (
        f"Part 1 offline evaluation run on {run_date} "
        f"({meta['n_items']} golden items, judge: {meta.get('judge_provider')}/{meta.get('judge_model')}): "
        f"overall rubric average **{aggregate['overall_average']:.2f} / 3.0**. "
        f"Quadrant split -- true_quality: {counts.get('true_quality', 0)}, "
        f"shipped_luck: {counts.get('shipped_luck', 0)}, "
        f"doom_loop: {counts.get('doom_loop', 0)}, "
        f"honest_failure: {counts.get('honest_failure', 0)}. "
        f"Full breakdown: [reports/offline_eval_v1.md](reports/offline_eval_v1.md)."
    )


def update_readme_results(offline_eval: dict, readme_path: Path = README_PATH) -> None:
    summary = render_readme_summary(offline_eval)
    text = readme_path.read_text(encoding="utf-8")
    marker = "## Results"
    idx = text.index(marker)
    if idx == -1:
        raise ValueError(f"'{marker}' section not found in {readme_path}")
    before = text[: idx + len(marker)]
    new_text = f"{before}\n\n{summary}\n"
    readme_path.write_text(new_text, encoding="utf-8")


def main():
    offline_eval = _load(OFFLINE_EVAL_PATH)
    if offline_eval is None:
        raise SystemExit(f"{OFFLINE_EVAL_PATH} not found. Run `python -m src.eval.offline_eval` first.")

    calibration = _load(CALIBRATION_REPORT_PATH)
    taxonomy = _load(FAILURE_TAXONOMY_PATH)

    report = render_report(offline_eval, calibration, taxonomy)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")

    update_readme_results(offline_eval)
    print(f"Updated {README_PATH} Results section")


if __name__ == "__main__":
    main()
