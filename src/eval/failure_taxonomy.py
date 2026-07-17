"""
Bottom-up failure taxonomy (PRD Section 6).

Clusters failed golden items (judge scored answer_correct=False, i.e.
quadrant in {doom_loop, honest_failure}) by the *content* of why they
failed -- not by a predefined category list. TF-IDF over each item's
question + the judge's answer_correct_reasoning feeds an agglomerative
clustering pass with no fixed number of clusters (a cosine distance
threshold decides how many groups emerge), so the taxonomy reflects
whatever patterns actually show up in this run's failures rather than
categories assumed in advance. Clusters are ranked by frequency (size).

Semantic embeddings (the same all-MiniLM-L6-v2 model the RAG pipeline
uses) were tried first and dropped: on short judge-reasoning sentences they
gave a noisier signal than TF-IDF -- on a hand-checked synthetic test set
with two clear failure themes, TF-IDF's exact-phrase overlap (e.g. both
sentences containing "required" and "missing") cleanly separated the
themes at distance_threshold=0.85, while embedding cosine distances didn't
separate them at any threshold tested. TF-IDF is simpler, needs no model
download, and empirically worked better here, so it's both the clustering
signal and the source of each cluster's descriptive top terms.

Each cluster gets an automatic label: if a judge LLM is available, it's
asked to name the common failure pattern from a sample of the cluster's
reasoning text; otherwise the top TF-IDF terms are used as a fallback label,
so the tool still works with no API access (--no-llm-labels).

Run from the repo root:
    python -m src.eval.failure_taxonomy
    python -m src.eval.failure_taxonomy --distance-threshold 0.8
    python -m src.eval.failure_taxonomy --no-llm-labels
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFLINE_EVAL = REPO_ROOT / "reports" / "offline_eval_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "failure_taxonomy.json"

_LABEL_PROMPT = """These are the reasons a RAG system's answer was judged incorrect, for several different questions. In 3-6 words, name the common failure pattern they share. Respond with ONLY the short label -- no punctuation, no explanation, no quotes.

{reasons}
"""


def load_offline_eval(path: Path = DEFAULT_OFFLINE_EVAL) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_failed_items(offline_eval: dict) -> list:
    """A 'failure' is any item the judge scored answer_correct=False. shipped_luck
    items got the right answer this time (via the wrong retrieval) so they
    aren't counted as failures here, even though they're a latent risk worth
    noting separately (see the quadrant_counts in offline_eval_v1.json)."""
    return [item for item in offline_eval["items"] if not item["answer_correct"]]


def _failure_text(item: dict) -> str:
    return f"{item['question']} {item.get('answer_correct_reasoning', '')}"


def top_terms(vectorizer: TfidfVectorizer, X, indices: list, top_n: int = 6) -> list:
    sub = X[indices]
    mean_tfidf = np.asarray(sub.mean(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    top_idx = mean_tfidf.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_idx if mean_tfidf[i] > 0]


def llm_label_cluster(judge, reasonings: list, max_examples: int = 6) -> str:
    sample = [r for r in reasonings[:max_examples] if r]
    reasons_block = "\n".join(f"- {r}" for r in sample)
    prompt = _LABEL_PROMPT.format(reasons=reasons_block)
    raw = judge.llm.invoke(prompt).content.strip()
    return raw.splitlines()[0].strip(" .\"'")


def build_taxonomy(
    offline_eval: dict,
    distance_threshold: float = 0.85,
    use_llm_labels: bool = True,
    judge=None,
) -> dict:
    failed_items = get_failed_items(offline_eval)
    n_failed = len(failed_items)
    n_total = len(offline_eval["items"])

    if n_failed == 0:
        return {"n_total_items": n_total, "n_failed": 0, "failure_rate": 0.0, "clusters": []}

    texts = [_failure_text(item) for item in failed_items]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X = vectorizer.fit_transform(texts)

    if n_failed == 1:
        cluster_indices = [[0]]
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X.toarray())
        grouped: dict = {}
        for idx, lbl in enumerate(labels):
            grouped.setdefault(int(lbl), []).append(idx)
        cluster_indices = list(grouped.values())

    clusters = []
    for indices in cluster_indices:
        items = [failed_items[i] for i in indices]
        reasonings = [it.get("answer_correct_reasoning", "") for it in items]
        terms = top_terms(vectorizer, X, indices)

        label = None
        if use_llm_labels and judge is not None:
            try:
                label = llm_label_cluster(judge, reasonings)
            except Exception as e:  # noqa: BLE001 -- labeling is best-effort, never fatal
                print(f"[failure_taxonomy] LLM labeling failed for a cluster ({e}); falling back to top terms")
        if not label:
            label = ", ".join(terms) if terms else "(no distinguishing terms)"

        clusters.append(
            {
                "label": label,
                "size": len(items),
                "item_ids": [it["id"] for it in items],
                "categories": dict(Counter(it["category"] for it in items)),
                "top_terms": terms,
                "example_reasoning": reasonings[0] if reasonings else "",
            }
        )

    clusters.sort(key=lambda c: c["size"], reverse=True)

    return {
        "n_total_items": n_total,
        "n_failed": n_failed,
        "failure_rate": n_failed / n_total,
        "distance_threshold": distance_threshold,
        "clusters": clusters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline-eval", type=Path, default=DEFAULT_OFFLINE_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.85,
        help="Cosine distance threshold (TF-IDF space) for merging failures into the same cluster; lower = stricter, more/smaller clusters",
    )
    parser.add_argument("--no-llm-labels", action="store_true", help="Skip LLM cluster labeling; use top TF-IDF terms only (no API calls)")
    parser.add_argument("--judge-provider", choices=["mistral", "openai", "anthropic"], default=None)
    args = parser.parse_args()

    offline_eval = load_offline_eval(args.offline_eval)

    judge = None
    if not args.no_llm_labels:
        from src.eval.judge import get_judge

        try:
            judge = get_judge(preferred_provider=args.judge_provider)
        except Exception as e:  # noqa: BLE001
            print(f"[failure_taxonomy] no judge provider available ({e}); using top-TF-IDF-term labels only")

    taxonomy = build_taxonomy(
        offline_eval,
        distance_threshold=args.distance_threshold,
        use_llm_labels=not args.no_llm_labels,
        judge=judge,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"{taxonomy['n_failed']} / {taxonomy['n_total_items']} items failed ({taxonomy['failure_rate']:.0%})")
    print(f"{len(taxonomy['clusters'])} failure clusters, ranked by frequency:")
    for c in taxonomy["clusters"]:
        print(f"  [{c['size']}] {c['label']}  (e.g. {', '.join(c['item_ids'][:3])})")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
