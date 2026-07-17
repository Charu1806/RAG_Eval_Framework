# RAG Evaluation Framework

A companion evaluation framework for
[RAG_LangChain_Demo](https://github.com/Charu1806/RAG_LangChain_Demo) — the
AcmeTech RAG demo (LangChain + ChromaDB + Mistral).

This repo does not reimplement or modify that system. It measures it:
retrieval quality, answer quality, and failure modes, scored against a
hand-built golden dataset instead of spot-checked by eye.

Full spec: [PRD.md](PRD.md). Currently implemented: **Part 1** (golden
dataset, offline eval, judge, calibration, failure taxonomy). Part 2
(traffic simulation) and Part 3 (closing the loop) are documented in the PRD
but not yet built.

## Setup

```bash
git clone --recurse-submodules https://github.com/Charu1806/RAG_Eval_Framework.git
cd RAG_Eval_Framework

# this repo's eval dependencies
pip install -r requirements.txt

# the RAG pipeline being evaluated (separate dependency set)
pip install -r rag_pipeline/config/requirements.txt
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

The RAG pipeline requires `MISTRAL_API_KEY` to be set — see
[rag_pipeline/README.md](rag_pipeline/README.md) for how to get one.

## Structure

```
rag_pipeline/              submodule — RAG_LangChain_Demo (read-only from here)
data/golden_dataset/       hand-built Q&A golden dataset
src/eval/
  rubric.py                 5-dimension scoring rubric + 2x2 classification
  offline_eval.py           runs the golden dataset through rag_pipeline, scores it
  judge.py                  RAGAS + custom LLM-as-judge
  calibration.py            hand-label vs. judge agreement (Cohen's kappa)
  failure_taxonomy.py       bottom-up clustering of failures
reports/                    generated eval reports
```

## Results

_To be filled in once the Part 1 offline evaluation has run — see
`reports/offline_eval_v1.md`._
