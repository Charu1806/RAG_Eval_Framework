# PRD: RAG Evaluation Framework — AcmeTech Demo

Author: Charu Gupta | Status: Draft | Last updated: July 2026

## 1. Summary

Build a structured evaluation layer on top of the existing AcmeTech RAG system
(LangChain + ChromaDB + Mistral, github.com/Charu1806/RAG_LangChain_Demo) that
makes retrieval and answer quality measurable, diagnosable, and improvable —
rather than judged by spot-checking a few outputs.

This is a personal/teaching project, not a production system. Scope is
deliberately bounded to what can be built and demoed within a few weeks, using
free-tier tools (Colab, local machine, open-source eval libraries).

## 2. Problem

The original RAG project proved the system works, but "works" was judged
informally — a handful of example Q&As, eyeballed. There's no way to answer:
which questions does it get wrong, why, and did a given change (embedding
model swap, retrieval tweak) actually make things better or just move the
failures around.

This mirrors a real gap in most public RAG tutorials and demo projects: they
show retrieval working, but skip the harder and more valuable question of how
you know when it's not.

## 3. Goals

- Build a golden dataset that captures correctness as invariants (required
  facts), not fixed answer strings — since RAG answers can be correctly
  phrased many ways.
- Score answers on a decomposed rubric (faithfulness, context precision/recall,
  answer relevancy, citation accuracy) instead of a single pass/fail judgment.
- Distinguish "right answer for the wrong reason" (shipped luck) from "right
  answer via correct retrieval" (true quality) — the single most important
  idea borrowed from the source framework.
- Calibrate the automated judge against my own hand-labels before trusting it
  at scale.
- Build a failure taxonomy bottom-up from real failures, not a predefined
  list.
- Test one concrete improvement (hybrid search) against the golden dataset
  with a real before/after delta, not a vibe.
- Close the loop once: take the biggest failure category, make one deliberate
  fix, re-measure.
- Produce a class-ready live demo and a set of graphical reports reusable in
  a Medium follow-up article.

## 4. Non-goals

- No production infrastructure — no ship gates, no real A/B testing, no
  persistent online judges running 24/7.
- No fine-tuning of the LLM itself (out of scope for RAG; noted only as a
  possible future stretch goal on the embedding model, not required here).
- No automated closed-loop retraining — the flywheel is closed manually,
  once, deliberately (see Part 3).
- No public, always-on hosted endpoint — live demo is a temporary Gradio
  share link during class, not a persistent service.

## 5. Users

| User | Need |
|---|---|
| Charu (builder/instructor) | Understand system quality well enough to teach it and speak to it in interviews |
| Students (class demo) | See retrieval and evaluation happen live, including real failures they cause |
| Medium readers | A concrete, honest account of building and evaluating a RAG system, including what didn't improve |
| Interviewers | Evidence of hands-on judgment about RAG quality, not just tool usage |

## 6. Scope — three parts

### Part 1 — Golden dataset, offline eval, failure taxonomy

- Environment: Google Colab.
- Golden dataset of 30–50 Q&A pairs across all 6 AcmeTech categories (HR,
  Finance, Engineering, Support, Product, Employee), schema below.
- Rubric scoring (5 dimensions, 0–3 scale) via RAGAS for standard metrics +
  a custom LLM-as-judge prompt for citation accuracy and the
  retrieval-correct/answer-correct classification.
- Calibration pass: hand-label 15–20 items myself, compare against judge
  scores, report Cohen's kappa agreement.
- Failure taxonomy: cluster failed items bottom-up, rank by frequency.
- Output: `reports/offline_eval_v1.json` + `reports/offline_eval_v1.md`

### Part 2 — Online / traffic simulation

- Environment: local machine (not Colab — avoids idle disconnects over a
  long run).
- A script generates/paraphrases questions and queries the RAG system
  continuously for a fixed duration (~1 hour), logging question, retrieved
  chunks, and answer for each call.
- Each answer is scored live using the same rubric/judge from Part 1, using
  the same vector store and embedding model as the Colab run (must match
  exactly for valid comparison).
- Output: a running average score (drift line) across the hour, plus its
  own failure taxonomy.
- Output: `reports/online_eval_run.json`
- Results are pushed to the shared GitHub repo so Colab can read them
  without re-running anything.

### Part 3 — Close the loop, once

- From the combined Part 1 + Part 2 failure taxonomy, pick the single
  largest failure category.
- Make one deliberate, human-decided fix (e.g., increase retrieval k, add a
  reranker, or introduce hybrid search — candidate solutions evaluated as
  experiments, not assumptions).
- Re-run the golden dataset against the modified pipeline; report the
  before/after delta per rubric dimension and per failure category — not
  just a total score, since totals can hide redistribution (per the "model
  upgrades redistribute failure mass" finding from the source framework).
- Add the worst remaining failure cases as new golden dataset rows
  (`golden_v2.json`) — the flywheel closes here, manually, once.

> **Build status: only Part 1 is in scope for the current build.** Parts 2
> and 3 are documented here for context but not implemented yet.

## 7. Data — golden dataset schema

| Field | Description |
|---|---|
| `id` | Unique identifier (e.g. G001) |
| `category` | HR / Finance / Engineering / Support / Product / Employee |
| `question` | Natural-language query |
| `invariants` | List of facts/claims that must appear in a correct answer |
| `difficulty` | easy / medium / hard |
| `source_chunk_ids` | Chunk id(s) in the vector store containing the answer |
| `notes` | Anything ambiguous, or why an item is hard |

Hard items should require synthesizing across 2+ chunks — these are what
expose context-recall failures.

## 8. Environments and handoff

| Stage | Environment | Why |
|---|---|---|
| Offline eval (Part 1) | Colab | Notebook format matches teaching style; short-duration runs, safe for idle limits |
| Traffic simulation (Part 2) | Local machine | Long-running (~1 hr); avoids Colab idle disconnects; full control to pause/resume |
| Sync | GitHub repo (reports/*.json) | Single source of truth for results; versioned like the code |
| Visualization + demo | Colab | Loads pre-computed JSON results and renders charts — never re-runs a slow eval live |

**Critical constraint:** the local traffic run must use the identical vector
store and embedding model as the Colab offline run — any drift between
environments invalidates the before/after comparison.

## 9. Class demo plan

- Pre-class: Parts 1 and 2 already run; results committed to GitHub.
- In Colab (live): load pre-computed results, render the four report charts
  (below).
- Live interactive segment: "break my RAG bot" — Gradio share link, students
  submit real adversarial/ambiguous questions, logged and scored live.
- Compare live class failures against the pre-built failure taxonomy —
  synthetic vs. real human-generated failure modes.
- Post-class: fold any new failure modes surfaced by students into
  `golden_v2.json`.

## 10. Reports and visualizations

| Report | Chart type | Source data |
|---|---|---|
| Offline eval | 2×2 quadrant scatter (shipped luck / true quality / doom loop / honest failure) | Part 1 |
| Offline eval | Bar chart, average score per rubric dimension | Part 1 |
| Failure taxonomy | Ranked bar chart of failure categories | Part 1 + Part 2 combined |
| Online eval | Drift line — running average score across the hour | Part 2 |
| Baseline vs. experiment | Grouped bar chart, per failure category (not just total score) | Part 3 |

## 11. Success criteria

- Golden dataset, offline eval, and failure taxonomy fully built and
  reproducible in Colab.
- Judge calibration completed with a reported kappa score and at least one
  flagged disagreement case.
- Online traffic run completed for the full target duration with results
  synced back to Colab.
- One experiment run with a reported, non-hand-wavy delta (even if the
  delta is small or null — that's still a valid, reportable result).
- Live class demo executed without needing to re-run a slow job on stage.
- Article and/or interview-ready artifacts: four chart types, one flagged
  "shipped luck" example, one calibration disagreement example.

## 12. Risks / open questions

- Judge cost/latency at scale — mitigate by keeping the golden dataset
  small (30–50 items) and considering a local judge via Ollama for the
  1-hour traffic run.
- Embedding/version drift between Colab and local environments — mitigate
  by pinning package versions and re-using the same persisted vector store
  rather than rebuilding it in each environment.
- Class demo network dependency (Gradio share link, live traffic) — have
  the pre-computed results as a fallback if live demo has issues.
- Open question: how many students' worth of live traffic is realistic to
  expect, and is that enough to meaningfully expand the golden dataset
  afterward.

## 13. Rollout plan

| Step | Deliverable |
|---|---|
| 1 | Golden dataset v1 (Colab) |
| 2 | Offline eval + failure taxonomy v1 (Colab) |
| 3 | Judge calibration (Colab) |
| 4 | Traffic simulator + 1-hr run (local) |
| 5 | Sync results to GitHub, load + visualize in Colab |
| 6 | One experiment (hybrid search) vs. baseline (Colab) |
| 7 | Close the loop: one fix, re-measure, golden_v2.json |
| 8 | Class demo (live Gradio + pre-built reports) |
| 9 | Article write-up + interview-ready summary |

## 14. Gap to production grade

This project deliberately builds the top half of the evaluation loop
(golden dataset → offline evals → failure taxonomy) and simulates the
bottom half rather than building it for real. Below is what each simulated
piece would actually require at production scale.

| Loop stage | What this project does | What production requires |
|---|---|---|
| Ship gate | Not built — experiments are judged manually, once | Automated pass/fail threshold blocking deploy in CI; attested quality thresholds tied to release process, not a person's judgment call |
| Gradual rollout | Not built — the fix from Part 3 applies to the whole system at once | Phased exposure (5% to 25% to 100%), real A/B cohorts, statistical significance testing before full rollout, instant rollback path |
| Online judges | Simulated via a 1-hour scripted traffic run, scored after the fact | Live scoring on every real production call, continuously, with judge latency low enough not to slow the response |
| Metrics ladder | Not built — no real users, so no product/business metrics exist to climb to | Agent metrics must be shown to predict product metrics (retention, session completion), which must predict business metrics (revenue, churn) — each layer validated, not assumed |
| Drift + review | One offline snapshot; no ongoing monitoring | Scheduled (e.g. weekly) review of aggregated online scores, with alarms when judge-human agreement or quality drops below threshold |
| The flywheel | Closed once, manually, by me choosing what to fix | Continuous, semi-automated: failures are triaged, attributed, and converted into new golden data on a regular cadence, feeding retraining or reconfiguration without a person doing it by hand each time |

### Other production concerns out of scope here

- Data governance and privacy — synthetic data has no PII; production RAG
  over real company/customer data needs access controls on what gets
  retrieved for whom, redaction, and audit trails.
- Latency and cost at scale — judge calls, reranking, and retrieval all add
  latency; production needs SLAs and cost budgets per request, not just per
  eval run.
- Judge governance — who owns recalibrating the judge when it drifts, and
  how often, is an organizational process, not just a script.
- Versioning and rollback — golden datasets, prompts, and retrieval configs
  all need version control and the ability to roll back a bad change
  quickly, not just re-run a notebook.
- On-call / incident response — a doom-looping agent in production needs
  someone to page, not just a chart to look at later.
- Multi-tenant safety — with real users, adversarial or malicious queries
  need guardrails this demo doesn't need against a fixed synthetic
  knowledge base.

The honest framing for the article and interview: this project builds real
evaluation rigor at demo scale, and simulates rather than builds the
operational infrastructure (ship gates, live monitoring, automated
flywheel) that separates a personal project from a production eval system.
That gap is worth naming explicitly rather than glossing over, since
knowing exactly what's missing is more credible than implying the demo is
production-ready.
