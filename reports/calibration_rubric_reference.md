# Calibration Rubric Reference

Use these exact level definitions when hand-labeling -- they're the same ones given to the judge.

## faithfulness
- **0**: Answer contradicts the retrieved context, or states claims with no support in it.
- **1**: Answer is mostly unsupported: at least one major claim has no basis in the retrieved context.
- **2**: Answer is mostly grounded, but includes a minor unsupported detail or an inference that goes slightly beyond the context.
- **3**: Every claim in the answer is directly supported by the retrieved context.

## context_precision
- **0**: None of the retrieved chunks are relevant to the question.
- **1**: A minority of retrieved chunks are relevant; most are noise.
- **2**: Most retrieved chunks are relevant, with one or two irrelevant chunks mixed in.
- **3**: All (or nearly all) retrieved chunks are relevant to the question.

## context_recall
- **0**: None of the golden invariants' source chunks were retrieved.
- **1**: A minority of the golden invariants' source chunks were retrieved -- key information is missing.
- **2**: Most of the golden invariants' source chunks were retrieved; a minor supporting chunk was missed.
- **3**: All of the golden invariants' source chunks were retrieved.

## answer_relevancy
- **0**: Answer does not address the question asked.
- **1**: Answer is tangentially related but misses the core of the question.
- **2**: Answer addresses the question but includes irrelevant padding or hedges more than necessary.
- **3**: Answer directly and completely addresses the question asked, with no irrelevant content.

## citation_accuracy
- **0**: No sources cited, or cited sources do not correspond to any retrieved chunk.
- **1**: Sources are cited but are frequently wrong or attribute claims to chunks that don't support them.
- **2**: Sources are mostly correct, with one citation misattributed or a claim left uncited.
- **3**: Every claim requiring a citation is correctly attributed to the retrieved chunk that supports it.

## retrieval_correct
- true if every chunk needed to answer the question (the golden item's `source_chunk_ids`) was actually retrieved; false otherwise.

## answer_correct
- true only if the answer covers ALL of the golden item's required facts (invariants) in substance -- paraphrasing is fine, a missing or contradicted fact makes it false.