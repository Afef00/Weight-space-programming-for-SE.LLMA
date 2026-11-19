# Can an activation-informed weight edit reduce repetitive Swedish generation?

**Status: implementation prepared; model run required.** This note intentionally contains no fabricated model scores.

## Motivation and hypothesis

SE.LLMA's goal of building strong Swedish language models makes generation reliability a relevant target. The paper *Sparks of Weight-Space Programming* argues that neural weights can sometimes be inspected and edited directly, and reports a large improvement on a mathematical benchmark after a small edit to a language model. Its most important methodological distinction is between (i) inferring behavior from weights without running the network and (ii) using forward-pass feedback to guide edits. These are different claims.

I test a narrow hypothesis: **an activation-informed, low-rank edit to the output projection can reduce repeated Swedish 3-grams on held-out prompts, without a material increase in repetition or loss of diversity on English prompts.**

Repetition is measurable, but it is not equivalent to poor Swedish. A fluent answer can legitimately reuse function words or topic terms. Thus repetition is only one behavioral proxy; fluency requires blinded human assessment.

## Research critique

The article's strongest evidence is its concrete intervention on Qwen3.5-0.8B: it reports that editing a small fraction of MLP parameters reduced a looping failure and increased MATH-500 accuracy. This is more informative than a qualitative claim because it reports a before/after benchmark and a proposed internal failure mechanism.

The main limitation is causal attribution. The language-model edit was informed by model behavior and iterative forward-pass feedback. It therefore supports the usefulness of feedback-guided weight editing, but does not isolate the ability to understand weights without executing the model. The reported mathematical improvement may also reflect removal of a decoding/reasoning failure that blocked an existing capability, rather than acquisition of new mathematical knowledge. A small number of models and tasks limits generalization. The post is an early research report/blog rather than a peer-reviewed benchmark study.

## Experiment

The experiment uses Qwen2.5-0.5B-Instruct, a small multilingual causal model, on CPU. GPT-SW3 126M was initially considered, but its Hugging Face repository currently restricts access to eligible European research organizations and requires institutional verification. Qwen2.5 avoids that access barrier, but it is not Swedish-specialized. Eight Swedish development prompts are used only to identify repeated 3-token sequences. For each repeated occurrence after the first, the final-layer hidden state immediately before the sequence is recorded. The first token of the repeated sequence is selected as the output token to suppress.

For each selected output row \(w_t\), the intervention is:

\[
w_t' = w_t - \alpha \cdot \frac{1}{m}\sum_{j=1}^{m}
\frac{h_j}{\|h_j\|_2^2+\epsilon}
\]

where \(h_j\) are hidden states from repeated contexts. This is a context-conditioned rank-one row edit, not gradient descent. The output projection is untied from input embeddings before editing to avoid modifying token input embeddings as a side effect.

Four conditions are compared on a held-out set of ten Swedish and five English prompts: unchanged baseline, targeted edit, random-direction edit with matched rows and edit norms, and baseline decoding with a repetition penalty. Generation is greedy and uses a fixed maximum token budget and seed. The primary metric is repeated 3-gram rate; distinct-2 and output length are secondary diagnostics. Raw generations are saved.

## Results

**To be completed after running the experiment.** The reproducible command is in `README.md`. Report the full condition table from `results/summary.md`, plus a blind 1–5 human fluency rating for every output or a balanced sample. Do not tune the edit using the held-out prompts.

If the development set contains no repeated 3-grams, the code records `no_edit_triggered`. This means the selected model and development prompts did not expose the hypothesized failure; it is not evidence that weight editing cannot work.

## Interpretation and recommendation

A positive result would require the targeted edit to lower Swedish repeated 3-gram rate relative to baseline and random edit, while maintaining output length, human-rated fluency, and English metrics. If repetition falls only because outputs become shorter or less coherent, the hypothesis is not supported. If repetition penalty performs as well as the weight edit, the more complex weight intervention has not shown an advantage.

**Provisional recommendation:** investigate the approach further only if the targeted edit beats both random editing and the inference-time baseline on held-out prompts without fluency regressions. The most informative next experiment is a multi-seed replication on a larger held-out Swedish benchmark, with blind human evaluation and a matched conventional steering/fine-tuning baseline. A separate ablation should compare weight-only diagnosis with feedback-guided edit selection.

## Reproducibility and limitations

The repository records model ID, resolved revision hash, software versions, seed, prompts, raw generations and metrics. The model weights are not redistributed. Qwen2.5 is multilingual rather than Swedish-specialized, and the prompt set is small and hand-authored; conclusions cannot be generalized to Swedish models broadly. Surface repetition metrics are not a complete language-quality measure. The intervention itself is feedback-informed and tests neither pure weight-only interpretability nor whether the model acquired new Swedish knowledge.
