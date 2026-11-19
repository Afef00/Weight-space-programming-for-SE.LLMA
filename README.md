# Swedish Weight-Space Programming: a small CPU experiment

**Research question.** Can a small, activation-informed edit to a causal language model's output projection reduce repeated Swedish n-grams, while preserving Swedish and English generation?

This is a deliberately small, falsifiable pilot. It is **not** a reproduction of the headline result in *Sparks of Weight-Space Programming*. It tests one specific intervention and includes controls.

## Important status note

The repository contains runnable code, fixed prompt sets, evaluation logic, and a report template. **No model inference results are claimed in this package.** The execution environment used to prepare it did not have model weights or network access, so the actual model run must be performed on your computer (or Colab). The script writes raw outputs and metrics to `results/`. Do not describe the experiment as completed until those files exist.

## Method in brief

1. Load the Swedish/multilingual GPT-SW3 126M instruct checkpoint (CPU).
2. Generate from a development set and find repeated 3-token sequences.
3. For each repeated sequence, collect the final-layer hidden state immediately before the repeated sequence starts.
4. Apply a rank-one, activation-conditioned subtraction to the corresponding output-embedding row. This lowers the repeated sequence's first-token logit primarily in similar contexts.
5. Compare:
   - untouched baseline;
   - targeted output-projection edit;
   - random-direction edit with matched row/token selection and edit norm;
   - unchanged model with a repetition penalty.
6. Evaluate on a separate Swedish and English set. Record all generations and metrics.

This is a gradient-free, feedback-informed edit: the edit locations are selected using model outputs and hidden-state probes. It is **not** evidence that an LLM can infer model function from raw weights alone.

## Model and licensing

Default checkpoint: `Qwen/Qwen2.5-0.5B-Instruct` (multilingual, including Swedish; Apache-2.0 model license according to its model card).

GPT-SW3 is a multilingual decoder-only model developed by AI Sweden with Swedish, Norwegian, Danish, Icelandic, English and code in its training mix. The model card describes a modified RAIL license. Review the current model card and license before use or redistribution. This repository does not include or redistribute model weights.

Model page: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

The originally considered GPT-SW3 126M instruct checkpoint is access-restricted: Hugging Face states that access requires an eligible European university/research organization, institutional email, and agreement to research-use conditions. Qwen2.5 is used here to keep the pilot runnable without that gate. It is multilingual rather than Swedish-specialized, so conclusions must be scoped accordingly. The model identifier, resolved commit hash, library versions, and seed are recorded in the results file.

## CPU setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Download the model once (requires internet), then run:

```bash
python -m src.run_experiment \
  --model AI-Sweden-Models/gpt-sw3-126m-instruct \
  --device cpu \
  --seed 17 \
  --max-new-tokens 80
```

The first run downloads model files. CPU inference may be slow. Reduce `--max-new-tokens` to 48 for a quick smoke run. For strict offline reproduction, download the model in advance and use a local model directory.

## Outputs

The script writes:

- `results/experiment_results.json`: metadata, prompts, generations and aggregate metrics;
- `results/summary.md`: compact condition comparison;
- `results/edited_model/`: edited model weights and tokenizer, if `--save-edited-model` is supplied.

The script stops with an explicit message if the development set contains no repeated 3-grams to target. This is a meaningful negative/untestable outcome, not a reason to invent an edit. You may then report that the chosen model/prompts did not expose the hypothesized failure, and investigate a different failure mode.

## Metrics

- **Repeated 3-gram rate:** fraction of generated 3-gram positions whose 3-gram has appeared earlier in the same output. Lower is better, but not sufficient on its own.
- **Distinct-2:** unique bigrams divided by all bigrams. Higher is more lexical diversity, but may reward incoherence.
- **Token count:** guards against the trivial strategy of producing shorter answers.
- **Human fluency check:** manually rate Swedish and English outputs on a 1–5 scale, blind to condition. The script cannot establish grammaticality by itself.

The evaluation set is fixed before method development and is not used to select edit directions. Do not tune `alpha` on this evaluation set. For a stronger study, create a second held-out set and repeat with multiple seeds.

## Interpretation cautions

- Qwen2.5 is multilingual, not Swedish-specialized.
- The prompts are small and hand-authored; they are not a representative Swedish benchmark.
- Repetition metrics are surface-level proxies.
- An output projection row edit may affect many contexts and may be entangled with tied embeddings. The implementation unties the output projection before editing to avoid silently changing the input embedding table.
- The targeted edit uses generated examples and hidden-state feedback. It does not test pure weight-only interpretation.
- A random-direction control is not a substitute for fine-tuning or a carefully designed activation-steering baseline.
- A successful result on this pilot would motivate a larger test, not prove general Swedish improvement.

## Repository map

```text
src/run_experiment.py   end-to-end experiment
src/editing.py          targeted and random low-rank output-row edits
src/metrics.py          repetition and diversity metrics
data/development.json   edit-development prompts
data/evaluation.json    held-out Swedish and English prompts
configs/experiment.json default settings
tests/test_metrics.py   lightweight tests (no model download)
research_note.md        concise research note with results slots
report.pdf              PDF version of the research note
AI_USAGE.md             AI-use disclosure
```

## Reproducibility checklist

- [ ] Record Python, PyTorch and Transformers versions.
- [ ] Record model ID and resolved model commit hash.
- [ ] Keep `seed=17` for the primary run.
- [ ] Do not modify evaluation prompts after inspecting results.
- [ ] Save raw generations, not only aggregate scores.
- [ ] Rate fluency blind to condition.
- [ ] Report failed runs and negative results.
- [ ] Do not upload model weights unless their license explicitly permits it.

## Reference

Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*. Blog/research report, November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/
