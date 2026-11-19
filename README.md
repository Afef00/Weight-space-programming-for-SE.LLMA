# Swedish Weight-Space Programming: a small CPU experiment

**Research question.** Can a small, activation-informed edit to a causal language model's output projection reduce repeated Swedish n-grams, while preserving Swedish and English generation.

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


## CPU setup

Python 3.10 or 3.11 

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_experiment --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --seed 17 --max-new-tokens 48
```


## Outputs

The script writes:

- `results/experiment_results.json`: metadata, prompts, generations and aggregate metrics;
- `results/summary.md`: compact condition comparison;
- `results/edited_model/`: edited model weights and tokenizer, if `--save-edited-model` is supplied.


## Metrics

- **Repeated 3-gram rate:** fraction of generated 3-gram positions whose 3-gram has appeared earlier in the same output. Lower is better, but not sufficient on its own.
- **Distinct-2:** unique bigrams divided by all bigrams. Higher is more lexical diversity, but may reward incoherence.
- **Token count:** guards against the trivial strategy of producing shorter answers.
- **Human fluency check:** manually rate Swedish and English outputs on a 1–5 scale, blind to condition. The script cannot establish grammaticality by itself.

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
