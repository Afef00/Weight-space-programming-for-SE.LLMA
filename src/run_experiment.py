import argparse, json, os, random, platform
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from .metrics import summarize_token_sequences
from .editing import collect_edit_directions, apply_targeted_edit, apply_random_control

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def encode_prompt(tokenizer, prompt, device):
    # Instruct checkpoints should receive their native chat template when available.
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True
        )
        return {k: v.to(device) for k, v in encoded.items()}
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}

@torch.no_grad()
def generate_condition(model, tokenizer, records, max_new_tokens, repetition_penalty=1.0):
    model.eval()
    outputs, sequences = [], []
    device = next(model.parameters()).device
    for rec in records:
        enc = encode_prompt(tokenizer, rec["prompt"], device)
        ids = enc["input_ids"]
        mask = enc.get("attention_mask")
        result = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )[0]
        new_ids = result[ids.shape[1]:].tolist()
        outputs.append({
            "id": rec["id"], "language": rec["language"], "prompt": rec["prompt"],
            "completion": tokenizer.decode(new_ids, skip_special_tokens=True),
            "new_token_ids": new_ids
        })
        sequences.append(new_ids)
    return outputs, summarize_token_sequences(sequences)

def language_subset(records, outputs, language):
    idx = [i for i, r in enumerate(records) if r["language"] == language]
    seqs = [outputs[i]["new_token_ids"] for i in idx]
    return summarize_token_sequences(seqs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AI-Sweden-Models/gpt-sw3-126m-instruct")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--dev-file", default="data/development.json")
    ap.add_argument("--eval-file", default="data/evaluation.json")
    ap.add_argument("--output", default="results/experiment_results.json")
    ap.add_argument("--save-edited-model", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    dev, evaluation = load_json(args.dev_file), load_json(args.eval_file)

    print(f"Loading {args.model} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(args.device)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer must define eos_token_id for generation.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    revision_hash = getattr(model.config, "_commit_hash", None)

    # A. Baseline evaluation, before any development-based editing.
    baseline_out, baseline_all = generate_condition(
        model, tokenizer, evaluation, args.max_new_tokens
    )

    # B. Develop an edit using only the development prompts.
    directions, evidence = collect_edit_directions(
        model, tokenizer, dev, args.max_new_tokens
    )
    if not directions:
        print("NO EDIT GENERATED: development prompts produced no repeated 3-grams.")
        payload = {
            "status": "no_edit_triggered",
            "reason": "No repeated 3-gram found in development generations.",
            "model": args.model, "resolved_revision": revision_hash,
            "seed": args.seed, "max_new_tokens": args.max_new_tokens,
            "baseline": {"all": baseline_all,
                         "sv": language_subset(evaluation, baseline_out, "sv"),
                         "en": language_subset(evaluation, baseline_out, "en")},
            "baseline_outputs": baseline_out,
            "development_evidence": evidence,
            "note": "This is a valid negative/untestable outcome. Do not force an edit."
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # Keep an untouched state for independent conditions.
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # B. Targeted edit.
    targeted = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(args.device)
    targeted.load_state_dict(base_state, strict=True)
    targeted, edit_log = apply_targeted_edit(targeted, directions, args.alpha)
    targeted_out, targeted_all = generate_condition(
        targeted, tokenizer, evaluation, args.max_new_tokens
    )

    # C. Random edit with matched rows and L2 norms.
    random_model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(args.device)
    random_model.load_state_dict(base_state, strict=True)
    random_model = apply_random_control(random_model, directions, edit_log, args.seed)
    random_out, random_all = generate_condition(
        random_model, tokenizer, evaluation, args.max_new_tokens
    )

    # D. Inference-time baseline.
    penalty_out, penalty_all = generate_condition(
        model, tokenizer, evaluation, args.max_new_tokens,
        repetition_penalty=args.repetition_penalty
    )

    conditions = {
        "baseline": (baseline_out, baseline_all, model),
        "targeted_edit": (targeted_out, targeted_all, targeted),
        "random_edit": (random_out, random_all, random_model),
        "repetition_penalty": (penalty_out, penalty_all, model)
    }
    results = {}
    for name, (outs, all_metrics, _) in conditions.items():
        results[name] = {
            "all": all_metrics,
            "sv": language_subset(evaluation, outs, "sv"),
            "en": language_subset(evaluation, outs, "en"),
            "outputs": outs
        }

    payload = {
        "status": "completed",
        "model": args.model, "requested_revision": args.revision,
        "resolved_revision": revision_hash,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": args.device,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "alpha": args.alpha, "repetition_penalty": args.repetition_penalty,
        "development_prompts": dev, "evaluation_prompts": evaluation,
        "development_evidence": evidence, "edit_log": edit_log,
        "conditions": results
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = ["# Experiment summary", "", f"- Status: `{payload['status']}`",
               f"- Model: `{args.model}`", f"- Revision: `{revision_hash}`",
               f"- Seed: `{args.seed}`", "", "| Condition | SV repeated 3-gram ↓ | SV distinct-2 ↑ | EN repeated 3-gram ↓ | EN distinct-2 ↑ |",
               "|---|---:|---:|---:|---:|"]
    for name, obj in results.items():
        sv, en = obj["sv"], obj["en"]
        summary.append(f"| {name} | {sv['repeated_3gram_rate']:.4f} | {sv['distinct_2']:.4f} | {en['repeated_3gram_rate']:.4f} | {en['distinct_2']:.4f} |")
    Path("results/summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if args.save_edited_model:
        out_dir = Path("results/edited_model")
        targeted.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
    print(Path("results/summary.md").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
