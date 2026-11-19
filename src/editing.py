import copy
import torch
from collections import Counter, defaultdict
from .metrics import ngram_repetition_rate

def repeated_ngram_starts(token_ids, n=3, min_occurrences=2):
    """Return token-start positions for n-grams occurring at least min_occurrences."""
    counts = Counter(tuple(token_ids[i:i+n]) for i in range(max(0, len(token_ids)-n+1)))
    positions = defaultdict(list)
    for i in range(max(0, len(token_ids)-n+1)):
        gram = tuple(token_ids[i:i+n])
        if counts[gram] >= min_occurrences:
            positions[gram].append(i)
    return {g: ps for g, ps in positions.items() if len(ps) >= min_occurrences}

def clone_output_head(model):
    """Untie output projection from input embeddings before editing."""
    old = model.get_output_embeddings()
    if old is None or not hasattr(old, "weight"):
        raise RuntimeError("Model has no accessible output embedding projection.")
    new = copy.deepcopy(old)
    new.weight = torch.nn.Parameter(old.weight.detach().clone())
    model.set_output_embeddings(new)
    return new

@torch.no_grad()
def collect_edit_directions(model, tokenizer, dev_records, max_new_tokens=80,
                             ngram_size=3, min_occurrences=2):
    """
    Generate development outputs; for every repeated n-gram occurrence after
    its first occurrence, capture the final hidden state immediately before
    the repeated n-gram starts. Target the first token of that n-gram.
    """
    model.eval()
    device = next(model.parameters()).device
    directions = defaultdict(list)
    evidence = []
    for rec in dev_records:
        if getattr(tokenizer, "chat_template", None):
            enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}
        else:
            enc = {k: v.to(device) for k, v in tokenizer(rec["prompt"], return_tensors="pt").items()}
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )[0]
        prompt_len = input_ids.shape[1]
        new_ids = generated[prompt_len:].tolist()
        repeated = repeated_ngram_starts(new_ids, ngram_size, min_occurrences)
        for gram, positions in repeated.items():
            # The first occurrence is the reference; edit later recurrences.
            for pos in positions[1:]:
                full_prefix = generated[:prompt_len + pos]
                if full_prefix.numel() < 1:
                    continue
                out = model(input_ids=full_prefix.unsqueeze(0), output_hidden_states=True)
                h = out.hidden_states[-1][0, -1].detach().float().cpu()
                target_token = int(new_ids[pos])
                directions[target_token].append(h)
                evidence.append({
                    "record_id": rec["id"], "ngram_token_ids": list(gram),
                    "repeat_position": int(pos), "target_token_id": target_token
                })
    return directions, evidence

@torch.no_grad()
def apply_targeted_edit(model, directions, alpha=1.5):
    """
    For each targeted output row w_t, apply:
      w_t <- w_t - alpha * mean(h / ||h||^2)
    This reduces token t's logit by approximately alpha in each matching
    activation direction. The intervention is rank-one per target token.
    """
    head = clone_output_head(model)
    W = head.weight
    device, dtype = W.device, W.dtype
    edit_log = []
    for token_id, hs in directions.items():
        H = torch.stack(hs).to(device=device, dtype=torch.float32)
        normalized = H / (H.square().sum(dim=1, keepdim=True) + 1e-8)
        delta = alpha * normalized.mean(dim=0)
        W[token_id].sub_(delta.to(dtype=dtype))
        edit_log.append({
            "token_id": int(token_id), "contexts": len(hs),
            "delta_l2": float(delta.norm().item())
        })
    return model, edit_log

@torch.no_grad()
def apply_random_control(model, directions, edit_log, seed=17):
    """Matched control: same output rows and per-row edit norms, random direction."""
    head = clone_output_head(model)
    W = head.weight
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for item in edit_log:
        token_id = item["token_id"]
        norm = item["delta_l2"]
        rand = torch.randn(W.shape[1], generator=gen, dtype=torch.float32)
        rand = rand / (rand.norm() + 1e-8) * norm
        W[token_id].sub_(rand.to(device=W.device, dtype=W.dtype))
    return model
