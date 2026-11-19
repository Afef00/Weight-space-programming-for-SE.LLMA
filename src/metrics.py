from collections import Counter

def ngram_repetition_rate(token_ids, n=3):
    """Fraction of n-gram positions that repeat an earlier n-gram."""
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids)-n+1)]
    seen, repeated = set(), 0
    for gram in grams:
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / max(1, len(grams))

def distinct_n(token_ids, n=2):
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids)-n+1)]
    return len(set(grams)) / len(grams)

def summarize_token_sequences(sequences):
    reps, d2, lengths = [], [], []
    for ids in sequences:
        reps.append(ngram_repetition_rate(ids, 3))
        d2.append(distinct_n(ids, 2))
        lengths.append(len(ids))
    if not sequences:
        return {"repeated_3gram_rate": None, "distinct_2": None, "mean_new_tokens": None}
    return {
        "repeated_3gram_rate": sum(reps)/len(reps),
        "distinct_2": sum(d2)/len(d2),
        "mean_new_tokens": sum(lengths)/len(lengths)
    }
