from src.metrics import ngram_repetition_rate, distinct_n

def test_repetition_detects_duplicate_ngrams():
    # (1,2,3) appears twice; there are four 3-gram positions.
    assert ngram_repetition_rate([1,2,3,1,2,3], n=3) == 0.25

def test_distinct_bigram():
    assert distinct_n([1,2,1,2], n=2) == 2/3

def test_short_sequence():
    assert ngram_repetition_rate([1,2], n=3) == 0.0
