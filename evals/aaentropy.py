# metrics_aaentropy.py

import math
from collections import Counter
from itertools import chain

def aa_entropy_for_set(seqs):
    seqs = [s for s in seqs if isinstance(s, str) and len(s) > 0]
    if not seqs:
        return float("nan")

    counts = Counter(chain.from_iterable(seqs))
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p)
    return H
