# metrics_tcrdist.py

import os
import numpy as np
from itertools import combinations

# --- Amino acid map ---
AMINO_MAP = {'@':24, '*': 23, 'A': 0, 'C': 4, 'B': 20,
             'E': 6, 'D': 3, 'G': 7, 'F': 13, 'I': 9, 'H': 8,
             'K': 11, 'M': 12, 'L': 10, 'N': 2, 'Q': 5, 'P': 14,
             'S': 15, 'R': 1, 'T': 16, 'W': 17, 'V': 19, 'Y': 18,
             'X': 22, 'Z': 21}

# --- Load BLOSUM once ---
def load_blosum(blosum_path):
    lines = []
    with open(blosum_path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                lines.append(line)

    embedding = [[float(x) for x in l.strip().split()[1:]]
                 for l in lines[1:]]
    embedding.append([0.0] * len(embedding[0]))
    return np.array(embedding)

def imgt_align(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    if len1 < len2:
        diff = len2 - len1
        seq1 = seq1[:len1//2] + "-" * diff + seq1[len1//2:]
    elif len2 < len1:
        diff = len1 - len2
        seq2 = seq2[:len2//2] + "-" * diff + seq2[len2//2:]
    return seq1, seq2

def compute_aadist(seq1, seq2, embedding):
    seq1, seq2 = imgt_align(seq1, seq2)
    dist = 0
    for a, b in zip(seq1, seq2):
        if a == "-" or b == "-":
            dist += 8
        elif a == b:
            continue
        else:
            dist += min(4, 4 - embedding[AMINO_MAP[a]][AMINO_MAP[b]])
    return dist

def mean_tcrdist(seqs, embedding, max_samples=100):
    seqs = list(seqs)
    if len(seqs) < 2:
        return np.nan
    if len(seqs) > max_samples:
        seqs = np.random.choice(seqs, max_samples, replace=False)

    dists = [compute_aadist(a, b, embedding)
             for a, b in combinations(seqs, 2)]
    return float(np.mean(dists)) if dists else np.nan
