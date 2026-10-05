# genome.py
import hashlib
import random

import numpy as np

from config import (
    GENOME_SIMILARITY_THRESHOLD,
    GENOME_NAMES_A, GENOME_NAMES_B, GENOME_NAMES_C,
)


def normalize_matrix(X_raw):
    """
    Z-score per column then L2-normalize per row.
    Returns (X_norm, mean, std).
    """
    mean = np.nanmean(X_raw, axis=0)
    std = np.nanstd(X_raw, axis=0)
    std = np.where(std == 0, 1.0, std)
    X = (X_raw - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = X / norms
    return X, mean, std


def normalize_vector(v_raw, mean, std):
    v = (v_raw - mean) / std
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return v / n


def fingerprint(v_norm):
    payload = ",".join(f"{x:.5f}" for x in v_norm)
    return hashlib.sha1(payload.encode()).hexdigest()


def make_genome_name(symbol, uid):
    a = random.choice(GENOME_NAMES_A)
    b = random.choice(GENOME_NAMES_B)
    c = random.choice(GENOME_NAMES_C)
    return f"{symbol}-{a}-{b}-{c}-{uid}"


def build_genomes(X_norm, ts_list, cid_list,
                  threshold=GENOME_SIMILARITY_THRESHOLD):
    """
    Sequentially processes rows in temporal order, assigning each to an
    existing genome (if similarity to its genesis >= threshold) or
    creating a new genome.

    Returns:
      genomes: list of dicts
          {"local_idx", "genesis_idx", "genesis_ts", "genesis_candle_id"}
      instances: list of tuples
          (local_genome_idx, instance_number, ts, candle_id, similarity)
    """
    genomes = []
    instances = []
    genesis_matrix = None

    n = X_norm.shape[0]
    for i in range(n):
        x = X_norm[i]
        if genesis_matrix is None or genesis_matrix.shape[0] == 0:
            g_idx = 0
            genesis_matrix = x[None, :].copy()
            genomes.append({
                "local_idx": g_idx,
                "genesis_idx": i,
                "genesis_ts": ts_list[i],
                "genesis_candle_id": cid_list[i],
            })
            instances.append((g_idx, 0, ts_list[i], cid_list[i], 1.0))
            continue

        sims = genesis_matrix @ x
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            num = sum(1 for inst in instances if inst[0] == best_idx)
            instances.append((best_idx, num, ts_list[i],
                              cid_list[i], best_sim))
        else:
            g_idx = len(genomes)
            genesis_matrix = np.vstack([genesis_matrix, x[None, :]])
            genomes.append({
                "local_idx": g_idx,
                "genesis_idx": i,
                "genesis_ts": ts_list[i],
                "genesis_candle_id": cid_list[i],
            })
            instances.append((g_idx, 0, ts_list[i], cid_list[i], 1.0))

    return genomes, instances