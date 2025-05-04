# -*- coding: utf-8 -*-
"""Student‑ranking simulation with three fixed groups (254, 311, 319 classmates).

*2025‑05‑01 – Patch 3*
──────────────────────
• **Multithreading support** — `simulate_student_ranking` now accepts a
  `n_workers` parameter (≥ 1).  When `n_workers > 1`, the workload is split
  across a :class:`concurrent.futures.ThreadPoolExecutor` so that multiple
  cohorts are simulated in parallel.  NumPy releases the GIL during heavy
  array maths, so thread‑level parallelism brings a noticeable speed‑up while
  avoiding the pickling overhead of ``multiprocessing``.
• No behavioural change when ``n_workers == 1`` (default).
• Refactored core loop into the private helper ``_simulate_chunk`` so it can
  be reused by each worker.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_SIZES_FULL = np.array([254, 311, 320])
GROUP_SIZES: np.ndarray = GROUP_SIZES_FULL.copy()
GROUP_SIZES[-1] -= 1  # 319
NB_CLASSMATES: int = int(GROUP_SIZES.sum())  # 884

GROUP_LABELS: np.ndarray = np.repeat(np.arange(3), GROUP_SIZES)  # 0/1/2

# ---------------------------------------------------------------------------
# Helper: conversion rank → notes
# ---------------------------------------------------------------------------


def _m1_from_ranks(ranks: np.ndarray) -> np.ndarray:
    """Convert *global* ranks (1‑based in the full cohort) to M1 marks."""
    k = 420 + (ranks - 1)
    return 20.0 * (1.0 - k / 1798.0)


def _rank_inside_groups(ranks: np.ndarray) -> np.ndarray:
    """Return *1‑based* ranks inside each pre‑defined group.

    Supports both a 1‑D array (single cohort) and a 2‑D array of shape
    ``(n_cohorts, 884)``.  The returned array has the same shape as *ranks*.
    """
    ranks = np.asarray(ranks)

    if ranks.ndim == 1:
        intra = np.empty_like(ranks)
        for g in range(3):
            mask = GROUP_LABELS == g
            intra[mask] = ranks[mask].argsort().argsort() + 1
        return intra

    if ranks.ndim == 2:
        n_cohorts = ranks.shape[0]
        intra = np.empty_like(ranks)
        for g, size in enumerate(GROUP_SIZES):
            mask = GROUP_LABELS == g  # (884,)
            sub = ranks[:, mask]  # (n_cohorts, size)
            order = np.argsort(sub, axis=1)
            tmp = np.empty_like(sub)
            tmp[np.arange(n_cohorts)[:, None], order] = np.arange(size) + 1
            intra[:, mask] = tmp
        return intra

    raise ValueError("ranks array must be 1‑D or 2‑D")


def _m2_from_ranks(ranks: np.ndarray) -> np.ndarray:
    intra = _rank_inside_groups(ranks) - 1  # 0‑based intra ranks
    gsize = GROUP_SIZES[GROUP_LABELS]
    if ranks.ndim == 2:
        gsize = gsize[None, :]  # broadcast over rows
    return 20.0 * (1.0 - intra / (gsize - 1))


# ---------------------------------------------------------------------------
# Core simulation helpers (serial + threaded wrappers)
# ---------------------------------------------------------------------------


def _simulate_chunk(
    n_simulations: int,
    rang_souhaite: int,
    note_m1_perso: float,
    note_m2_perso: float,
    rho: float,
    batch: int,
    seed: int | None = None,
) -> int:
    """Run *n_simulations* and return the number of *successes* (n_ok)."""
    rng = np.random.default_rng(seed)

    target_avg = (note_m1_perso + note_m2_perso) / 2.0
    cov = np.array([[1.0, rho], [rho, 1.0]])
    n_done = n_ok = 0

    while n_done < n_simulations:
        n = min(batch, n_simulations - n_done)
        z = rng.multivariate_normal([0.0, 0.0], cov, size=(n, NB_CLASSMATES))

        rank_m1 = np.argsort(z[:, :, 0], axis=1).argsort(axis=1) + 1
        rank_m2 = np.argsort(z[:, :, 1], axis=1).argsort(axis=1) + 1

        m1 = _m1_from_ranks(rank_m1)
        m2 = _m2_from_ranks(rank_m2)

        averages = (m1 + m2) / 2.0
        n_better = (averages > target_avg).sum(axis=1)
        n_ok += (n_better <= rang_souhaite - 1).sum()
        n_done += n

    return n_ok


def simulate_student_ranking(
    n_simulations: int = 10000,
    rang_souhaite: int = 200,
    note_m1_perso: float = 13,
    note_m2_perso: float = 13,
    rho: float = 0.5,
    rng: np.random.Generator | None = None,
    batch: int = 250,
    n_workers: int = 1,
    *,
    _thread_pool_max_workers: int | None = None,
) -> Tuple[float, float]:
    """Estimate the probability of achieving *rang_souhaite*.

    Parameters
    ----------
    n_simulations : int, default 10000
        Number of Monte‑Carlo cohorts.
    rang_souhaite : int, default 124
        Target rank (global position among 884 classmates).
    note_m1_perso, note_m2_perso : float
        Personal M1/M2 marks.
    rho : float, default 0.5
        Rank‑to‑rank correlation between M1 and M2.
    rng : numpy.random.Generator, optional
        Custom RNG for reproducibility (ignored when *n_workers > 1*).
    batch : int, default 250
        Cohorts are simulated in batches of this size for memory efficiency.
    n_workers : int, default 1
        Number of worker threads.  ``1`` keeps the original serial behaviour.
    _thread_pool_max_workers : int | None
        Internal knob to override the size of the pool (for testing).
    """
    if n_workers < 1:
        raise ValueError("n_workers must be ≥ 1")

    if n_workers == 1:
        # Keep the original deterministic behaviour when a custom RNG is given.
        seed = None if rng is None else rng.bit_generator.random_raw()
        n_ok = _simulate_chunk(
            n_simulations,
            rang_souhaite,
            note_m1_perso,
            note_m2_perso,
            rho,
            batch,
            seed,
        )
        p = n_ok / n_simulations
        se = np.sqrt(p * (1 - p) / n_simulations)
        return p, se

    # ---------------------------------------------------------------------
    # Parallel case: split the workload across *n_workers* threads.
    # ---------------------------------------------------------------------
    # We purposely spawn fresh RNGs in each worker so that their streams are
    # independent.  Seeds are derived from NumPy's BitGenerator for safety.
    global_seed = (
        np.random.SeedSequence() if rng is None else rng.bit_generator.seed_seq
    )
    child_seeds: List[int] = global_seed.spawn(n_workers)

    # Distribute the total number of simulations as evenly as possible.
    base = n_simulations // n_workers
    sims_per_worker = [base] * n_workers
    for i in range(n_simulations % n_workers):
        sims_per_worker[i] += 1

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_thread_pool_max_workers or n_workers
    ) as pool:
        futs = [
            pool.submit(
                _simulate_chunk,
                n_simulations=sims,
                rang_souhaite=rang_souhaite,
                note_m1_perso=note_m1_perso,
                note_m2_perso=note_m2_perso,
                rho=rho,
                batch=batch,
                seed=int(child_seeds[i].generate_state(1)[0]),
            )
            for i, sims in enumerate(sims_per_worker)
        ]
        n_ok = sum(f.result() for f in concurrent.futures.as_completed(futs))

    p = n_ok / n_simulations
    se = np.sqrt(p * (1 - p) / n_simulations)
    return p, se


# ---------------------------------------------------------------------------
# Cohort generation & export helpers (unchanged)
# ---------------------------------------------------------------------------


def simulate_one_cohort(
    rho: float = 0.5, seed: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(
        [0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=NB_CLASSMATES
    )

    rank_m1 = np.argsort(z[:, 0]).argsort() + 1
    rank_m2 = np.argsort(z[:, 1]).argsort() + 1

    df = (
        pd.DataFrame(
            {
                "group": GROUP_LABELS,
                "rank_m2": rank_m2,
                "rank_in_group": _rank_inside_groups(rank_m2),
                "note_m1": _m1_from_ranks(rank_m1),
                "note_m2": _m2_from_ranks(rank_m2),
            }
        )
        .sort_values(["group", "rank_in_group"])
        .reset_index(drop=True)
    )

    return df


def cohort_to_excel(
    rho: float = 0.7, seed: int | None = None, path: str | Path = "cohort.xlsx"
) -> Path:
    df = simulate_one_cohort(rho=rho, seed=seed)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path
