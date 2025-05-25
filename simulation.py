# -*- coding: utf-8 -*-
"""Student‑ranking simulation with three fixed groups (254, 311, 319 classmates).

*2025‑05‑01 – Patch 3 - CORRECTED*
──────────────────────────────────
• **Fix du bug de classement** — Correction de la fonction rank_inside_groups
  pour s'assurer que les rangs intra-groupe sont calculés correctement
• **Cohérence des rangs** — Un étudiant rang 40 dans sa filière ne peut plus
  être rang 40 global s'il y a plusieurs filières
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
NB_PASS = 1799
GROUP_SIZES_FULL = np.array([254, 289, 70, 31, 32, 33, 34, 35, 36, 37])
GROUP_SIZES: np.ndarray = GROUP_SIZES_FULL.copy()
GROUP_SIZES[-1] -= 1  # 319
NB_LAS: int = int(GROUP_SIZES.sum())  # 850

GROUP_LABELS: np.ndarray = np.repeat(np.arange(10), GROUP_SIZES)  # 0/1/2/3
GROUP_LABELS_FULL = [f"G{i+1}" for i in range(len(GROUP_SIZES_FULL))]
# ---------------------------------------------------------------------------
# Helper: conversion rank → notes
# ---------------------------------------------------------------------------
SEED_COHORT = 26


def get_cohort1(seed: int = 26) -> np.ndarray:
    """
    Simule les 3 étapes de sélection et retourne un tableau NumPy
    contenant les rangs initiaux (Step 1) des 850 candidats de la
    cohorte triés par ordre croissant.

    Paramètres
    ----------
    seed : int, par défaut 42
        Graine pour la reproductibilité.

    """
    rng = np.random.default_rng(seed)

    # — Étape 1 : rangs initiaux —
    scores = rng.random(NB_PASS)  # scores aléatoires dans [0,1)
    ranks = scores.argsort()[::-1].argsort() + 1  # 1 = meilleur, 1799 = pire

    # — Étape 2 : sélection pondérée parmi les rangs 201-650 —
    pool_mask = (ranks >= 170) & (ranks <= 650)  # bool (N,)
    pool_indices = np.nonzero(pool_mask)[0]  # indices des 450 candidats
    weights = 651 - ranks[pool_indices]
    probs = weights / weights.sum()  # probabilités normalisées
    selected2 = rng.choice(pool_indices, size=250, replace=False, p=probs)

    # candidats NON retenus à l'étape 2
    not_selected2_mask = pool_mask.copy()
    not_selected2_mask[selected2] = False  # on enlève les 250 tirés

    # — Étape 3 : cohorte « seconde chance » (850 candidats) —
    second_chance_mask = not_selected2_mask | ((ranks >= 651) & (ranks <= 1269))

    # Tableau des rangs initiaux puis tri croissant
    cohort_ranks = np.sort(ranks[second_chance_mask])  # shape (850,)
    return cohort_ranks


COHORT1 = get_cohort1(SEED_COHORT)


def _m1_from_ranks(ranks: np.ndarray) -> np.ndarray:
    """Convert *global* ranks (1‑based in the full cohort) to M1 marks."""
    return 20.0 * (1.0 - (ranks - 1) / (NB_PASS - 1))


def rank_inside_groups(scores, group_labels):
    """
    Calcule les rangs intra-groupe basés sur les SCORES (pas les rangs globaux).

    Parameters:
    -----------
    scores : array-like
        Les scores/notes des étudiants
    group_labels : array-like
        Les labels de groupe pour chaque étudiant

    Returns:
    --------
    intra_ranks : array
        Les rangs intra-groupe (1 = meilleur dans le groupe)
    """
    scores = np.asarray(scores)
    group_labels = np.asarray(group_labels)
    intra_ranks = np.empty_like(scores, dtype=int)

    for g in np.unique(group_labels):
        mask = group_labels == g
        group_scores = scores[mask]

        # Rang intra-groupe : 1 = meilleur score dans le groupe
        # argsort() donne l'ordre croissant, on inverse pour avoir décroissant
        sorted_indices = np.argsort(group_scores)[::-1]
        ranks_in_group = np.empty_like(sorted_indices)
        ranks_in_group[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

        intra_ranks[mask] = ranks_in_group

    return intra_ranks


def _m2_from_ranks(intra_ranks, group_labels):
    """
    Convertit les rangs intra-groupe en notes M2.

    Parameters:
    -----------
    intra_ranks : array
        Rangs intra-groupe (1-based)
    group_labels : array
        Labels de groupe
    """
    intra_0based = intra_ranks - 1  # Conversion en 0-based
    group_sizes = GROUP_SIZES[group_labels]
    return 20.0 * (1.0 - intra_0based / (group_sizes - 1))


# ---------------------------------------------------------------------------
# Core simulation helpers (serial + threaded wrappers)
# ---------------------------------------------------------------------------


def assign_groups_round_robin(n, group_sizes, rng=None):
    """
    Attribue les groupes de manière équitable mais aléatoire.

    Parameters:
    -----------
    n : int
        Nombre total d'étudiants
    group_sizes : array-like
        Tailles de chaque groupe
    rng : numpy.random.Generator, optional
        Générateur aléatoire pour mélanger les attributions

    Returns:
    --------
    group_labels : array
        Labels de groupe pour chaque étudiant
    """
    group_labels = np.empty(n, dtype=int)
    group_counters = np.zeros(len(group_sizes), dtype=int)

    # Attribution séquentielle d'abord
    for i in range(n):
        for g in range(len(group_sizes)):
            if group_counters[g] < group_sizes[g]:
                group_labels[i] = g
                group_counters[g] += 1
                break

    # CORRECTION : Mélange aléatoire des attributions
    if rng is not None:
        rng.shuffle(group_labels)
    else:
        # Fallback si pas de RNG fourni
        np.random.shuffle(group_labels)

    return group_labels


def assign_groups_round_robin_indices_and_names(
    n, group_sizes, group_names=None
):
    assert n == group_sizes.sum()
    if group_names is None:
        group_names = [f"G{i+1}" for i in range(len(group_sizes))]
    group_indices = np.empty(n, dtype=int)
    group_name_labels = np.empty(n, dtype=object)
    group_counters = np.zeros(len(group_sizes), dtype=int)
    i = 0
    group_idx = 0
    while i < n:
        if group_counters[group_idx] < group_sizes[group_idx]:
            group_indices[i] = group_idx  # entier pour indexation numpy
            group_name_labels[i] = group_names[
                group_idx
            ]  # label pour affichage
            group_counters[group_idx] += 1
            i += 1
        group_idx = (group_idx + 1) % len(group_sizes)
    return group_indices, group_name_labels


def _simulate_chunk(
    n_simulations: int,
    rang_souhaite: int,
    note_m1_perso: float,
    note_m2_perso: float,
    rho: float,
    batch: int,
    seed: int | None = None,
) -> int:
    """Run *n_simulations* and return the number of *successes* (n_ok),
    avec cohorte et groupes aléatoires cohérents, en tenant compte de
    l'effet de taille des filières sur la note M2."""

    rng = np.random.default_rng(seed)
    target_avg = (note_m1_perso + note_m2_perso) / 2.0
    n_done = n_ok = 0

    while n_done < n_simulations:
        n = min(batch, n_simulations - n_done)

        for _ in range(n):
            # 1. Tirage d'une nouvelle cohorte aléatoire (note M1 issue de COHORT1)
            cohort_ranks = get_cohort1(rng.integers(1e9))
            note_m1_fixed = _m1_from_ranks(cohort_ranks)

            # 2. Attribution homogène des groupes (filières)
            group_labels = assign_groups_round_robin(NB_LAS, GROUP_SIZES, rng)

            # 3. Génération de M2 corrélée à M1, groupe par groupe (effet filière activé)
            m2_std = np.empty(NB_LAS)
            for g in np.unique(group_labels):
                idx = group_labels == g
                size = idx.sum()

                m1_std_g = (
                    note_m1_fixed[idx] - note_m1_fixed[idx].mean()
                ) / note_m1_fixed[idx].std()
                noise = rng.standard_normal(size)

                m2_std_g = rho * m1_std_g + np.sqrt(1 - rho**2) * noise
                m2_std[idx] = m2_std_g

            # 4. CORRECTION : Calculer les rangs intra-groupe basés sur les SCORES M2
            # et non pas sur les rangs globaux
            rank_in_group = rank_inside_groups(m2_std, group_labels)

            # 5. Conversion rang intra-groupe → note M2 avec échelle par groupe
            note_m2_in_group = _m2_from_ranks(rank_in_group, group_labels)

            # 6. Calcul de la moyenne M1/M2 (pondération 50/50 ici)
            averages = (note_m1_fixed + note_m2_in_group) / 2.0

            # 7. Rang global du candidat fictif (note moyenne personnalisée)
            score_perso = target_avg
            rang_perso = (averages > score_perso).sum() + 1  # rang 1 = meilleur

            # 8. Est-il classé dans les premiers ?
            n_ok += rang_perso <= rang_souhaite

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
        Custom RNG for reproducibility (ignored when *n_workers > 1*).
    batch : int, default 250
        Cohorts are simulated in batches of this size for memory efficiency.
    n_workers : int, default 1
        Number of worker threads.  ``1`` keeps the original serial behaviour.
    _thread_pool_max_workers : int | None
        Internal knob to override the size of the pool (for testing).
    """
    if n_workers < 1:
        raise ValueError("n_workers must be ≥ 1")

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
