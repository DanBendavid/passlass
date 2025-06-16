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
NB_PASS = 1799  # PASS
NB_LAS_1 = 450  # LAS1 Sciences
NB_LAS_3 = 400  # LAS2 Sciences

# LAS 2 GROUPES
GROUP_SIZES_FULL = np.array(
    [
        30,  # Chimie
        40,  # Maths et applications
        30,  # MEDPHY
        30,  # MIASHS
        30,  # Physique
        289,  # Sciences Biomédicales
        30,  # Sciences de la Vie
        37,  # Sciences de la Vie de la Terre (IPGP)
        254,  # SIAS
        40,  # STAPS
    ]
)
GROUP_SIZES: np.ndarray = GROUP_SIZES_FULL.copy()
NB_LAS2: int = int(GROUP_SIZES.sum())

GROUP_LABELS: np.ndarray = np.repeat(np.arange(10), GROUP_SIZES)
GROUP_LABELS_FULL = [f"G{i+1}" for i in range(len(GROUP_SIZES_FULL))]


def get_cohort(
    Nb_total: int,
    rang_min: int,
    rang_pivot: int,
    rang_max: int,
    n_pool1: int,
    n_pool2: int,
    alpha: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Génère une cohorte par sélection pondérée sur deux intervalles de rangs.

    Cette fonction est la base générique pour créer des cohortes spécifiques.

    Parameters:
    -----------
    Nb_total : int
        Effectif total de la population de départ.
    rang_min, rang_pivot, rang_max : int
        Bornes définissant les deux pools de sélection.
    n_pool1, n_pool2 : int
        Nombre d'individus à tirer dans chaque pool.
    alpha : float, default 2.0
        Intensité du biais de sélection.
    seed : int, optional
        Graine pour la reproductibilité du générateur aléatoire.
    """
    if not (1 <= rang_min < rang_pivot < rang_max <= Nb_total):
        raise ValueError("Les bornes de rangs ne sont pas valides.")

    rng = np.random.default_rng(seed)

    # Rangs globaux simulés (1 = meilleur)
    scores = rng.random(Nb_total)
    ranks = scores.argsort()[::-1].argsort() + 1

    # --- Pool 1 (rangs min à pivot) ---
    p1 = np.where((rang_min <= ranks) & (ranks <= rang_pivot))[0]
    if n_pool1 > len(p1):
        raise ValueError(
            f"n_pool1 ({n_pool1}) dépasse la taille du Pool 1 ({len(p1)})."
        )
    w1 = (ranks[p1] - rang_min + 1) ** alpha
    sel1 = rng.choice(p1, n_pool1, replace=False, p=w1 / w1.sum())

    # --- Pool 2 (rangs pivot+1 à max) ---
    p2 = np.where((rang_pivot + 1 <= ranks) & (ranks <= rang_max))[0]
    if n_pool2 > len(p2):
        raise ValueError(
            f"n_pool2 ({n_pool2}) dépasse la taille du Pool 2 ({len(p2)})."
        )
    w2 = (rang_max - ranks[p2] + 1) ** alpha
    sel2 = rng.choice(p2, n_pool2, replace=False, p=w2 / w2.sum())

    # Renvoyer les rangs retenus, triés
    return np.sort(ranks[np.concatenate([sel1, sel2])])


def get_cohort1(seed: int = 26) -> np.ndarray:
    """Génère la cohorte spécifique aux étudiants PASS."""
    return get_cohort(
        Nb_total=NB_PASS,
        rang_min=170,
        rang_pivot=500,
        rang_max=1270,
        n_pool1=250,
        n_pool2=320,
        alpha=2.0,
        seed=seed,
    )


def get_cohort2(seed: int = 26) -> np.ndarray:
    """Génère la cohorte spécifique aux étudiants LAS 1."""
    return get_cohort(
        Nb_total=NB_LAS_1,
        rang_min=40,
        rang_pivot=80,
        rang_max=200,
        n_pool1=40,
        n_pool2=80,
        alpha=2.0,
        seed=seed,
    )


def get_cohort3(seed: int = 26) -> np.ndarray:
    """Génère la cohorte spécifique aux étudiants LAS 1."""
    return get_cohort(
        Nb_total=NB_LAS_3,
        rang_min=40,
        rang_pivot=80,
        rang_max=200,
        n_pool1=40,
        n_pool2=80,
        alpha=2.0,
        seed=seed,
    )


def _m1_from_ranks(ranks: np.ndarray, size: int) -> np.ndarray:
    """Convert *global* ranks (1‑based in the full cohort) to M1 marks."""
    return 20.0 * (1.0 - (ranks - 1) / (size - 1))


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


import numpy as np


def assign_groups_balanced_pass_only_g9(
    scores_pass: np.ndarray,  # notes M1  PASS
    scores_las1: np.ndarray,  # notes M1  LAS-1
    scores_las3: np.ndarray,  # notes M1  LAS-3
    group_sizes: np.ndarray,  # quotas (somme = 810)
    rng: np.random.Generator | None = None,
    pass_only_group: int = 8,  # index 8  → « groupe 9 »
) -> np.ndarray:
    """
    Retourne `labels` (longueur 810) tel que :
    • le groupe 9 ne contient QUE des PASS ;
    • la distribution des niveaux M1 est la plus homogène possible.

    Hypothèse : l’appelant passe d’abord les 650 PASS puis les 160 LAS-1.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_pass = scores_pass.size
    n_las1 = scores_las1.size
    n_las3 = scores_las3.size
    n_total = n_pass + n_las1 + n_las3  # 810
    assert group_sizes.sum() == n_total

    # ------------------------------------------------------------------
    # 1.   Remplissage aléatoire du groupe 9 avec des PASS
    # ------------------------------------------------------------------
    quota_g9 = group_sizes[pass_only_group]
    all_pass_idx = np.arange(n_pass)
    g9_pass_idx = rng.choice(all_pass_idx, size=quota_g9, replace=False)
    rest_pass_idx = np.setdiff1d(all_pass_idx, g9_pass_idx, assume_unique=True)

    labels = np.full(n_total, -1, dtype=int)
    labels[g9_pass_idx] = pass_only_group

    remaining_quota = group_sizes.copy()
    remaining_quota[pass_only_group] = 0  # quota déjà rempli

    # ------------------------------------------------------------------
    # 2.   Candidats restant à répartir (PASS restants + LAS-1 + LAS-3)
    #      ==> équilibrage niveau par attribution pondérée
    # ------------------------------------------------------------------
    # concaténation des niveaux & indicateurs d'origine
    scores_mix = np.concatenate(
        [scores_pass[rest_pass_idx], scores_las1, scores_las3]
    )
    is_pass_mix = np.concatenate(
        [
            np.ones(rest_pass_idx.size, dtype=bool),  # True  pour PASS restants
            np.zeros(n_las1, dtype=bool),  # False pour LAS-1
            np.zeros(n_las3, dtype=bool),  # False pour LAS-3
        ]
    )

    # ordre décroissant de score
    order = scores_mix.argsort()[::-1]

    for idx in order:  # du meilleur au moins bon
        cand_is_pass = is_pass_mix[idx]

        # liste des groupes encore disponibles
        allowed_groups = np.flatnonzero(remaining_quota)
        if not cand_is_pass:  # LAS-1 → on retire le groupe 9
            allowed_groups = allowed_groups[allowed_groups != pass_only_group]

        # pondération proportionnelle au quota restant
        weights = remaining_quota[allowed_groups].astype(float)
        weights /= weights.sum()

        g = rng.choice(allowed_groups, p=weights)
        labels_idx = (
            rest_pass_idx[idx]  # si PASS restant
            if cand_is_pass
            else n_pass + (idx - rest_pass_idx.size)  # LAS-1
        )
        labels[labels_idx] = g
        remaining_quota[g] -= 1

    assert (labels >= 0).all(), "Affectation incomplète"
    assert (remaining_quota == 0).all(), "Quotas non respectés"
    assert (labels[n_pass:] != pass_only_group).all(), "LAS-1 dans le groupe 9"

    return labels


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
    cohorte_fixe: bool = True,
) -> int:
    """Run *n_simulations* and return the number of *successes* (n_ok),
    avec cohorte et groupes aléatoires cohérents, en tenant compte de
    l'effet de taille des filières sur la note M2."""

    rng = np.random.default_rng(seed)
    target_avg = (note_m1_perso + note_m2_perso) / 2.0
    n_done = n_ok = 0

    if cohorte_fixe:
        cohort_ranks1 = get_cohort1(rng.integers(1e9))
        note_m1_fixed1 = _m1_from_ranks(cohort_ranks1, NB_PASS)

        cohort_ranks2 = get_cohort2(rng.integers(1e9))
        note_m1_fixed2 = _m1_from_ranks(cohort_ranks2, NB_LAS_1)

        cohort_ranks3 = get_cohort3(rng.integers(1e9))
        note_m1_fixed3 = _m1_from_ranks(cohort_ranks3, NB_LAS_3)

        note_m1_fixed = np.concatenate(
            [note_m1_fixed1, note_m1_fixed2, note_m1_fixed3]
        )

        group_labels = assign_groups_balanced_pass_only_g9(
            scores_pass=note_m1_fixed1,
            scores_las1=note_m1_fixed2,
            scores_las3=note_m1_fixed3,
            group_sizes=GROUP_SIZES,
            rng=rng,
        )

    while n_done < n_simulations:
        n = min(batch, n_simulations - n_done)

        for _ in range(n):

            if not cohorte_fixe:
                # Tirage d'une nouvelle cohorte à chaque simulation
                cohort_ranks1 = get_cohort1(rng.integers(1e9))
                note_m1_fixed1 = _m1_from_ranks(cohort_ranks1, NB_PASS)

                cohort_ranks2 = get_cohort2(rng.integers(1e9))
                note_m1_fixed2 = _m1_from_ranks(cohort_ranks2, NB_LAS_1)

                cohort_ranks3 = get_cohort3(rng.integers(1e9))
                note_m1_fixed3 = _m1_from_ranks(cohort_ranks2, NB_LAS_3)

                note_m1_fixed = np.concatenate(
                    [note_m1_fixed1, note_m1_fixed2], note_m1_fixed3
                )

                group_labels = assign_groups_balanced_pass_only_g9(
                    scores_pass=note_m1_fixed1,
                    scores_las1=note_m1_fixed2,
                    scores_las3=note_m1_fixed3,
                    group_sizes=GROUP_SIZES,
                    rng=rng,
                )

            # 3. Génération de M2 corrélée à M1, groupe par groupe (effet filière activé)
            m2_std = np.empty(NB_LAS2)
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
        Target rank (global position among 777 classmates).
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
            cohorte_fixe=True,
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
