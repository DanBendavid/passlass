import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation import (
    GROUP_LABELS_FULL,
    GROUP_SIZES,
    NB_LAS2,
    NB_LAS_1,
    NB_PASS,
    _m1_from_ranks,
    _m2_from_ranks,
    assign_groups_balanced_pass_only_g9,
    assign_groups_round_robin,
    get_cohort1,
    get_cohort2,
    rank_inside_groups,
)


def export_simulation_to_excel(
    filename: str = "simulation_complete.xlsx",
    note_m1_perso: float = 13,
    note_m2_perso: float = 17,
    rho: float = 0.85,
    seed: int = 42,
) -> str:
    """
    Exporte une simulation complète vers Excel avec toutes les données détaillées.

    Parameters:
    -----------
    filename : str
        Nom du fichier Excel à créer
    note_m1_perso, note_m2_perso : float
        Notes personnelles M1 et M2 pour le candidat fictif
    rho : float
        Corrélation entre M1 et M2
    seed : int
        Graine pour la reproductibilité

    Returns:
    --------
    str : Message de confirmation avec statistiques
    """
    rng = np.random.default_rng(seed)
    target_avg = (note_m1_perso + note_m2_perso) / 2.0

    # 1. Génération d'une cohorte
    cohort_ranks1 = get_cohort1(rng.integers(1e9))
    note_m1_fixed1 = _m1_from_ranks(cohort_ranks1, NB_PASS)
    cohort_ranks2 = get_cohort2(rng.integers(1e9))
    note_m1_fixed2 = _m1_from_ranks(cohort_ranks2, NB_LAS_1)
    note_m1_fixed = np.concatenate([note_m1_fixed1, note_m1_fixed2])
    cohort_ranks = np.concatenate([cohort_ranks1, cohort_ranks2])
    # 2. Attribution des groupes (ALÉATOIRE)
    rng_groups = np.random.default_rng(
        seed + 1000
    )  # RNG séparé pour les groupes
    group_labels = assign_groups_balanced_pass_only_g9(
        scores_pass=note_m1_fixed1,
        scores_las1=note_m1_fixed2,
        group_sizes=GROUP_SIZES,
        rng=rng,
    )
    group_names = [GROUP_LABELS_FULL[i] for i in group_labels]

    # 3. Génération de M2 corrélée à M1
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

    # 4. Calcul des rangs et notes
    rank_m1_global = cohort_ranks  # Rangs M1 globaux (issus de la cohorte)
    rank_m2_intra = rank_inside_groups(m2_std, group_labels)
    note_m2_scaled = _m2_from_ranks(rank_m2_intra, group_labels)

    # 5. Moyennes et classement final
    averages = (note_m1_fixed + note_m2_scaled) / 2.0
    rang_global_final = np.argsort(averages)[::-1].argsort() + 1

    # 6. Position du candidat fictif
    score_perso = target_avg
    rang_perso = (averages > score_perso).sum() + 1

    # 7. Création du DataFrame
    df = pd.DataFrame(
        {
            "Etudiant_ID": range(1, NB_LAS2 + 1),
            "Groupe_Num": group_labels + 1,  # 1-indexed pour Excel
            "Groupe_Nom": group_names,
            "Rang_M1_Global": rank_m1_global,
            "Note_M1": np.round(note_m1_fixed, 2),
            "Score_M2_Std": np.round(m2_std, 4),
            "Rang_M2_IntraGroupe": rank_m2_intra,
            "Note_M2_Scaled": np.round(note_m2_scaled, 2),
            "Note_Moyenne": np.round(averages, 2),
            "Rang_Global_Final": rang_global_final,
        }
    )

    # 8. Ajout des statistiques par groupe
    stats_by_group = []
    for g in range(len(GROUP_SIZES)):
        mask = group_labels == g
        group_data = df[mask]
        stats_by_group.append(
            {
                "Groupe": GROUP_LABELS_FULL[g],
                "Taille": len(group_data),
                "Note_M1_Moyenne": np.round(group_data["Note_M1"].mean(), 2),
                "Note_M1_Std": np.round(group_data["Note_M1"].std(), 2),
                "Note_M2_Moyenne": np.round(
                    group_data["Note_M2_Scaled"].mean(), 2
                ),
                "Note_M2_Std": np.round(group_data["Note_M2_Scaled"].std(), 2),
                "Note_Finale_Moyenne": np.round(
                    group_data["Note_Moyenne"].mean(), 2
                ),
                "Note_Finale_Std": np.round(
                    group_data["Note_Moyenne"].std(), 2
                ),
                "Meilleur_Rang_Global": group_data["Rang_Global_Final"].min(),
                "Pire_Rang_Global": group_data["Rang_Global_Final"].max(),
            }
        )

    stats_df = pd.DataFrame(stats_by_group)

    # 9. Informations sur le candidat fictif
    candidat_info = pd.DataFrame(
        {
            "Parametre": [
                "Note_M1_Perso",
                "Note_M2_Perso",
                "Note_Moyenne_Perso",
                "Rang_Simule",
                "Correlation_rho",
                "Seed",
            ],
            "Valeur": [
                note_m1_perso,
                note_m2_perso,
                target_avg,
                rang_perso,
                rho,
                seed,
            ],
        }
    )

    # 10. Export vers Excel
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Données principales
        df.to_excel(writer, sheet_name="Simulation_Complete", index=False)

        # Statistiques par groupe
        stats_df.to_excel(writer, sheet_name="Stats_Par_Groupe", index=False)

        # Informations candidat
        candidat_info.to_excel(
            writer, sheet_name="Parametres_Simulation", index=False
        )

        # Top 50 et Bottom 50
        df_sorted = df.sort_values("Rang_Global_Final")
        df_sorted.head(50).to_excel(writer, sheet_name="Top_50", index=False)
        df_sorted.tail(50).to_excel(writer, sheet_name="Bottom_50", index=False)

    # 11. Statistiques de retour
    stats_msg = f"""
Simulation exportée vers '{filename}' avec succès !

STATISTIQUES GÉNÉRALES:
- Nombre d'étudiants: {NB_LAS2}
- Nombre de groupes: {len(GROUP_SIZES)}
- Corrélation M1-M2: {rho}

CANDIDAT FICTIF:
- Note M1: {note_m1_perso}
- Note M2: {note_m2_perso}
- Note moyenne: {target_avg}
- Rang simulé: {rang_perso}/{NB_LAS2}
- Percentile: {100 * (1 - rang_perso/NB_LAS2):.1f}%

RÉPARTITION PAR NOTES MOYENNES:
- [18-20]: {len(df[df['Note_Moyenne'] >= 18])} étudiants
- [16-18): {len(df[(df['Note_Moyenne'] >= 16) & (df['Note_Moyenne'] < 18)])} étudiants  
- [14-16): {len(df[(df['Note_Moyenne'] >= 14) & (df['Note_Moyenne'] < 16)])} étudiants
- [12-14): {len(df[(df['Note_Moyenne'] >= 12) & (df['Note_Moyenne'] < 14)])} étudiants
- [10-12): {len(df[(df['Note_Moyenne'] >= 10) & (df['Note_Moyenne'] < 12)])} étudiants
- [0-10): {len(df[df['Note_Moyenne'] < 10])} étudiants

FEUILLES EXCEL CRÉÉES:
- 'Simulation_Complete': Tous les étudiants avec détails complets
- 'Stats_Par_Groupe': Statistiques détaillées par filière  
- 'Parametres_Simulation': Paramètres utilisés et résultat candidat fictif
- 'Top_50': Les 50 meilleurs étudiants
- 'Bottom_50': Les 50 moins bons étudiants
"""

    return stats_msg


# Fonction de test pour vérifier la correction
def test_ranking_logic():
    """Test pour vérifier que la logique de classement est correcte."""
    np.random.seed(42)

    # Simulation simple avec 3 groupes
    group_sizes = np.array([3, 3, 4])  # 10 étudiants total
    n_students = group_sizes.sum()

    # Attribution des groupes
    group_labels = assign_groups_round_robin(n_students, group_sizes)
    print(f"Groupes: {group_labels}")

    # Scores M2 simulés
    m2_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    print(f"Scores M2: {m2_scores}")

    # Rangs intra-groupe
    intra_ranks = rank_inside_groups(m2_scores, group_labels)
    print(f"Rangs intra-groupe: {intra_ranks}")

    # Rangs globaux (pour comparaison)
    global_ranks = np.argsort(m2_scores)[::-1].argsort() + 1
    print(f"Rangs globaux: {global_ranks}")

    # Vérification : l'étudiant avec le meilleur score global doit être rang 1 global
    # mais pas forcément rang 1 dans son groupe s'il y a de meilleurs étudiants dans son groupe
    print(f"\nVérification:")
    for i, (score, global_rank, intra_rank, group) in enumerate(
        zip(m2_scores, global_ranks, intra_ranks, group_labels)
    ):
        print(
            f"Étudiant {i}: Score={score:.2f}, Rang global={global_rank}, "
            f"Rang intra-groupe={intra_rank}, Groupe={group}"
        )


if __name__ == "__main__":
    export_simulation_to_excel()
