import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation import NB_LAS_1, get_cohort1, simulate_student_ranking

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PLOT_ = True
if __name__ == "__main__":
    rng = np.random.default_rng(26)
    cohort_ranks = get_cohort1(rng.integers(1e9))
    df = pd.DataFrame(
        {
            "Etudiant_ID": np.arange(1, len(cohort_ranks) + 1),  # 1 … 160
            "Rang_M1_Global": cohort_ranks,  # 45 … 270
        }
    )

    print(df)

    if PLOT_:

        # Probability curve example (ρ = 0.5, 0.7) using multithreading
        rhos, ranks = [0.7, 1], range(160, 200, 1)
        for rho in rhos:
            pvals = [
                simulate_student_ranking(
                    rang_souhaite=r,
                    rho=rho,
                    n_simulations=2000,
                    n_workers=4,
                    note_m1_perso=12.84,
                    note_m2_perso=17.24,
                )[0]
                for r in ranks
            ]
            plt.plot(list(ranks), pvals, label=f"ρ = {rho}")

        plt.xlabel("Rang souhaité")
        plt.ylabel("Probabilité")
        plt.title("Probabilité d'atteindre un rang donné (groupes fixes)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
