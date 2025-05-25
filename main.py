import os
import sys

import matplotlib.pyplot as plt

from simulation import simulate_student_ranking

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PLOT_ = True
if __name__ == "__main__":

    if PLOT_:

        # Probability curve example (ρ = 0.5, 0.7) using multithreading
        rhos, ranks = [0.7, 1], range(120, 160, 2)
        for rho in rhos:
            pvals = [
                simulate_student_ranking(
                    rang_souhaite=r,
                    rho=rho,
                    n_simulations=1000,
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
