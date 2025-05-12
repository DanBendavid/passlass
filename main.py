import os
import sys

from simulation import cohort_to_excel, simulate_student_ranking

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Probability curve example (ρ = 0.5, 0.7) using multithreading
    rhos, ranks = [0.8, 0.9, 1.0], range(200, 300)
    for rho in rhos:
        pvals = [
            simulate_student_ranking(
                rang_souhaite=r, rho=rho, n_simulations=1000, n_workers=4
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

    cohort_to_excel()
