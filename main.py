import os
import sys

from simulation import simulate_student_ranking

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    # Compare serial vs. threaded execution speed
    start = time.perf_counter()
    p_serial, _ = simulate_student_ranking(
        n_simulations=5000, rho=0.6, n_workers=1
    )
    t_serial = time.perf_counter() - start

    start = time.perf_counter()
    p_thread, _ = simulate_student_ranking(
        n_simulations=5000, rho=0.6, n_workers=4
    )
    t_thread = time.perf_counter() - start

    print(f"Serial (1 worker): p = {p_serial:.4f}  •  {t_serial:.2f}s")
    print(f"Threaded (4 workers): p = {p_thread:.4f}  •  {t_thread:.2f}s")

    # Probability curve example (ρ = 0.5, 0.7) using multithreading
    rhos, ranks = [0.8, 0.9, 1.0], range(100, 160)
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
