import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from simulation import simulate_student_ranking

# Constants

rank_target = 200
# Fonction pour tracer le graphique


def graph_student_ranking(
    rank_target, rho, n_simulations, note_m1, note_m2, n_workers
):
    rank_fifty = None
    rhos = [rho, 1.0]

    ranks = list(range(max(1, rank_target - 50), min(884, rank_target + 51), 2))
    fig, ax = plt.subplots()

    progress_bar = st.progress(0)
    total_steps = len(rhos) * len(ranks)
    step = 0

    for r in rhos:
        pvals = []
        for rank_ in ranks:
            p_y = simulate_student_ranking(
                rang_souhaite=rank_,
                rho=r,
                n_simulations=min(2000, n_simulations),
                note_m1_perso=note_m1,
                note_m2_perso=note_m2,
                n_workers=n_workers,
            )[0]
            pvals.append(p_y)

            if rank_fifty is None and p_y > 0.5:
                rank_fifty = rank_

            step += 1
            progress_bar.progress(step / total_steps)

        ax.plot(ranks, pvals, label=f"ρ = {r}")

    progress_bar.empty()  # Supprime la barre une fois terminé

    ax.set_xlabel("Rang souhaité")
    ax.set_ylabel("Probabilité")
    ax.set_title("Probabilité d'atteindre un rang donné")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    if rank_fifty is not None:
        st.success(f"📊 Rang 50/50 avec ρ = {rho} : {rank_fifty}")
        return rank_fifty
    else:
        st.warning(
            f"📊 Pas de rang 50/50 trouvé avec ρ = {rho} dans la plage de simulation. Augmenter le rang cible"
        )
