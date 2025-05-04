# app.py
import streamlit as st
import matplotlib.pyplot as plt
from simulation import simulate_student_ranking

from core import (
    convert_rank_to_note_m1,
    convert_rank_to_note_m2,
    get_google_sheet,
    collect_to_google_sheet,
    get_dataframe_from_sheet,
    calculate_rho,
)

st.set_page_config(page_title="Simulation Classement PASS/LAS")

st.title("Simulation de classement")

st.info("⚠️ Une seule simulation par personne est autorisée. Vérifiez bien vos rangs avant de lancer.")

rank_m1 = st.number_input("🎓 Rang PASS (sur 1799)", min_value=1, max_value=1799, value=100)
nom_las = st.text_input("🏫 Nom de votre LAS", max_chars=100)
size_m2 = st.number_input("👥 Effectif LAS2", min_value=2, value=300)
rank_m2 = st.number_input("🎓 Rang LAS2", min_value=1, max_value=size_m2, value=50)
rang_souhaite = st.number_input("🎯 Rang souhaité (sur 884)", min_value=1, max_value=884, value=200)
rho = st.slider("🔗 Corrélation PASS/LAS", 0.7, 1.0, 0.85, step=0.05)
n = st.number_input("🔁 Nombre de simulations", min_value=100, max_value=20000, value=10000, step=1000)
show_graph = st.checkbox("📈 Afficher le graphique de probabilité", value=True)
n_workers = 4

if st.button("Lancer la simulation"):
    note_m1 = convert_rank_to_note_m1(rank_m1)
    note_m2 = convert_rank_to_note_m2(rank_m2, size_m2)

    st.write(f"🧮 Note M1 estimée : {note_m1:.2f}")
    st.write(f"🧮 Note M2 estimée : {note_m2:.2f}")

    try:
        sheet = get_google_sheet(st.secrets["GOOGLE_SHEET_KEY"], dict(st.secrets))
        success, status = collect_to_google_sheet(
            sheet,
            (nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, rang_souhaite)
        )

        if not success and status == "DUPLICATE":
            st.error("🚫 Simulation déjà enregistrée. Contactez l’administrateur pour une nouvelle tentative.")
            st.stop()

        p, se = simulate_student_ranking(
            n_simulations=n,
            rang_souhaite=rang_souhaite,
            note_m1_perso=note_m1,
            note_m2_perso=note_m2,
            rho=rho,
            n_workers=n_workers,
        )

        if show_graph:
            st.subheader("📉 Probabilité selon le rang souhaité")
            fig, ax = plt.subplots()
            rhos = [0.8, 0.9, 1.0]
            ranks = list(range(max(1, rang_souhaite - 50), min(884, rang_souhaite + 51), 2))

            progress_bar = st.progress(0)
            total = len(rhos) * len(ranks)
            count = 0

            for r in rhos:
                pvals = []
                for target_rank in ranks:
                    p_y, _ = simulate_student_ranking(
                        rang_souhaite=target_rank,
                        rho=r,
                        n_simulations=1000,
                        note_m1_perso=note_m1,
                        note_m2_perso=note_m2,
                        n_workers=n_workers,
                    )
                    pvals.append(p_y)
                    count += 1
                    progress_bar.progress(count / total)

                ax.plot(ranks, pvals, label=f"ρ = {r}")
            progress_bar.empty()
            ax.set_xlabel("Rang souhaité")
            ax.set_ylabel("Probabilité")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        st.success(f"📊 Probabilité d’être dans le top {rang_souhaite} avec ρ = {rho} : {int(p * 100)}% ± {int(se * 100)}%")

        st.subheader("🔗 Corrélation empirique entre les notes M1 et M2")
        df = get_dataframe_from_sheet(sheet)
        rho_empirique = calculate_rho(df)
        if rho_empirique:
            st.dataframe(df[["note m1", "note m2"]])
            st.success(f"🔗 Corrélation empirique observée : **{rho_empirique:.3f}**")
        else:
            st.warning(f"📉 Pas assez de données pour calculer une corrélation fiable : {len(df)}")

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
