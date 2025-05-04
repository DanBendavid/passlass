import hashlib
import json

import gspread
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials

from simulation import (
    simulate_student_ranking,
)  # adapte l'import selon ton fichier


def generate_user_hash(rank_m1, size_m2):
    key = f"{rank_m1}-{size_m2}"
    return hashlib.sha256(key.encode()).hexdigest()


def convert_rank_to_note_m1(rank_m1):
    return 20.0 * (1.0 - (rank_m1 - 1) / 1798.0)


def convert_rank_to_note_m2(rank_m2, size):
    return 20.0 * (1.0 - (rank_m2 - 1) / (size - 1))


def collect_to_google_sheet(
    nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2
):
    try:
        sheet_id = st.secrets["GOOGLE_SHEET_KEY"]
        json_keyfile = dict(st.secrets)
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            json_keyfile, scope
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1

        user_hash = generate_user_hash(rank_m1, size_m2)
        rows = sheet.get_all_values()

        if not rows:
            header = [
                "Nom LAS",
                "Rang M1",
                "Rang M2",
                "Taille M2",
                "Note M1",
                "Note M2",
                "Hash",
            ]
            sheet.append_row(header)
        else:
            hashes = [row[-1] for row in rows[1:] if len(row) > 6]
            las_names = [row[0] for row in rows[1:] if len(row) > 0]
            if nom_las in las_names:
                if user_hash not in hashes:
                    st.error(
                        "🚫 Une tentative avec un autre classement a déjà été effectué. Envoyer une nouvelle demande de simulation à l'adminstrateur du site "
                    )
                    return False  # ne pas continuer

        # Ajouter la ligne avec le hash
        sheet.append_row(
            [nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, user_hash]
        )
        st.success("✅ Partager ce lien avec vos amis.")
        return True
    except Exception as e:
        st.error(f"Erreur : {e}")


st.title("Simulation de classement")
st.text(
    "Attention une seule simulation est autorisée . Saissez correctement votre rang de Pass et de LAS2"
)
rank_m1 = st.number_input(
    "🎓 Votre rang en PASS (sur 1799)", min_value=1, max_value=1799, value=100
)
nom_las = st.text_input("🏫 Nom de votre LAS", max_chars=100)
rank_m2 = st.number_input("🎓 Votre rang en LAS2", min_value=1, value=50)
size_m2 = st.number_input(
    "👥 Effectif total de votre LAS2", min_value=2, value=300
)
rang_souhaite = st.number_input(
    "🎯 Rang souhaité dans la promo (sur 884)",
    min_value=1,
    max_value=884,
    value=150,
)
rho = st.slider("🔗 Corrélation PASS / LASS", 0.7, 1.0, 0.85, step=0.05)
n = st.number_input(
    "🔁 Nombre de simulations", min_value=100, value=10000, step=100
)
n_workers = st.number_input("🧵 Threads", min_value=1, value=4)

show_graph = st.checkbox(
    "📈 Afficher le graphique de probabilité par rang", value=True
)

if st.button("Lancer la simulation"):
    note_m1 = convert_rank_to_note_m1(rank_m1)
    note_m2 = convert_rank_to_note_m2(rank_m2, size_m2)
    st.write(f"🧮 Note M1 estimée : {note_m1:.2f}")
    st.write(f"🧮 Note M2 estimée : {note_m2:.2f}")

    if collect_to_google_sheet(
        nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2
    ):  # Verifie que l'enregistrement a réussi et que l'utilisateur n'a pas déjà enregistré une simulation

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

            rhos = [0.8, 0.9, 1.0]
            ranks = list(
                range(
                    max(1, rang_souhaite - 50), min(884, rang_souhaite + 51), 2
                )
            )
            fig, ax = plt.subplots()

            progress_bar = st.progress(0)
            total_steps = len(rhos) * len(ranks)
            step = 0

            for r in rhos:
                pvals = []
                for target_rank in ranks:
                    p = simulate_student_ranking(
                        rang_souhaite=target_rank,
                        rho=r,
                        n_simulations=1000,
                        note_m1_perso=note_m1,
                        note_m2_perso=note_m2,
                        n_workers=n_workers,
                    )[0]
                    pvals.append(p)

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
        st.success(
            (
                f"📊 Probabilité d'être dans le top {rang_souhaite} avec ρ = {rho} : "
                f"{int(p * 100)}% ± {int(se * 100)}%"
            )
        )
