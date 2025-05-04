import hashlib
import json
from datetime import datetime

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from oauth2client.service_account import ServiceAccountCredentials
from scipy.stats import pearsonr

def set_cookie(name, value):
    script = f"""
        <script>
        document.cookie = "{name}={value};path=/;SameSite=Lax;";
        </script>
    """
    components.html(script, height=0)


# Crée un champ texte caché que le JS peut remplir
cookie_val = st.text_input(
    label="", value="", key="simu_lock_hidden", label_visibility="hidden"
)

# JS injecté qui lit le cookie et le met dans le champ invisible
components.html(
    f"""
    <script>
    const value = document.cookie
        .split('; ')
        .find(row => row.startsWith('simu_lock='))
        ?.split('=')[1];
    if (value !== undefined) {{
        const streamlitInput = window.parent.document.querySelector('input[data-streamlit-key="simu_lock"]');
        if (streamlitInput) {{
            streamlitInput.value = value;
            streamlitInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
    }}
    </script>
    """,
    height=0,
)

    
def get_cookie(name):
    script = f"""
        <script>
        const value = document.cookie
            .split('; ')
            .find(row => row.startsWith('{name}='))
            ?.split('=')[1];

        if (value) {{
            const streamlitInput = window.parent.document.querySelector('iframe');
            window.parent.postMessage({{
                isStreamlitMessage: true,
                type: 'streamlit:setComponentValue',
                key: '{name}',
                value: value
            }}, '*');
        }}
        </script>
    """
    components.html(script, height=0)
    return None  # Toujours None ici, la valeur sera injectée plus tard


# Init session state
for key in ["rank_m1_locked", "rank_m2_locked", "size_m2_locked", "nom_las_locked"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Si cookie lu, remplir session_state
if cookie_val and cookie_val.count("-") == 3:
    try:
        rank_m1_c, rank_m2_c, size_m2_c, nom_las_c = cookie_val.split("-")
        st.session_state["rank_m1_locked"] = int(rank_m1_c)
        st.session_state["rank_m2_locked"] = int(rank_m2_c)
        st.session_state["size_m2_locked"] = int(size_m2_c)
        st.session_state["nom_las_locked"] = nom_las_c
        st.info("🔒 Simulation déjà effectuée. Champs verrouillés.")
    except Exception as e:
        st.warning(f"⚠️ Cookie mal formé : {e}")


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
    nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, rank_souhaite
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
                "Rang souhaite",
                "Timestamp",
            ]
            sheet.append_row(header)
        else:
            hashes = [row[-1] for row in rows[1:] if len(row) > 6]
            if user_hash in hashes:
                st.error(
                    "🚫 Une tentative avec un autre classement a déjà été effectué. Envoyer une nouvelle demande de simulation à l'adminstrateur du site "
                )
                return False  # ne pas continuer
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Format du timestamp
        # Ajouter la ligne avec le hash
        sheet.append_row(
            [
                nom_las,
                rank_m1,
                rank_m2,
                size_m2,
                note_m1,
                note_m2,
                user_hash,
                rang_souhaite,
                timestamp,
            ]
        )
        st.success("✅ Partager ce lien avec vos amis.")
        return True
    except Exception as e:
        st.error(f"Erreur : {e}")


def afficher_rho_empirique():
    try:
        # Authentification Google Sheets
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

        # Lire les données
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Nettoyer les colonnes
        df.columns = df.columns.str.strip().str.lower()

        # Convertir les notes : remplacer la virgule par un point
        df["note m1"] = (
            df["note m1"].str.replace(",", ".", regex=False).astype(float)
        )
        df["note m2"] = (
            df["note m2"].str.replace(",", ".", regex=False).astype(float)
        )

        # Supprimer les lignes incomplètes
        df = df.dropna(subset=["note m1", "note m2"])

        if len(df) < 50:
            st.warning(
                f"📉 Pas assez de données [Progression : {int(len(df)/50*100)}% ] pour calculer une corrélation fiable. Invitez vos amis."
            )
            return False
        else:
            # st.subheader("📋 Données utilisées pour le calcul de ρ")
            #        st.dataframe(df[["note m1", "note m2"]])

            # Corrélation de Pearson
            rho_e, p = pearsonr(df["note m1"], df["note m2"])
            # rho_e = np.corrcoef(df["note m1"], df["note m2"])[0, 1]
            st.success(
                f"🔗 Corrélation empirique ρ entre notes M1 et M2 : **{rho_e:.3f}** calculé avec {len(df)} notes. la significativité {p}"
            )
    #        for i, (m1, m2) in enumerate(zip(df["note m1"], df["note m2"])):
    #            st.write(f"Ligne {i+1}: M1 = {m1}, M2 = {m2}")
    except Exception as e:
        st.error(f"Erreur lors du calcul de la corrélation empirique : {e}")


def afficher_rho_empirique_test():
    try:
        # Authentification Google Sheets
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

        # Lecture des données brutes
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        df.columns = df.columns.str.strip().str.lower()

        # Vérification de la présence des colonnes
        colonnes_attendues = {"note m1", "note m2"}
        if not colonnes_attendues.issubset(df.columns):
            st.error("❌ Colonnes 'Note M1' et 'Note M2' manquantes.")
            return False

        # Nettoyage des colonnes
        df["note m1"] = (
            df["note m1"].str.replace(",", ".", regex=False).astype(str)
        )
        df["note m2"] = (
            df["note m2"].str.replace(",", ".", regex=False).astype(str)
        )

        # Filtrer uniquement les lignes où les deux colonnes sont numériques
        df_clean = df[
            df["note m1"].str.replace(".", "", 1).str.isnumeric()
            & df["note m2"].str.replace(".", "", 1).str.isnumeric()
        ]
        df_clean["note m1"] = df_clean["note m1"].astype(float)
        df_clean["note m2"] = df_clean["note m2"].astype(float)

        st.subheader("🔎 Données nettoyées (utilisées pour le test)")
        st.dataframe(df_clean[["note m1", "note m2"]])

        st.write("📊 Types de données :")
        st.write(df_clean[["note m1", "note m2"]].dtypes)

        st.write("❓ Valeurs manquantes :")
        st.write(df_clean[["note m1", "note m2"]].isna().sum())

        # Affichage graphique
        fig, ax = plt.subplots()
        ax.scatter(df_clean["note m1"], df_clean["note m2"])
        ax.set_xlabel("Note M1")
        ax.set_ylabel("Note M2")
        ax.set_title("Nuage de points : Note M1 vs Note M2")
        st.pyplot(fig)

        if len(df_clean) < 3:
            st.warning(
                f"⚠️ Pas assez de données fiables pour une corrélation : {len(df_clean)} lignes valides"
            )
            return False

        # Corrélation via pandas
        rho = df_clean["note m1"].corr(df_clean["note m2"])
        st.success(f"✅ Corrélation testée avec `.corr()` : **{rho:.3f}**")

    except Exception as e:
        st.error(f"❌ Erreur pendant le test de corrélation : {e}")


st.title("Simulation de classement")

st.text(
    "Attention : une seule simulation avec votre rang PASS et LAS2 est autorisée. Ces champs seront verrouillés."
)

# Inputs verrouillés si déjà soumis
rank_m1 = st.number_input(
    "🎓 Votre rang en PASS (sur 1799)",
    min_value=1,
    max_value=1799,
    value=st.session_state["rank_m1_locked"] or 100,
    disabled=st.session_state["rank_m1_locked"] is not None,
)

nom_las = st.text_input(
    "🏫 Nom de votre LAS",
    value=st.session_state["nom_las_locked"] or "",
    max_chars=100,
    disabled=st.session_state["nom_las_locked"] is not None,
)

size_m2 = st.number_input(
    "👥 Effectif total de votre LAS2",
    min_value=2,
    value=st.session_state["size_m2_locked"] or 300,
    disabled=st.session_state["size_m2_locked"] is not None,
)

rank_m2 = st.number_input(
    "🎓 Votre rang en LAS2",
    min_value=1,
    max_value=size_m2,
    value=st.session_state["rank_m2_locked"] or 50,
    disabled=st.session_state["rank_m2_locked"] is not None,
)

rang_souhaite = st.number_input(
    "🎯 Rang estimé dans la promo -  Simuation autour de ce nombre (sur 884)",
    min_value=1,
    max_value=884,
    value=200,
)
rho = st.slider("🔗 Corrélation PASS / LASS", 0.7, 1.0, 0.85, step=0.05)
n = st.number_input(
    "🔁 Nombre de simulations",
    min_value=100,
    value=10000,
    max_value=20000,
    step=1000,
)
# n_workers = st.number_input("🧵 Threads", min_value=1, value=4)
n_workers = 4
show_graph = st.checkbox(
    "📈 Afficher le graphique de probabilité par rang", value=True
)

if st.button("Lancer la simulation"):
    note_m1 = convert_rank_to_note_m1(rank_m1)
    note_m2 = convert_rank_to_note_m2(rank_m2, size_m2)
    st.write(f"🧮 Note M1 estimée : {note_m1:.2f}")
    st.write(f"🧮 Note M2 estimée : {note_m2:.2f}")

    if collect_to_google_sheet(
        nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, rang_souhaite
    ):  # Verifie que l'enregistrement a réussi et que l'utilisateur n'a pas déjà enregistré une simulation

        cookie_str = f"{rank_m1}-{rank_m2}-{size_m2}-{nom_las}"
        set_cookie("simu_lock", cookie_str)

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
                    p_y = simulate_student_ranking(
                        rang_souhaite=target_rank,
                        rho=r,
                        n_simulations=1000,
                        note_m1_perso=note_m1,
                        note_m2_perso=note_m2,
                        n_workers=n_workers,
                    )[0]
                    pvals.append(p_y)

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
            f"📊 Probabilité d'être dans le top {rang_souhaite} avec ρ = {rho} : {int(p * 100)}% ± {int(se * 100)}%"
        )
    # Affichage du ρ empirique à la fin de la page
st.subheader("🔗 Corrélation empirique entre les notes M1 et M2")
if st.button("Calculer le ρ empirique"):
    afficher_rho_empirique()
