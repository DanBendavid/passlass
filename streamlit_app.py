# streamlit_app.py

import hashlib
import json
import os
from datetime import datetime, timedelta

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
from scipy.stats import pearsonr
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_option_menu import option_menu

from simulation import simulate_student_ranking

# ─── 1. Config de la page ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulation de classement V1.1 (13/05/2025)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. Barre de navigation ──────────────────────────────────────────────────
with st.sidebar:
    choix_page = option_menu(
        menu_title="Menu principal",
        options=["Accueil", "PASS LAS2", "LAS1 LAS2", "LAS2 LAS3"],
        icons=["house", "table", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "nav-link-selected": {"background-color": "#f0f0f0"},
        },
    )

    # ─── 3. Fonctions utilitaires (identiques à votre code) ──────────────────────
    rho = st.slider("🔗 Corrélation PASS / LAS", 0.65, 1.0, 0.85, step=0.05)
    n = st.number_input(
        "🔁 Nombre de simulations (Monte Carlo)", 100, 20000, 10000, step=1000
    )
    n_workers = 4
    show_graph = st.checkbox("📈 Afficher graphique", value=True)


def generate_user_hash(rank_m1, size_m2):
    key = f"{rank_m1}-{size_m2}"
    return hashlib.sha256(key.encode()).hexdigest()


def convert_rank_to_note_m1(rank_m1):
    return 20.0 * (1.0 - (rank_m1 - 1) / 1798.0)


def convert_rank_to_note_m2(rank_m2, size):
    return 20.0 * (1.0 - (rank_m2 - 1) / (size - 1))


def collect_to_google_sheet(
    nom_las,
    rank_m1,
    rank_m2,
    size_m2,
    note_m1,
    note_m2,
    rank_target,
    rank_fifty,
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
                "Rang 5050",
            ]
            sheet.append_row(header)
        else:
            header = rows[0]
            hash_idx = header.index(
                "Hash"
            )  # On trouve dynamiquement où est “Hash”
            hashes = [row[hash_idx] for row in rows[1:] if len(row) > 6]
            if user_hash in hashes:
                # st.error(
                #    "🚫 Une tentative avec un autre classement a déjà été effectué. Envoyer une nouvelle demande de simulation à l'adminstrateur du site "
                # )
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
                rank_target,
                timestamp,
                rank_fifty,
            ]
        )
        st.success("✅ Partager ce lien avec vos amis.")
        return True
    except Exception as e:
        st.error(f"Erreur : {e}")


# --- Page Accueil -------------------------------------------------------------
if choix_page == "Accueil":
    st.title("🏠 Bienvenue dans l'application de simulation")
    st.markdown(
        """
        Cette application vous permet de simuler votre classement en LAS 2/LAS3 en 
        fonction de votre rang PASS ou de votre note LAS 1 ou LAS 2 .
        """
    )
    st.markdown(
        "Sélectionnez votre situation dans la barre de navigation à gauche."
    )

# --- Page PASS LAS2 ----------------------------------------------------------
elif choix_page == "PASS LAS 2":
    st.title("🧮 Simulation PASS → LAS 2")
    st.text(
        "Les champs Rang PASS et LAS2 seront verrouillés après la première simulation."
    )
    # ─── Constantes et clés de session ─────────────────────────────────────────
    COOKIE_NAME = "simu_lock"
    PREFIX = "demo_app/"
    PASSWORD = os.environ.get("COOKIES_PASSWORD", "changeme_en_local")

    for k in (
        "rank_m1_locked",
        "rank_m2_locked",
        "size_m2_locked",
        "nom_las_locked",
        "cookie_processed",
    ):
        st.session_state.setdefault(k, None)

    # ─── 1. Initialisation du gestionnaire de cookies ─────────────────────────
    cookies = EncryptedCookieManager(prefix=PREFIX, password=PASSWORD)
    if not cookies.ready():
        st.write("⌛ Initialisation des cookies…")
        st.stop()

    # ─── 2. Lecture du cookie existant ─────────────────────────────────────────
    cookie_val = cookies.get(COOKIE_NAME, "")
    if cookie_val and st.session_state.get("cookie_processed") is None:
        # On ne fait le parsing qu'une seule fois
        parts = cookie_val.split("-")
        if len(parts) == 4:
            try:
                r1, r2, sz, nom = parts
                st.session_state.update(
                    {
                        "rank_m1_locked": int(r1),
                        "rank_m2_locked": int(r2),
                        "size_m2_locked": int(sz),
                        "nom_las_locked": nom,
                        "cookie_processed": True,
                    }
                )
                st.rerun()
            except ValueError:
                st.warning("Cookie mal formé ; ignoré.")
        else:
            st.warning("Cookie mal formé ; ignoré.")

    # ─── 3. Fonctions utilitaires (unchanged) ───────────────────────────────────

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
                    f"🔗 Corrélation empirique ρ entre notes PASS et LAS : **{rho_e:.3f}** calculé avec {len(df)} notes. la significativité {p}"
                )
        #        for i, (m1, m2) in enumerate(zip(df["note m1"], df["note m2"])):
        #            st.write(f"Ligne {i+1}: M1 = {m1}, M2 = {m2}")
        except Exception as e:
            st.error(f"Erreur lors du calcul de la corrélation empirique : {e}")

    # ─── 4. UI principale ───────────────────────────────────────────────────────
    st.title("Simulation de classement")
    st.text(
        "⚠️ Les champs Rang PASS et LAS seront verrouillés apres votre premiere simulation. Vous pouvez changer la correlation ainsi que le rang estimé."
    )

    # Valeurs verrouillées si présentes dans session_state
    rank_m1_locked = st.session_state.get("rank_m1_locked") or 0
    rank_m2_locked = st.session_state.get("rank_m2_locked") or 0
    size_m2_locked = st.session_state.get("size_m2_locked") or 0
    nom_las_locked = st.session_state.get("nom_las_locked") or ""

    rank_m1 = st.number_input(
        "🎓 Rang PASS 'non coefficienté' (1–1799)",
        min_value=1,
        max_value=1799,
        value=rank_m1_locked or 100,
        disabled=bool(rank_m1_locked),
        key="rank_m1_input",
    )

    nom_las = st.text_input(
        "🏫 Nom de votre LAS",
        value=nom_las_locked,
        max_chars=100,
        disabled=bool(nom_las_locked),
    )

    size_m2 = st.number_input(
        "👥 Taille LAS2 (Attention, l'effetif de votre LAS doit etre saisi précisement)",
        min_value=2,
        value=size_m2_locked or 456,
        disabled=bool(size_m2_locked),
    )

    rank_m2 = st.number_input(
        "🎓 Rang LAS2",
        min_value=1,
        max_value=300,
        value=rank_m2_locked or 50,
        disabled=bool(rank_m2_locked),
    )

    rank_target = st.number_input(
        "🎯 Rang souhaité (sur 884)",
        min_value=1,
        max_value=884,
        value=200,
    )

    if st.button("Lancer la simulation"):
        note_m1 = convert_rank_to_note_m1(rank_m1)
        note_m2 = convert_rank_to_note_m2(rank_m2, size_m2)
        st.write(f"🧮 Note de Rang PASS : {note_m1:.2f}")
        st.write(f"🧮 Note de Rang LASS : {note_m2:.2f}")

        rank_fifty = None

        # Verrouillage en session
        st.session_state.update(
            {
                "rank_m1_locked": rank_m1,
                "rank_m2_locked": rank_m2,
                "size_m2_locked": size_m2,
                "nom_las_locked": nom_las,
            }
        )

        # → Enregistrement du cookie chiffré
        cookies[COOKIE_NAME] = f"{rank_m1}-{rank_m2}-{size_m2}-{nom_las}"
        cookies.save()
        p, se = simulate_student_ranking(
            n_simulations=n,
            rang_souhaite=rang_souhaite,
            note_m1_perso=note_m1,
            note_m2_perso=note_m2,
            rho=rho,
            n_workers=n_workers,
        )

        if show_graph:
            st.subheader("📉 Probabilité autour du rang souhaité")

            rhos = [rho, 0.7, 1.0]

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

                    if rank_fifty is None and p_y > 0.5:
                        rank_fifty = target_rank

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
            else:
                st.warning(
                    f"📊 Pas de rang 50/50 trouvé avec ρ = {rho} dans la plage de simulation. Augmenter le rang cible"
                )
        # Affichage de la probabilité
        if p > 0.5:
            st.success(
                f"📊 Probabilité d'être dans le top {rang_souhaite} avec ρ = {rho} : {int(p * 100)}% ± {int(se * 100)}%"
            )
        else:
            st.warning(
                f"📊 Augmenter le rang cible car vos chance d'être dans le top {rang_souhaite} avec ρ = {rho} sont inférieures à 50% [p ={int(p * 100)}% ± {int(se * 100)}%]"
            )
        # Affichage du ρ empirique à la fin de la page
        # Relancer pour prendre en compte le verrouillage
        # st.rerun()

        if collect_to_google_sheet(
            nom_las,
            rank_m1,
            rank_m2,
            size_m2,
            note_m1,
            note_m2,
            rank_target,
            rank_fifty,
        ):
            st.success(
                f"Merci. Votre simulation a été enregistrée. Vous pouvez partager le lien avec vos amis."
            )
    # Section corrélation empirique en bas
    st.subheader(
        "🔗 Corrélation empirique entre votre rang en Pass et votre rang en LAS2"
    )
    if st.button("Calculer le ρ empirique"):
        afficher_rho_empirique()

# --- Page LAS1 LAS2 ----------------------------------------------------------
elif choix_page == "LAS1 LAS2":
    st.title("🔄 Simulation LAS1 → LAS2")
    st.info("Version 0.5 (13/05/2025) ")
    # placeholder : ajoutez ici vos widgets et votre logique

    nom_las = st.text_input(
        "🏫 Nom de votre LAS",
        max_chars=100,
    )

    note_m1 = st.number_input(
        "🎓 Note en LAS 1 en année 2023-2034 ",
        min_value=0,
        max_value=20,
    )

    rank_m1 = None

    size_m2 = st.number_input(
        "👥 Taille LAS 2 (Attention, l'effetif de votre LAS doit etre saisi précisement)",
        min_value=2,
    )

    rank_m2 = st.number_input(
        "🎓 Rang LAS2",
        min_value=1,
        max_value=300,
    )

    note_m2 = convert_rank_to_note_m2(rank_m2, size_m2)

    rank_fifty = None
    rank_target = st.number_input(
        "🎯 Rang souhaité ",
        min_value=1,
        max_value=884,
    )

    if st.button("Lancer la simulation"):
        if collect_to_google_sheet(
            nom_las,
            rank_m1,
            rank_m2,
            size_m2,
            note_m1,
            note_m2,
            rank_target,
            rank_fifty,
        ):
            st.success(
                f"Merci. Votre simulation a été enregistrée. Vous pouvez partager le lien avec vos amis."
            )

# --- Page LAS2 LAS3 ----------------------------------------------------------
elif choix_page == "LAS2 LAS3":
    st.title("➡️ Simulation LAS2 → LAS3")
    st.info("Experimental peu fiable.")
    # placeholder : ajoutez ici vos widgets et votre logique
