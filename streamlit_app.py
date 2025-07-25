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

from plot import graph_student_ranking
from simulation import simulate_student_ranking

size_pass = 1799


# ─── 1. Config de la page ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulation de classement V1.5 (09/06/2025)",
    layout="centered",
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
# ________Fin   de la barre de navigation___________


# ─── 3. Fonctions utilitaires (unchanged) ───────────────────────────────────
def generate_user_hash(rank_m1, size_m2):
    key = f"{rank_m1}-{size_m2}"
    return hashlib.sha256(key.encode()).hexdigest()


def rank_to_note(rank, size):
    return 20.0 * (1.0 - (rank - 1) / (size - 1))


def collect_to_google_sheet(
    nom_las,
    rank_m1,
    rank_m2,
    size_m2,
    note_m1,
    note_m2,
    rank_souhaite,
    rank_fifty,
    size_las1=None,
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
                "Size LAS1",
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
                rang_souhaite,
                timestamp,
                rank_fifty,
                size_las1,
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
                f"🔗 Corrélation empirique ρ entre notes PASS et LAS : **{rho_e:.3f}** calculé avec {len(df)} notes. la significativité {p}"
            )
    #        for i, (m1, m2) in enumerate(zip(df["note m1"], df["note m2"])):
    #            st.write(f"Ligne {i+1}: M1 = {m1}, M2 = {m2}")
    except Exception as e:
        st.error(f"Erreur lors du calcul de la corrélation empirique : {e}")


@st.cache_data(show_spinner="⏳ Simulation…")
def run_simulation(
    n_simulations, rang_souhaite, note_m1, note_m2, rho, n_workers
):
    return simulate_student_ranking(
        n_simulations=n_simulations,
        rang_souhaite=rang_souhaite,
        note_m1_perso=note_m1,
        note_m2_perso=note_m2,
        rho=rho,
        n_workers=n_workers,
    )


# --- Page Accueil -------------------------------------------------------------
if choix_page == "Accueil":
    st.title("🏠 Bienvenue dans l'application de simulation")
    st.markdown(
        """
        Cette application vous permet de simuler votre classement en LAS 2/LAS3 en 
        fonction de votre rang PASS ou de votre note LAS 1 ou LAS 2 .
        """
    )

    st.text("Seules les LAS Sciences sont disponibles pour le moment.")

    licences_sciences = [
        "Chimie",  # 30 HYP
        "Maths et applications",  # 31 HYP
        "MEDPHY",  # 32 OK
        "MIASHS",  # 33 HYP
        "Physique",  # 34 HYP
        "Sciences Biomédicales",  # 289 OK
        "Sciences de la Vie",  # 35 HYP
        "Sciences de la Vie de la Terre (IPGP)",  # 36 HYP
        "SIAS",  # 254 OK
        "STAPS",  # 37 HYP
    ]
    licences_sciences_size = [
        20,  # Chimie
        20,  # Maths et applications
        32,  # MEDPHY
        25,  # MIASHS
        15,  # Physique
        289,  # Sciences Biomédicales
        30,  # Sciences de la Vie
        25,  # Sciences de la Vie de la Terre (IPGP)
        254,  # SIAS
        40,  # STAPS
    ]
    licences_dict = dict(zip(licences_sciences, licences_sciences_size))

    license_name = st.selectbox("Choisissez votre licence :", licences_sciences)

    license_size = licences_dict[license_name]
    st.session_state["license_name"] = license_name
    st.session_state["license_size"] = license_size

    st.success(f"Vous avez sélectionné : **{license_name}**")
    st.markdown(
        "Sélectionnez votre situation dans la barre de navigation à gauche."
    )
    st.markdown(
        """
    <div style='border: 1px solid #f1c40f; padding: 10px; border-radius: 5px; background-color: #fcf8e3; color: #8a6d3b;'>
    ⚠️ <strong>Disclaimer :</strong><br>
    Les données présentées dans cette application sont fournies à titre informatif et ne sauraient engager la responsabilité de l’auteur.<br>
    Pour des informations officielles, veuillez consulter les sites des universités concernées.
    </div>
    """,
        unsafe_allow_html=True,
    )
# --- Page PASS LAS2 ----------------------------------------------------------
elif choix_page == "PASS LAS2":
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
    st.title("🧮 Simulation PASS → LAS 2")
    st.text(
        "⚠️ Les champs Rang PASS et LAS seront verrouillés apres votre premiere simulation. Vous pouvez changer la correlation ainsi que le rang estimé."
    )

    # Valeurs verrouillées si présentes dans session_state
    rank_m1_locked = st.session_state.get("rank_m1_locked") or 0
    rank_m2_locked = st.session_state.get("rank_m2_locked") or 0
    size_m2_locked = (
        st.session_state.get("license_size", "")
        or st.session_state.get("size_m2_locked")
        or 0
    )
    nom_las_locked = st.session_state.get(
        "license_name", ""
    ) or st.session_state.get("nom_las_locked")

    with st.form("pass_las2"):
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
        )

        rank_m2 = st.number_input(
            "🎓 Rang LAS2",
            min_value=1,
            max_value=300,
            value=rank_m2_locked or 50,
            disabled=bool(rank_m2_locked),
        )

        rang_souhaite = st.number_input(
            "🎯 Rang souhaité (sur 884)",
            min_value=1,
            max_value=884,
            value=200,
        )

        rho_pl = st.slider(
            "🔗 Corrélation PASS / LAS", 0.65, 1.0, 0.85, step=0.05
        )

        n = st.number_input(
            "🔁 Nombre de simulations (Monte Carlo)",
            100,
            20000,
            10000,
            step=1000,
        )
        n_workers = 1

        show_graph = st.checkbox("📈 Afficher graphique", value=True)

        submitted = st.form_submit_button("Lancer la simulation")

    if submitted:
        note_m1 = rank_to_note(rank_m1, size_pass)
        note_m2 = rank_to_note(rank_m2, size_m2)
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

        # Simulation
        p, se = run_simulation(
            n_simulations=n,
            rang_souhaite=rang_souhaite,
            note_m1=note_m1,
            note_m2=note_m2,
            rho=rho_pl,
            n_workers=n_workers,
        )
        # Affichage de la probabilité
        if p > 0.5:
            st.success(
                f"📊 Probabilité d'être dans le top {rang_souhaite} avec ρ = {rho_pl} : {int(p * 100)}% ± {int(se * 100)}%"
            )
        else:
            st.warning(
                f"📊 Augmenter le rang cible car vos chance d'être dans le top {rang_souhaite} avec ρ = {rho_pl} sont inférieures à 50% [p ={int(p * 100)}% ± {int(se * 100)}%]"
            )

        # → Enregistrement du cookie chiffré
        cookies[COOKIE_NAME] = f"{rank_m1}-{rank_m2}-{size_m2}-{nom_las}"
        cookies.save()
        # Affichage du graphique
        if show_graph:
            st.subheader("📉 Probabilité autour du rang souhaité")

            rank_fifty = graph_student_ranking(
                rank_target=rang_souhaite,
                rho=rho_pl,
                n_simulations=n,
                note_m1=note_m1,
                note_m2=note_m2,
                n_workers=n_workers,
            )

        if collect_to_google_sheet(
            choix_page + nom_las,
            rank_m1,
            rank_m2,
            size_m2,
            note_m1,
            note_m2,
            rang_souhaite,
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
    st.info("Version 0.6 (9/6/2025) ")
    # placeholder : ajoutez ici vos widgets et votre logique

    with st.form("las1_las2"):

        nom_las = st.text_input(
            "🏫 Nom de votre LAS",
            max_chars=100,
            value=st.session_state.get("license_name", ""),
        )

        note_las1 = st.number_input(
            "📝 Note LAS1 (20.0)",
            min_value=0.00,
            max_value=20.00,
            step=0.01,
        )
        rank_las1 = st.number_input(
            "🎓 Rang LAS1",
            min_value=1,
            max_value=300,
        )
        size_las1 = st.number_input(
            "🎓 Taille LAS1",
            min_value=1,
            max_value=300,
        )

        size_m2 = st.number_input(
            "👥 Taille LAS2 (Attention, l'effetif de votre LAS doit etre saisi précisement)",
            min_value=2,
            value=st.session_state.get("license_size", 456),
        )

        rank_m2 = st.number_input(
            "🎓 Rang LAS2",
            min_value=1,
            max_value=300,
            value=300,
        )

        rang_souhaite = st.number_input(
            "🎯 Rang estimé",
            min_value=1,
            max_value=884,
        )

        rho_pl = st.slider(
            "🔗 Corrélation LAS 1 / LAS 2", 0.65, 1.0, 0.85, step=0.05
        )

        n = st.number_input(
            "🔁 Nombre de simulations (Monte Carlo)",
            100,
            20000,
            10000,
            step=1000,
        )
        n_workers = 1

        show_graph = st.checkbox("📈 Afficher graphique", value=True)

        submitted_las = st.form_submit_button(
            "Lancer la simulation LAS1 → LAS2"
        )

    if submitted_las:
        note_m1 = rank_to_note(rank_las1, size_las1)
        note_m2 = rank_to_note(rank_m2, size_m2)
        st.write(f"🧮 Note de Rang PASS : {note_m1:.2f}")
        st.write(f"🧮 Note de Rang LASS : {note_m2:.2f}")

        rank_fifty = None

        # Verrouillage en session
        st.session_state.update(
            {
                "rank_m1_locked": rank_las1,
                "rank_m2_locked": rank_m2,
                "size_m2_locked": size_m2,
                "nom_las_locked": nom_las,
            }
        )

        # Simulation
        p, se = run_simulation(
            n_simulations=n,
            rang_souhaite=rang_souhaite,
            note_m1=note_m1,
            note_m2=note_m2,
            rho=rho_pl,
            n_workers=n_workers,
        )
        # Affichage de la probabilité
        if p > 0.5:
            st.success(
                f"📊 Probabilité d'être dans le top {rang_souhaite} avec ρ = {rho_pl} : {int(p * 100)}% ± {int(se * 100)}%"
            )
        else:
            st.warning(
                f"📊 Augmenter le rang cible car vos chance d'être dans le top {rang_souhaite} avec ρ = {rho_pl} sont inférieures à 50% [p ={int(p * 100)}% ± {int(se * 100)}%]"
            )

        # Affichage du graphique
        if show_graph:
            st.subheader("📉 Probabilité autour du rang souhaité")

            rank_fifty = graph_student_ranking(
                rank_target=rang_souhaite,
                rho=rho_pl,
                n_simulations=n,
                note_m1=note_m1,
                note_m2=note_m2,
                n_workers=n_workers,
            )

        if collect_to_google_sheet(
            choix_page + nom_las,
            rank_las1,
            rank_m2,
            size_m2,
            note_m1,
            note_m2,
            rang_souhaite,
            rank_fifty,
            size_las1,
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


# --- Page LAS2 LAS3 ----------------------------------------------------------
elif choix_page == "LAS2 LAS3":
    st.title("➡️ Simulation LAS2 → LAS3")
    st.info("Experimental Version (13.05.2025).")
