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

from simulation import simulate_student_ranking

# ─── Config de la page ──────────────────────────────────────────────────────
st.set_page_config(page_title="Simulation de classement", layout="centered")

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
def generate_user_hash(rank_m1, size_m2):
    key = f"{rank_m1}-{size_m2}"
    return hashlib.sha256(key.encode()).hexdigest()


def convert_rank_to_note_m1(rank_m1):
    return 20.0 * (1.0 - (rank_m1 - 1) / 1798.0)


def convert_rank_to_note_m2(rank_m2, size):
    return 20.0 * (1.0 - (rank_m2 - 1) / (size - 1))


def collect_to_google_sheet(
    nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, rang_souhaite
):
    # ... (votre code inchangé pour Google Sheets) ...
    return True  # ou False si déjà soumis


def afficher_rho_empirique():
    # ... (votre code inchangé pour corrélation empirique) ...
    return


# ─── 4. UI principale ───────────────────────────────────────────────────────
st.title("Simulation de classement")
st.text(
    "⚠️ Une seule simulation par utilisateur. Les champs seront verrouillés."
)

# Valeurs verrouillées si présentes dans session_state
rank_m1_locked = st.session_state.get("rank_m1_locked") or 0
rank_m2_locked = st.session_state.get("rank_m2_locked") or 0
size_m2_locked = st.session_state.get("size_m2_locked") or 0
nom_las_locked = st.session_state.get("nom_las_locked") or ""

rank_m1 = st.number_input(
    "🎓 Rang PASS (1–1799)",
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
    "👥 Taille LAS2",
    min_value=2,
    value=size_m2_locked or 300,
    disabled=bool(size_m2_locked),
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

rho = st.slider("🔗 Corrélation PASS / LAS", 0.7, 1.0, 0.85, step=0.05)
n = st.number_input("🔁 Nombre de simulations", 100, 20000, 10000, step=1000)
n_workers = 4
show_graph = st.checkbox("📈 Afficher graphique", value=True)

if st.button("Lancer la simulation"):
    note_m1 = convert_rank_to_note_m1(rank_m1)
    note_m2 = convert_rank_to_note_m2(rank_m2, size_m2)
    st.write(f"🧮 Note M1 : {note_m1:.2f}")
    st.write(f"🧮 Note M2 : {note_m2:.2f}")

    if collect_to_google_sheet(
        nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, rang_souhaite
    ):
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

        # Relancer pour prendre en compte le verrouillage
        st.rerun()

# Section corrélation empirique en bas
st.subheader("🔗 Corrélation empirique M1 vs M2")
if st.button("Calculer le ρ empirique"):
    afficher_rho_empirique()
