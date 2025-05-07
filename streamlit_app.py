# demo_cookies.py

import os

import streamlit as st
from st_cookies_manager import EncryptedCookieManager

# ─── Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Démo Cookies Manager", layout="centered")

# Récupérez votre mot de passe depuis les Secrets (Créez COOKIES_PASSWORD dans Cloud)
COOKIE_PASSWORD = os.environ.get("COOKIES_PASSWORD", "changeme_en_local")

# Préfixe unique pour éviter les collisions entre apps
COOKIE_PREFIX = "demo_app/"

# Nom de notre cookie
COOKIE_NAME = "user_preference"

# ─── Initialisation du gestionnaire ────────────────────────────────────────
cookies = EncryptedCookieManager(
    prefix=COOKIE_PREFIX,
    password=COOKIE_PASSWORD,
)

# Le composant JS initialise le store des cookies avant toute lecture/écriture.
if not cookies.ready():
    st.write("⌛ Chargement des cookies…")
    st.stop()

# ─── Interface utilisateur ─────────────────────────────────────────────────
st.title("🔐 Démo st-cookies-manager")

# Afficher tous les cookies existants
st.subheader("Cookies actuels")
st.write(dict(cookies))

# Lecture d’un cookie spécifique
current_val = cookies.get(COOKIE_NAME, "non défini")
st.write(f"Valeur de `{COOKIE_NAME}` : **{current_val}**")

# Form pour modifier le cookie
new_val = st.text_input("Nouvelle valeur pour le cookie", value=current_val)

col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Enregistrer le cookie"):
        cookies[COOKIE_NAME] = new_val
        cookies.save()  # prend effet au prochain rerun
        st.experimental_rerun()

with col2:
    if st.button("❌ Supprimer le cookie"):
        cookies.pop(COOKIE_NAME, None)
        cookies.save()
        st.experimental_rerun()

# ─── Debug (facultatif) ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Debug session_state")
st.write(st.session_state)
