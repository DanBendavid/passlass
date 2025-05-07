# demo_cookies.py

import os
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

# ─── Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Démo Cookies Manager", layout="centered")

COOKIE_PASSWORD = os.environ.get("COOKIES_PASSWORD", "changeme_en_local")
COOKIE_PREFIX   = "demo_app/"
COOKIE_NAME     = "user_preference"

# ─── Initialisation ─────────────────────────────────────────────────────────
cookies = EncryptedCookieManager(
    prefix=COOKIE_PREFIX,
    password=COOKIE_PASSWORD,
)
if not cookies.ready():
    st.write("⌛ Chargement des cookies…")
    st.stop()

# ─── UI ─────────────────────────────────────────────────────────────────────
st.title("🔐 Démo streamlit-cookies-manager")

st.subheader("Cookies actuels")
st.write(dict(cookies))

current_val = cookies.get(COOKIE_NAME, "non défini")
st.write(f"Valeur de `{COOKIE_NAME}` : **{current_val}**")

new_val = st.text_input("Nouvelle valeur pour le cookie", value=current_val)

col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Enregistrer le cookie"):
        cookies[COOKIE_NAME] = new_val
        cookies.save()
        st.experimental_rerun()
with col2:
    if st.button("❌ Supprimer le cookie"):
        cookies.pop(COOKIE_NAME, None)
        cookies.save()
        st.experimental_rerun()

st.markdown("---")
st.subheader("Debug session_state")
st.write(st.session_state)
