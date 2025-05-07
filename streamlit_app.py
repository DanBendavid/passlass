from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components

COOKIE = "simu_lock"

st.title("🔐 Cookie test sur Streamlit Cloud")

# Étape 1 : Champ caché pour recevoir la valeur du cookie
cookie_val = st.text_input(
    "hidden_cookie_field", value="", key=COOKIE, label_visibility="collapsed"
)

# Étape 2 : JS injecté avec accès au document parent
components.html(
    f"""
    <script>
    setTimeout(() => {{
        const raw = document.cookie.split('; ').find(r => r.startsWith('{COOKIE}='));
        if (!raw) return;

        const value = decodeURIComponent(raw.split('=')[1]);

        // Monte jusqu'au document parent
        let root = window;
        while (root !== root.parent) root = root.parent;

        const el = root.document.querySelector('input[data-streamlit-key="{COOKIE}"]');
        if (el && el.value !== value) {{
            el.value = value;
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
    }}, 300);
    </script>
    """,
    height=0,
)

# Étape 3 : Si valeur injectée mais non encore traitée → rerun
if st.session_state.get("cookie_processed") is None and cookie_val:
    st.experimental_rerun()

# Étape 4 : Si valeur disponible → la traiter
if (
    st.session_state.get("cookie_processed") is None
    and cookie_val
    and cookie_val.count("-") == 3
):
    st.session_state["cookie_processed"] = True
    st.session_state["parsed_cookie"] = cookie_val
    st.experimental_rerun()

# Affichage
st.subheader("📦 Cookie lu")
st.write("cookie_val seen by Python:", repr(cookie_val))
st.write("session_state:", st.session_state)


# Ajout d’un bouton pour créer un cookie côté client
def set_cookie(val: str):
    exp = (datetime.utcnow() + timedelta(days=60)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    components.html(
        f"""
        <script>
          document.cookie = "{COOKIE}=" + encodeURIComponent("{val}") +
                            "; expires={exp}; path=/; SameSite=Lax;";
          alert("✅ Cookie défini : {val}");
        </script>
        """,
        height=0,
    )


if st.button("Créer un cookie de test"):
    set_cookie("123-45-300-JDoe")
