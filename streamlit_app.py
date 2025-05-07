from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components

COOKIE = "simu_lock"

st.title("🔐 Test cookie verrouillé (compatible Streamlit Cloud)")

# ─── 1. Champ caché pour recevoir la valeur du cookie ───
cookie_val = st.text_input(
    "hidden_cookie_field", value="", key=COOKIE, label_visibility="collapsed"
)

# ─── 2. JS pour injecter le cookie dans le champ caché ───
components.html(
    f"""
    <script>
    setTimeout(() => {{
        const raw = document.cookie.split('; ').find(r => r.startsWith('{COOKIE}='));
        if (!raw) return;

        const value = decodeURIComponent(raw.split('=')[1]);

        // Monter jusqu’au document racine (iframe → parent)
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

# ─── 3. Forcer rerun quand le cookie a été injecté ───
if st.session_state.get("cookie_processed") is None and cookie_val:

    @st.experimental_singleton
    def trigger_rerun_once():
        st.experimental_rerun()

    trigger_rerun_once()

# ─── 4. Traitement de la valeur du cookie une fois injectée ───
if (
    st.session_state.get("cookie_processed") is None
    and cookie_val
    and cookie_val.count("-") == 3
):
    st.session_state["cookie_processed"] = True
    st.session_state["parsed_cookie"] = cookie_val
    st.experimental_rerun()

# ─── 5. Affichage des infos de debug ───
st.subheader("📦 Cookie injecté")
st.write("cookie_val seen by Python:", repr(cookie_val))
st.write("session_state:", st.session_state)


# ─── 6. Bouton pour créer un cookie de test côté navigateur ───
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


if st.button("📦 Créer un cookie de test"):
    set_cookie("123-45-300-Danb")
