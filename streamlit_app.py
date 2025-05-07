# cookie_demo.py
from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components

COOKIE = "simu_lock"

st.title("🔒 Cookie lock demo")

# ─── ① hidden input that will eventually hold the cookie value ───
cookie_val = st.text_input(
    "hidden_cookie_field",
    value="",
    key=COOKIE,
    label_visibility="collapsed",
)

# ─── ② JS → writes cookie value into the hidden input ───
#  – climbs up to the top document, so it works no matter how many iframes
components.html(
    f"""
    <script>
    setTimeout(() => {{
        const raw = document.cookie.split('; ').find(r => r.startsWith('{COOKIE}='));
        if (!raw) return;

        const value = decodeURIComponent(raw.split('=')[1]);

        // IMPORTANT : accéder au document parent pour Streamlit Cloud
        let root = window;
        while (root !== root.parent) root = root.parent;

        const el = root.document.querySelector('input[data-streamlit-key="{COOKIE}"]');
        if (el && el.value !== value) {{
            el.value = value;
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
    }}, 300);  // assez de délai pour que le champ soit monté
    </script>
    """,
    height=0,
)

# ─── ③ FIRST pass: cookie just injected → force a rerun ───
if st.session_state.get("cookie_processed") is None and cookie_val:
    st.experimental_rerun()

# ─── ④ SECOND pass: we can finally use the value ───
if (
    st.session_state.get("cookie_processed") is None
    and cookie_val
    and cookie_val.count("-") == 3
):
    r1, r2, sz, nom = cookie_val.split("-")
    st.session_state.update(
        dict(
            rank_m1_locked=int(r1),
            rank_m2_locked=int(r2),
            size_m2_locked=int(sz),
            nom_las_locked=nom,
            cookie_processed=True,
        )
    )
    st.experimental_rerun()

# ─── ⑤ Show what we got ───
st.write("`cookie_val` seen by Python:", repr(cookie_val))
st.write("session_state:", st.session_state)

# ─── Utility to (re)create a 60-day cookie so you can test quickly ───
from datetime import datetime, timedelta


def set_cookie(val: str):
    exp = (datetime.now() + timedelta(days=60)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    components.html(
        f"""
        <script>
          document.cookie = "{COOKIE}=" + encodeURIComponent("{val}") +
                            "; expires={exp}; path=/; SameSite=Lax;";
          alert("Cookie set to: " + "{val}");
        </script>
        """,
        height=0,
    )


if st.button("📦 Créer / remplacer le cookie de test"):
    set_cookie("123-45-300-JDoe")
