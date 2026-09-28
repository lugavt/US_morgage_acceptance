"""ApexLend · Credit Model Review: the model review workspace for ApexLend's Credit Committee.

Run locally with `streamlit run app/app.py` from the repo root (the theme lives in .streamlit/).
Every page reads precomputed files from app/assets/ (built by notebooks/app_assets.ipynb), so no
model runs here.
"""
import streamlit as st

from common import APP_NAME, APP_SUBTITLE, ASSETS, BURGUNDY, MUTED, app_ref, meta, pct

st.set_page_config(page_title=APP_NAME, page_icon=":material/account_balance:", layout="wide")
# Inter everywhere (registered by .streamlit/config.toml); headings need the explicit rule.
st.markdown("<style>h1, h2, h3 {font-family: 'Inter', sans-serif !important; letter-spacing: -0.02em;}</style>",
            unsafe_allow_html=True)

start = [st.Page("views/about.py", title="About this review", icon=":material/info:", default=True)]
review = [
    st.Page("views/memo.py", title="Executive summary", icon=":material/description:",
            url_path="executive-summary"),
    st.Page("views/risk_appetite.py", title="Approval policy", icon=":material/tune:",
            url_path="approval-policy"),
    st.Page("views/explain.py", title="Decision explanation", icon=":material/fact_check:",
            url_path="decision-explanation"),
    st.Page("views/what_if.py", title="Applicant scenarios", icon=":material/swap_horiz:",
            url_path="applicant-scenarios"),
    st.Page("views/stability.py", title="Decision consistency", icon=":material/replay:",
            url_path="decision-consistency"),
]
reference = [st.Page("views/methodology.py", title="Methodology and limitations", icon=":material/menu_book:",
                     url_path="methodology"),
             st.Page("views/glossary.py", title="Glossary", icon=":material/dictionary:", url_path="glossary")]
page = st.navigation(start + review + reference, position="hidden")
st.logo(str(ASSETS / "hec_paris_logo.png"), size="large")

with st.sidebar:
    st.markdown(f"<div style='font-family:Inter,sans-serif;font-weight:600;font-size:1.15rem;line-height:1.3;letter-spacing:-0.01em'>{APP_NAME}</div>"
                f"<div style='color:{MUTED};font-size:0.85rem;margin-bottom:0.8rem'>{APP_SUBTITLE}</div>",
                unsafe_allow_html=True)
    for p in start:
        st.page_link(p)
    st.caption("Review")
    for p in review:
        st.page_link(p)
    st.caption("Reference")
    for p in reference:
        st.page_link(p)

page.run()

# Drawn after the page so it shows the values the page has just set.
with st.sidebar:
    with st.container(border=True):
        st.markdown(f"<span style='color:{BURGUNDY};font-size:0.8rem'>CURRENT SETTINGS</span>",
                    unsafe_allow_html=True)
        try:
            cutoff = st.session_state.get("threshold", meta()["threshold"])
            st.markdown(f"Approval cutoff: **{pct(cutoff)}**")
        except FileNotFoundError:
            st.markdown("Approval cutoff: **not set**")
        row_nr = st.session_state.get("row_nr")
        st.markdown(f"Application: **{app_ref(row_nr)[12:] if row_nr is not None else 'none selected'}**")
