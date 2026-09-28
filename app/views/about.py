import streamlit as st

from common import (BURGUNDY, COURSE, MEETING, MODEL_INFO, MUTED, PRIORITIES, PROFESSOR, REPO, TEAM, TEAM_NAME,
                    assets_ready, footer, md, meta, page_header, pct)

if not assets_ready():
    st.stop()
m = meta()

# The pages of the app, in the order a first-time visitor would use them.
GUIDE = [
    ("views/memo.py", "Executive summary", ":material/description:",
     "Our recommendation, the headline figures and a side-by-side comparison of the three models."),
    ("views/risk_appetite.py", "Approval policy", ":material/tune:",
     "Change the margin, the loss rate or the cutoff and see profit, approvals and fairness move together."),
    ("views/explain.py", "Decision explanation", ":material/fact_check:",
     "Pick an application and read each model's decision with its principal reasons."),
    ("views/what_if.py", "Applicant scenarios", ":material/swap_horiz:",
     "Move one factor, such as income, and see how far each model's score changes."),
    ("views/stability.py", "Decision consistency", ":material/replay:",
     "Check whether the same application gets the same answer after the model is retrained."),
    ("views/methodology.py", "Methodology", ":material/menu_book:",
     "How the data was prepared and the models built, and what the results cannot tell you."),
    ("views/glossary.py", "Glossary", ":material/dictionary:",
     "Every metric used in the review, with its formula and our figures."),
]


def link(path: str, label: str, icon: str) -> None:
    try:
        st.page_link(path, label=label, icon=icon)
    except st.errors.StreamlitPageNotFoundError:  # page run on its own, outside app.py's navigation
        st.markdown(f"**{label}**")


page_header(
    "About this review",
    "ApexLend wants a model to decide its mortgage applications. We tested three candidates against its five "
    "priorities, and this app lets you test them yourself.",
    "",
    [p for p, _ in PRIORITIES], meta_line=f"Credit model review · {MEETING} · {TEAM_NAME}", logo=True)

st.subheader("The client")
st.markdown(
    "ApexLend is a fast-growing fintech and an aggressive consumer lender. It wants a machine learning model to "
    "automate its credit decisions, provided the model stays within its risk tolerance, its operating constraints "
    "and fair lending law.\n\n"
    "The Credit Committee's question: **which of three candidate models, if any, should ApexLend deploy, "
    "and on what conditions?**")

st.markdown("**What ApexLend needs from the model**")
with st.container(border=True):
    st.markdown("".join(f"<div style='margin:0.15rem 0 0.45rem 0'><span style='color:{BURGUNDY};font-weight:600'>"
                        f"{name}.</span> {text}</div>" for name, text in PRIORITIES), unsafe_allow_html=True)

st.subheader("What we did")
steps = st.container(horizontal=True, gap="small")
with steps.container(border=True, width=350, height="stretch"):
    st.markdown("**1. The data**")
    st.markdown(f"ApexLend's own loan book is not available, so the 2025 national HMDA register stands in for it: "
                f"13.5 million applications, cleaned to 9.1 million with a clear approve or deny decision. Every model "
                f"is scored on the same test set of {m['n_test']:,} applications.")
with steps.container(border=True, width=350, height="stretch"):
    st.markdown("**2. The models**")
    st.markdown("\n".join(f"- **{MODEL_INFO[k]['name']}**, {MODEL_INFO[k]['blurb']}" for k in MODEL_INFO)
                + "\n\nNo model sees race, ethnicity, sex or age. They are kept only to audit fairness.")
with steps.container(border=True, width=350, height="stretch"):
    st.markdown("**3. The test**")
    st.markdown(md(f"HMDA records the lender's decisions, not repayments, so the models learn to reproduce the "
                   f"lender's own decisions ({pct(m['lender_approval_rate'])} approved). Each model is judged on "
                   f"the five priorities at a cutoff of {pct(m['threshold'])}, the score at which an approval "
                   f"breaks even under a {m['margin']:.0%} margin and a {m['lgd']:.0%} loss rate."))

st.subheader("How to use this app")
st.markdown("Read the recommendation, set the cutoff on Approval policy, then pick one application and follow it "
            "through the next three pages. Your cutoff and application carry across "
            "pages and show in the sidebar.")
with st.container(horizontal=True, gap="small"):
    for path, label, icon, text in GUIDE:
        with st.container(border=True, width=260, height="stretch"):
            link(path, label, icon)
            st.caption(text)

st.subheader("The team")
st.markdown(f"**{TEAM_NAME}**, for the {COURSE} course taught by {PROFESSOR}.")
cols = st.columns(3)
for i, name in enumerate(TEAM):
    cols[i % 3].markdown(f"<div style='padding:0.2rem 0'>{name}</div>", unsafe_allow_html=True)
st.markdown(f"<span style='color:{MUTED};font-size:0.85rem'>Code and notebooks: "
            f"<a href='{REPO}'>{REPO.removeprefix('https://')}</a></span>", unsafe_allow_html=True)
footer()
