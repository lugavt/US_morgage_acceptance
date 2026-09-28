import pandas as pd
import streamlit as st

from common import (ASSETS, BURGUNDY, DECK_STABILITY, EXPLAINABILITY, LENDER, MEETING, MODEL_INFO, PRIORITIES,
                    RECOMMENDED, TEAM_NAME, assets_ready, at_threshold, bn, consistency_rating, fairness, footer,
                    glossary_link, in_text, load, md, meta, page_header, pct, so_what, unchanged_share)

if not assets_ready():
    st.stop()
m = meta()
grid = load("grid")
t = m["threshold"]
oracle = m["margin"] * m["amt_pos_total"]
keys = [k for k in MODEL_INFO if k in m["models"]]
pnl_of = {k: m["models"][k]["pnl"] for k in keys}
rivals = [k for k in keys if k != RECOMMENDED]
runner_up = max(rivals, key=pnl_of.get) if rivals else None
lead = pnl_of[RECOMMENDED] / pnl_of[runner_up] - 1 if runner_up and RECOMMENDED in pnl_of else None
race = fairness(at_threshold(grid[grid["attribute"] == "derived_race"], t), "White")
race = race[race["group"] == "Black or African American"].set_index("model")["impact_ratio"]
unchanged = unchanged_share(t)

# Recommendation, as on slide 23 of the deck, with the verified figures.
RECOMMENDATION = {
    "headline": "XGBoost",
    "points": [
        ("Recommended: XGBoost, conditional on fairness remediation before deployment.",
         f"It earns the most at the shared {t:.3f} cutoff ({bn(pnl_of[RECOMMENDED])}, "
         f"{lead:.0%} more than {in_text(runner_up)} and {pnl_of[RECOMMENDED] / oracle:.0%} of the lender's own "
         "P&L), and its SHAP explanations are exact for each applicant, which supports adverse action reasons."
         if lead is not None else "It earns the most at the shared cutoff."),
        ("No model can be deployed as it stands.",
         "All three fall below the four-fifths line on race, and none is statistically equivalent within ±0.02 in "
         "the TOST test."),
        ("Logistic regression is not viable.",
         f"It approves too few applicants ({pct(m['models']['logreg']['approval_rate'])}) to earn a meaningful "
         "return, although its explanations are the easiest to read." if "logreg" in m["models"] else
         "It approves too few applicants to earn a meaningful return."),
        ("TabPFN is not recommended.",
         "Its accuracy is close and its drivers are steadier, but its answer for an applicant depends on the sample "
         "of context rows it reads, so the same applicant could get a different answer. Its SHAP values are "
         "approximations (KernelExplainer), so individual reasons are harder to defend."),
    ],
    "fix": "Before go-live: remediate the race gap for XGBoost (disparate impact ratio "
           f"{race.get(MODEL_INFO[RECOMMENDED]['short'], float('nan')):.2f}, against the four-fifths line of 0.80).",
}
DEPLOYABLE = {"xgboost": "Not yet: fairness remediation", "logreg": "No: approves too few",
              "tabpfn": "No: reproducibility, approximate SHAP"}

fails_all = all(race.get(MODEL_INFO[k]["short"], 1) < 0.8 for k in keys)
page_header(
    "Executive summary",
    f"At the shared {t:.3f} cutoff, XGBoost earns {bn(pnl_of[RECOMMENDED])}, {lead:.0%} more than "
    f"{in_text(runner_up)}, but " + ("no model passes the four-fifths test on race." if fails_all
                                     else "the race gap still needs review."),
    "", [p for p, _ in PRIORITIES],
    meta_line=f"Credit model review · {MEETING} · {TEAM_NAME}", logo=True)

with st.container(border=True):
    st.markdown(f"**Our recommendation: {RECOMMENDATION['headline']}**")
    st.markdown(md("\n".join(f"- **{head}** {body}" for head, body in RECOMMENDATION["points"])))
    st.markdown(md(f"<span style='color:{BURGUNDY}'>{RECOMMENDATION['fix']}</span>"), unsafe_allow_html=True)

short = MODEL_INFO[RECOMMENDED]["short"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("XGBoost P&L at 0.921", bn(pnl_of[RECOMMENDED]))
c2.metric("Share of the lender's P&L", pct(pnl_of[RECOMMENDED] / oracle, 0))
c3.metric("Approval rate", pct(m["models"][RECOMMENDED]["approval_rate"]))
c3.caption(f"Lender: {pct(m['lender_approval_rate'])}")
if short in race.index:
    c4.metric("Disparate impact, race", f"{race[short]:.2f}")
    c4.caption(f"Black vs White; lender {race[LENDER]:.2f}")

st.subheader("Model comparison")
col = {k: MODEL_INFO[k]["name"] for k in keys}
rows = [
    ("Financial", "P&L at 0.921", lambda k: bn(pnl_of[k]), bn(oracle)),
    ("Financial", "Share of the lender's P&L", lambda k: pct(pnl_of[k] / oracle, 0), "100%"),
    ("Financial", "Approval rate", lambda k: pct(m["models"][k]["approval_rate"]), pct(m["lender_approval_rate"])),
    ("Performance", "ROC-AUC", lambda k: f"{m['models'][k]['auc']:.3f}", "n/a"),
    ("Fairness", "Disparate impact, Black vs White",
     lambda k: f"{race.get(MODEL_INFO[k]['short'], float('nan')):.2f}", f"{race[LENDER]:.2f}"),
    ("Interpretability", "Explainability", lambda k: EXPLAINABILITY[k][0], "n/a"),
    ("Stability", "Rank distance across refits", lambda k: f"{DECK_STABILITY['rank'][k]:.1f}", "n/a"),
    ("Stability", "Decisions unchanged, 10 retrains",
     lambda k: f"{consistency_rating(unchanged.get(k))} ({pct(unchanged[k], 0)})" if k in unchanged else "n/a", "n/a"),
    ("", "Deployable as it stands", lambda k: DEPLOYABLE[k], "In place today"),
]
table = pd.DataFrame([{"Priority": p, "Measure": measure, **{col[k]: fn(k) for k in keys}, LENDER: bench}
                      for p, measure, fn, bench in rows])
st.dataframe(table, hide_index=True, width="stretch", row_height=36)
st.caption(f"Test set of {m['n_test']:,} applications, all models at {t:.3f}. Rows follow ApexLend's priorities. "
           "Rank distance: lower means steadier drivers.")
glossary_link("Definitions and rating rules in the Glossary")

with st.expander("ApexLend's priorities (slide 3)"):
    st.markdown("\n".join(f"{i}. **{name}.** {text}" for i, (name, text) in enumerate(PRIORITIES, 1)))

charts = {
    "financial_pnl_test.png": "P&L on the test set at each model's cutoff",
    "interpretability_shap_comparison.png": "The inputs each model relies on most (SHAP)",
    "interpretability_surrogate_fidelity_comparison.png": "How closely a depth-4 tree mimics each model",
    "fairness_disparate_impact_pair.png": "Disparate impact ratio by protected group",
    "fairness_equalized_odds_pair.png": "Equalized odds difference by protected group",
    "stability_bootstrap_distributions.png": "Spread of results across bootstrap refits",
}
available = [(f, c) for f, c in charts.items() if (ASSETS / "charts" / f).exists()]
if available:
    with st.expander("Supporting charts from the evaluation notebooks"):
        for file, title in available:
            st.markdown(f"**{title}**")
            st.image(str(ASSETS / "charts" / file))

with st.expander("Data and scope"):
    st.markdown("The 2025 national HMDA file stands in for ApexLend's book. The models predict the lender's "
                "decision, not default, so P&L rests on assumed margin and loss rates. Details in Methodology.")

so_what("XGBoost is the only candidate worth taking forward, and only once its race gap is fixed.")
footer()
