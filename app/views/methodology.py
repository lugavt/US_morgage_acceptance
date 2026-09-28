import pandas as pd
import streamlit as st

from common import (ASSETS, ATTRIBUTE_LABELS, EXPLAINABILITY, LABELS, MODEL_INFO, assets_ready, footer, glossary_link,
                    meta, page_header, pct)

if not assets_ready():
    st.stop()
m = meta()

page_header("Methodology and limitations",
            f"Every figure comes from one test set of {m['n_test']:,} applications, scored once by each model.",
            "",
            ["Financial", "Performance", "Fairness", "Interpretability", "Stability"])
glossary_link()

st.subheader("Data and scope")
st.markdown(
    "- **Source.** 2025 HMDA public Loan/Application Register, a market-wide proxy for ApexLend's own book.\n"
    f"- **Target.** The lender's decision: approved (originated or approved) or denied. The lender approved "
    f"{pct(m['lender_approval_rate'])} of test applications.\n"
    "- **Split.** 70% / 15% / 15%, stratified by outcome, race, ethnicity, sex and age.\n"
    "- **Excluded inputs.** Protected and geographic columns (kept for the fairness review only), the lender's "
    "identity (a proxy for race) and 12 leakage columns.")

st.subheader("The models")
st.markdown(
    "- **Logistic regression**, the white-box model: 12 inputs, scaled and one-hot encoded, L2 penalty (C = 0.01).\n"
    "- **XGBoost**, the machine-learning model: 31 inputs, 2,862 trees chosen by early stopping, tuned with Optuna, "
    "isotonic calibration.\n"
    "- **TabPFN**, the tabular foundation model (TabPFN-3.5): the same 31 inputs, 9,898 stratified context rows, "
    "4 estimators. Non-commercial licence; needs a GPU.")
with st.expander("The 30 raw inputs"):
    st.markdown(", ".join(sorted(LABELS.values())) + ".")

st.subheader("Rating rules in the scorecard")
st.markdown(
    "- **Explainability.** " + " ".join(f"{MODEL_INFO[k]['name']}: {v[0]}. {v[1]}" for k, v in EXPLAINABILITY.items())
    + "\n- **Decisions unchanged, 10 retrains.** Share of randomly sampled applications whose decision at 0.921 is "
      "the same under every retrain: Strong from 90%, Moderate from 70%, Limited below.\n"
      "- **Rank distance** is the deck's stability figure (stability notebook, 26 Sep run).")

framework = {"fairness_summary.csv": "Fairness summary", "fairness_detail.csv": "Fairness by protected group",
             "interpretability_summary.csv": "Interpretability summary"}
available = [(f, t) for f, t in framework.items() if (ASSETS / "framework" / f).exists()]
if available:
    with st.expander("Evaluation notebook outputs"):
        renames = {**{k: v["name"] for k, v in MODEL_INFO.items()}, **ATTRIBUTE_LABELS,
                   "co_applicant_age": "Co-applicant age", **LABELS}
        for file, title in available:
            table = pd.read_csv(ASSETS / "framework" / file, dtype=str).replace(renames).replace({chr(0x2014): "n/a"})
            for col in table.columns:  # e.g. "0.702 (derived_race)" -> "0.702 (race)"
                for raw, label in ATTRIBUTE_LABELS.items():
                    table[col] = table[col].str.replace(f"({raw})", f"({label.lower()})", regex=False)
            table = table.rename(columns=lambda c: "Approval gap (group minus reference)"
                                 if c.startswith("approval gap") else c[0].upper() + c[1:])
            st.markdown(f"**{title}**")
            st.dataframe(table, hide_index=True, width="stretch")

st.subheader("Known limitations")
st.markdown(
    "- The models predict the lender's decision, not default, and inherit any bias in it.\n"
    "- No credit score or loan performance in the data: credit losses are an assumption.\n"
    "- National data stands in for ApexLend's book; P&L figures are illustrative.\n"
    "- Excluding protected attributes does not remove proxies.\n"
    "- Stability is measured by retraining, not over time.\n"
    "- Logistic regression has 12 inputs against 31, so part of its accuracy gap is its inputs.")
footer()
