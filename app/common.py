"""Shared names, wording, formatting, calculations and chart styling for the app pages.

Everything comes from app/assets/, built by notebooks/app_assets.ipynb. No model is loaded here.
"""
import base64
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# APEXLEND_ASSETS points the app at another asset folder when testing (e.g. without TabPFN).
ASSETS = Path(os.environ.get("APEXLEND_ASSETS", Path(__file__).parent / "assets"))

APP_NAME = "ApexLend · Credit Model Review"
APP_SUBTITLE = "Prepared for the Credit Committee"
TEAM_NAME = "Group 6, HEC Paris"  # as on slide 1
MEETING = "Credit Committee · 28 September 2026"
TEAM = ["Arnolfo Acero", "Korouhanba Khuman Laikhuram", "Lucca Vettori", "Rémi Laget", "Wei-Chieh Chou",
        "Bhavesh Chauhan"]
COURSE = "Interpretability, Stability and Algorithmic Fairness"
PROFESSOR = "Prof. Christophe Pérignon"
REPO = "https://github.com/lugavt/US_morgage_acceptance"
FOOTER = (f"ApexLend is a fictional client. Prepared by {TEAM_NAME} for the {COURSE} course ({PROFESSOR}). "
          "Data: 2025 HMDA public Loan/Application Register. Not a lending decision tool.")

# The model the deck recommends (slide 23): XGBoost, conditional on fairness remediation before deployment.
RECOMMENDED = "xgboost"

# --- Palette (matches the slide deck) ---
NAVY, BURGUNDY, MUTED = "#1B2433", "#943340", "#5F6673"
GRID, RULE = "#E6E4DE", "#B9B6AE"
RAISES = "#5F7A99"  # bars for factors that raise the score; BURGUNDY for those that lower it

# --- Models: real names, as in the deck ---
MODEL_INFO = {
    "logreg": {"name": "Logistic regression", "short": "Logistic regression", "technique": "12 inputs",
               "blurb": "the white-box model"},
    "xgboost": {"name": "XGBoost", "short": "XGBoost", "technique": "31 inputs",
                "blurb": "the machine-learning model"},
    "tabpfn": {"name": "TabPFN", "short": "TabPFN", "technique": "TabPFN-3.5, 31 inputs",
               "blurb": "the tabular foundation model"},
}
MODELS = {k: v["short"] for k, v in MODEL_INFO.items()}  # key -> label used in charts and tables


def in_text(key: str) -> str:
    """Model name inside a sentence ("logistic regression", "XGBoost", "TabPFN")."""
    return "logistic regression" if key == "logreg" else MODEL_INFO[key]["name"]


LENDER = "Lender's own decisions"
LENDER_NAME = LENDER
COLOURS = {"Logistic regression": "#3B6EA5", "XGBoost": "#D9822B", "TabPFN": "#4E9A6B", LENDER: "#8A8F98"}

# Explainability rating (rule documented in the Glossary)
EXPLAINABILITY = {
    "logreg": ("Strong", "Readable coefficients; SHAP is exact (LinearExplainer)."),
    "xgboost": ("Moderate", "SHAP is exact per applicant (TreeExplainer), but a simple tree mimics only 55% of it."),
    "tabpfn": ("Limited", "SHAP is only approximate (KernelExplainer)."),
}


# Stability figures used in the deck: stability.ipynb summary on origin/main (c459759), 26 Sep run.
# They are not in app/assets because the notebook saves them outside git.
DECK_STABILITY = {
    "rank": {"xgboost": 39.9, "logreg": 3.4, "tabpfn": 16.0},
    "d1d2": {"xgboost": 41.9, "logreg": 2.8, "tabpfn": 11.7},
    "auc_sd": {"xgboost": 0.0023, "logreg": 0.0009, "tabpfn": 0.0023},
    "refits": {"xgboost": 30, "logreg": 30, "tabpfn": 10},
    "param_median": 0.62, "param_mean": 0.69,
}


def consistency_rating(share: float | None) -> str:
    """Rating from the share of applications whose decision is unchanged across all retrains."""
    if share is None:
        return "n/a"
    return "Strong" if share >= 0.9 else "Moderate" if share >= 0.7 else "Limited"


# --- Client expectations, word for word from slide 3, ranked by the slide's importance chart ---
PRIORITIES = [
    ("Financial", "Direct alignment with bottom-line ROI by minimizing expected credit loss (cost of defaulters) "
                  "while maximizing interest revenues from approved loans."),
    ("Performance", "High predictive power and classification accuracy to effectively differentiate between "
                    "reliable borrowers and potential defaulters."),
    ("Fairness", "Compliance with anti-discrimination laws, ensuring fair lending decisions across protected "
                 "demographic groups."),
    ("Interpretability", "High transparency and explainability to meet strict financial regulations and generate "
                         "clear adverse action reasons for denied applicants."),
    ("Stability", "Robustness against population and economic shifts over time, maintaining consistent predictive "
                  "reliability with minimal performance drift."),
]

ATTRIBUTE_LABELS = {"derived_race": "Race", "derived_ethnicity": "Ethnicity", "derived_sex": "Sex",
                    "applicant_age": "Age"}
REFERENCE_LABELS = {"White": "White applicants", "Not Hispanic or Latino": "applicants not Hispanic or Latino",
                    "Male": "male applicants", "35-44": "applicants aged 35 to 44"}
GROUP_LABELS = {"<25": "Under 25", ">74": "Over 74", "25-34": "25 to 34", "35-44": "35 to 44", "45-54": "45 to 54",
                "55-64": "55 to 64", "65-74": "65 to 74", "Joint": "Joint applicants"}

# --- The 30 model inputs, in plain English ---
LABELS = {
    "loan_amount": "Loan amount", "combined_loan_to_value_ratio": "Loan-to-value",
    "loan_term": "Loan term", "intro_rate_period": "Introductory rate period",
    "property_value": "Property value", "total_units": "Number of units", "income": "Income",
    "debt_to_income_ratio": "Debt-to-income", "conforming_loan_limit": "Conforming loan limit",
    "derived_loan_product_type": "Loan product", "derived_dwelling_category": "Dwelling type",
    "loan_type": "Loan type", "loan_purpose": "Loan purpose", "lien_status": "Lien position",
    "reverse_mortgage": "Reverse mortgage", "open_end_line_of_credit": "Open-end line of credit",
    "business_or_commercial_purpose": "Business purpose", "negative_amortization": "Negative amortization",
    "interest_only_payment": "Interest-only payments", "balloon_payment": "Balloon payment",
    "other_nonamortizing_features": "Other non-amortizing features", "construction_method": "Construction method",
    "occupancy_type": "Occupancy", "manufactured_home_secured_property_type": "Manufactured home security",
    "manufactured_home_land_property_interest": "Manufactured home land interest",
    "applicant_credit_score_type": "Applicant's credit score model",
    "co_applicant_credit_score_type": "Co-applicant's credit score model",
    "submission_of_application": "Application channel", "aus_1": "Automated underwriting system",
    "has_co_applicant": "Co-applicant",
}
_YES_NO = {1: "Yes", 2: "No"}
# HMDA codes (FFIEC filing instructions); 1111 means the lender was exempt from reporting the field.
CODES = {
    "loan_type": {1: "Conventional", 2: "FHA", 3: "VA", 4: "USDA"},
    "loan_purpose": {1: "Home purchase", 2: "Home improvement", 31: "Refinancing", 32: "Cash-out refinancing",
                     4: "Other purpose", 5: "Not applicable"},
    "lien_status": {1: "First lien", 2: "Subordinate lien"},
    "occupancy_type": {1: "Principal residence", 2: "Second residence", 3: "Investment property"},
    "reverse_mortgage": _YES_NO, "open_end_line_of_credit": _YES_NO, "negative_amortization": _YES_NO,
    "interest_only_payment": _YES_NO, "balloon_payment": _YES_NO, "other_nonamortizing_features": _YES_NO,
    "business_or_commercial_purpose": {1: "Primarily business", 2: "Not business"},
    "construction_method": {1: "Site-built", 2: "Manufactured home"},
    "manufactured_home_secured_property_type": {1: "Home and land", 2: "Home only", 3: "Not applicable"},
    "manufactured_home_land_property_interest": {1: "Direct ownership", 2: "Indirect ownership",
                                                 3: "Paid leasehold", 4: "Unpaid leasehold", 5: "Not applicable"},
    "submission_of_application": {1: "Direct to lender", 2: "Through a broker", 3: "Not applicable"},
    "aus_1": {1: "Desktop Underwriter", 2: "Loan Product Advisor", 3: "TOTAL Scorecard", 4: "GUS",
              5: "Other system", 6: "Not applicable", 7: "Internal proprietary system"},
    "applicant_credit_score_type": {1: "Equifax Beacon 5.0", 2: "Experian Fair Isaac", 3: "FICO Classic 04",
                                    4: "FICO Classic 98", 5: "VantageScore 2.0", 6: "VantageScore 3.0",
                                    7: "More than one model", 8: "Other model", 9: "Not applicable"},
    "conforming_loan_limit": {"C": "Conforming", "NC": "Nonconforming", "U": "Undetermined", "NA": "Not applicable"},
}
CODES["co_applicant_credit_score_type"] = {**CODES["applicant_credit_score_type"], 10: "No co-applicant"}

# Why an application is in the review sample (values of applicants.picked_because)
SAMPLE_REASONS = {"lender approved": "Lender approved (random sample)",
                  "lender denied": "Lender denied (random sample)",
                  "models disagree": "Models disagree",
                  "close to the threshold": "Close to the cutoff"}


# --- Loading ---

@st.cache_data
def meta() -> dict:
    return json.loads((ASSETS / "meta.json").read_text())


@st.cache_data
def load(name: str) -> pd.DataFrame | None:
    path = ASSETS / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def models_present(frame: pd.DataFrame | None) -> list[str]:
    """Model keys present in an asset file, in the standard order."""
    if frame is None:
        return []
    return [k for k in MODEL_INFO if k in set(frame["model"])]


def assets_ready() -> bool:
    if not (ASSETS / "meta.json").exists():
        st.error("Review data not found. Build it with notebooks/app_assets.ipynb before opening the app.")
        return False
    if meta().get("dry_run"):
        st.warning("Illustrative data. These figures come from a test run and will be replaced before the "
                   "committee meeting.")
    return True


# --- Formatting ---

def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def bn(x: float) -> str:
    return f"${x / 1e9:,.2f}bn"


def app_ref(row_nr: int) -> str:
    s = f"{int(row_nr):07d}"
    return f"Application {s[:4]} {s[4:]}"


def group_label(group) -> str:
    return GROUP_LABELS.get(group, group) if isinstance(group, str) else "not reported"


def sample_reason(value: str) -> str:
    key = "lender approved" if value.endswith("lender approved") else \
          "lender denied" if value.endswith("lender denied") else value
    return SAMPLE_REASONS.get(key, value)


def describe(value, feature: str) -> str:
    """A raw input value in plain English with its unit."""
    if pd.isna(value):
        return "not reported"
    if feature in CODES:
        code = value if isinstance(value, str) else int(value)
        if code == 1111:
            return "exempt from reporting"
        return CODES[feature].get(code, f"code {code}")
    if feature == "has_co_applicant":
        return "Yes" if value else "No"
    if feature == "income":
        return f"${value * 1000:,.0f}"  # HMDA income is in $000s
    if feature in ("loan_amount", "property_value"):
        return f"${value:,.0f}"
    if feature in ("combined_loan_to_value_ratio", "debt_to_income_ratio"):
        return f"{value:.0f}%"
    if feature == "loan_term":
        return f"{value / 12:.0f} years" if value % 12 == 0 else f"{value:.0f} months"
    if feature == "intro_rate_period":
        return f"{value:.0f} month" + ("" if value == 1 else "s")
    if feature in ("derived_dwelling_category", "derived_loan_product_type"):  # e.g. "Conventional:First Lien"
        text = str(value).replace("Single Family (1-4 Units)", "Single-family").replace(":", ", ")
        first, *rest = text.split(", ")
        return ", ".join([first] + [part.lower() for part in rest])
    if feature == "total_units":
        return f"{value:.0f}"
    return str(value)


# --- Calculations (unchanged from the first version of the app) ---

def break_even(margin: float, lgd: float, default_share: float) -> float:
    """Approve when p x margin > (1 - p) x LGD x d, i.e. p > LGD d / (margin + LGD d)."""
    return lgd * default_share / (margin + lgd * default_share)


def current_threshold() -> float:
    return st.session_state.get("threshold", meta()["threshold"])


def at_threshold(grid: pd.DataFrame, t: float) -> pd.DataFrame:
    """Grid rows at the grid threshold closest to t."""
    thresholds = grid["threshold"].unique()
    nearest = thresholds[abs(thresholds - t).argmin()]
    return grid[grid["threshold"] == nearest]


def fairness(rows: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Approval rate, approval rate among qualified applicants (TPR) and adverse impact ratio
    against the reference group, per model and group, for one attribute at one cutoff.

    The lender's own historical decisions are added as a fourth "model", so every ratio can be
    read against what already happens today.
    """
    out = rows.assign(model=rows["model"].map(MODELS),
                      approval=rows["approved"] / rows["n"],
                      tpr=rows["approved_pos"] / rows["n_pos"])
    lender = out.drop_duplicates("group").assign(model=LENDER, approval=lambda d: d["n_pos"] / d["n"], tpr=1.0)
    out = pd.concat([out, lender], ignore_index=True)
    ref = out[out["group"] == reference].set_index("model")["approval"]
    out["impact_ratio"] = out["approval"] / out["model"].map(ref)
    return out[["model", "group", "n", "approval", "tpr", "impact_ratio"]]


def pnl(rows: pd.DataFrame, margin: float, lgd: float, default_share: float) -> pd.Series:
    """Loan-level P&L in $: margin on approved loans the lender approved, minus LGD x d on those it denied."""
    return margin * rows["amt_approved_pos"] - lgd * default_share * rows["amt_approved_neg"]


def lowest_ratio(grid: pd.DataFrame, t: float, attribute: str = "derived_race") -> pd.DataFrame:
    """Per model label (and the lender): the group with the lowest disparate impact ratio at cutoff t."""
    reference = meta()["attributes"][attribute]
    table = fairness(at_threshold(grid[grid["attribute"] == attribute], t), reference)
    table = table[table["group"] != reference]
    return table.loc[table.groupby("model")["impact_ratio"].idxmin()].set_index("model")


def unchanged_share(t: float) -> dict:
    """Per model key: share of randomly sampled applications whose decision is the same under all retrains."""
    refits, applicants = load("refits"), load("applicants")
    if refits is None:
        return {}
    random_rows = applicants.loc[applicants["picked_because"].str.contains("lender"), "row_nr"]
    r = refits[refits["row_nr"].isin(random_rows)]
    share_approving = (r["score"] >= t).groupby([r["model"], r["row_nr"]]).mean()
    return ((share_approving == 0) | (share_approving == 1)).groupby(level="model").mean().to_dict()


# --- Page furniture ---

def md(text: str) -> str:
    """Escape dollar signs, which Streamlit markdown would otherwise read as maths."""
    return text.replace("$", "\\$")


def page_header(question: str, takeaway: str, what_you_can_do: str, priorities: list[str], where=None,
                meta_line: str | None = None, logo: bool = False) -> None:
    """Title (with the HEC Paris logo if asked), a rule, a meta line with the client priorities as tags, the
    takeaway sentence and what the page is for. `where` is a container reserved earlier, so the header can sit
    above widgets that were drawn first."""
    with where if where is not None else st.container():
        _page_header(question, takeaway, what_you_can_do, priorities, meta_line, logo)


def _page_header(question: str, takeaway: str, what_you_can_do: str, priorities: list[str],
                 meta_line: str | None, logo: bool) -> None:
    if logo:
        left, right = st.columns([6, 1], vertical_alignment="center")
        left.title(question)
        mark = base64.b64encode((ASSETS / "hec_paris_logo.png").read_bytes()).decode()
        right.markdown(f"<div style='text-align:right'><img src='data:image/png;base64,{mark}' "
                       f"alt='HEC Paris logo' style='height:40px'></div>", unsafe_allow_html=True)
    else:
        st.title(question)
    tags = "".join(f"<span style='display:inline-block;padding:0.1rem 0.65rem;margin:0 0.35rem 0.3rem 0;"
                   f"border-radius:999px;background:#F3E6E8;color:{BURGUNDY};font-size:0.75rem;"
                   f"letter-spacing:0.02em'>{p}</span>" for p in priorities)
    meta = (f"<span style='color:{MUTED};font-size:0.8rem;margin-right:0.8rem'>{meta_line}</span>"
            if meta_line else "")
    st.markdown(f"<div style='border-top:2px solid {NAVY};margin:-0.3rem 0 0.7rem 0;padding-top:0.6rem;"
                f"display:flex;flex-wrap:wrap;align-items:center'>{meta}{tags}</div>", unsafe_allow_html=True)
    st.markdown(md(f"**{takeaway}**"))
    if what_you_can_do:
        st.caption(md(what_you_can_do))


def glossary_link(label: str = "Definitions in the Glossary") -> None:
    try:
        st.page_link("views/glossary.py", label=label, icon=":material/dictionary:")
    except st.errors.StreamlitPageNotFoundError:  # page run on its own, outside app.py's navigation
        st.caption(label + ".")


def so_what(text: str) -> None:
    with st.container(border=True):
        st.markdown(f"<span style='color:{BURGUNDY};font-weight:600'>So what for ApexLend.</span> {md(text)}",
                    unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f"<div style='border-top:1px solid {RULE};margin-top:2.5rem;padding-top:0.6rem;"
                f"color:{MUTED};font-size:0.78rem'>{FOOTER}</div>", unsafe_allow_html=True)


def applicant_picker() -> pd.Series:
    """Application chooser shared by the per-application pages; the choice carries across pages."""
    applicants = load("applicants").set_index("row_nr")
    reasons = applicants["picked_because"].map(sample_reason)
    c1, c2 = st.columns(2)
    why = c1.selectbox("Why this file is in the sample", ["All files"] + sorted(reasons.unique()))
    race = c2.selectbox("Applicant group", ["All groups"] + sorted(applicants["derived_race"].dropna().unique()))
    pool = applicants
    if why != "All files":
        pool = pool[reasons.loc[pool.index] == why]
    if race != "All groups":
        pool = pool[pool["derived_race"] == race]
    if pool.empty:
        st.info("No application matches these filters.")
        st.stop()
    labels = {r: f"{app_ref(r)} · {sample_reason(a.picked_because)}"
                 + ("" if "lender" in a.picked_because else f" · lender {'approved' if a.target == 1 else 'denied'}")
              for r, a in pool.iterrows()}
    ids = list(labels)
    current = st.session_state.get("row_nr")
    row_nr = st.selectbox("Select an application to review", ids,
                          index=ids.index(current) if current in ids else 0, format_func=labels.get)
    st.session_state["row_nr"] = row_nr
    return pool.loc[row_nr]


def applicant_card(a: pd.Series) -> None:
    """Loan file summary: key facts, the lender's decision, and the fairness-only attributes."""
    left = ["loan_amount", "loan_purpose", "property_value", "combined_loan_to_value_ratio", "loan_type"]
    right = ["income", "debt_to_income_ratio", "loan_term", "occupancy_type", "has_co_applicant"]
    with st.container(border=True):
        st.markdown(f"**Loan file summary · {app_ref(a.name)}**")
        c1, c2 = st.columns(2)
        for col, fields in [(c1, left), (c2, right)]:
            col.markdown(md("  \n".join(f"{LABELS[f]}: **{describe(a[f], f)}**" for f in fields)))
        st.markdown(f"Lender's decision: **{'Approved' if a.target == 1 else 'Denied'}**")
        st.caption(f"Recorded for the fairness review only. Never used by the models: {a.derived_race}, "
                   f"{a.derived_ethnicity}, {a.derived_sex}, age {group_label(a.applicant_age)}.")


def model_scores(a: pd.Series) -> dict:
    """{model key: approval score} for the models scored for this application."""
    return {m: a[f"score_{m}"] for m in MODEL_INFO if f"score_{m}" in a.index and pd.notna(a[f"score_{m}"])}


def outcome(score: float, t: float) -> str:
    return "Approve" if score >= t else "Deny"


# --- Charts ---

def colour_scale(labels, **kwargs):
    """Colour encoding with the fixed model colours, for the labels actually present."""
    domain = [c for c in COLOURS if c in set(labels)]
    kwargs.setdefault("legend", alt.Legend(orient="bottom", title=None, labelFontSize=12, symbolType="stroke",
                                           symbolStrokeWidth=3))
    return alt.Color("Model:N", scale=alt.Scale(domain=domain, range=[COLOURS[c] for c in domain]), **kwargs)


def rule_label(value: float, text: str, axis: str = "x", colour: str = NAVY, dash=(4, 4), below: bool = False,
               label_at=None):
    """A labelled reference line (cutoff, benchmark, screening level). Vertical lines are labelled at the
    top, to the left of the line; horizontal ones at the left, above the line (or below it)."""
    frame = pd.DataFrame({"v": [value], "label": [text]})
    rule = alt.Chart(frame).mark_rule(color=colour, strokeDash=list(dash), strokeWidth=1.2)
    if axis == "x":
        text_mark = alt.Chart(frame).mark_text(align="right", dx=-4, baseline="top" if label_at is None else "bottom",
                                               fontSize=11, color=colour)
        y = alt.value(2) if label_at is None else alt.datum(label_at)  # label_at: a y value to write the label at
        return rule.encode(x="v:Q") + text_mark.encode(x="v:Q", y=y, text="label:N")
    text_mark = alt.Chart(frame).mark_text(align="left", dx=4, dy=4 if below else -4,
                                           baseline="top" if below else "bottom", fontSize=11, color=colour)
    return rule.encode(y="v:Q") + text_mark.encode(y="v:Q", x=alt.value(4), text="label:N")


def show_chart(chart, title: str | None = None, height: int = 320, right_pad: int = 20) -> None:
    """Render an Altair chart in the app's shared style: no borders, light horizontal gridlines,
    takeaway titles left-aligned."""
    chart = chart.properties(height=height, padding={"left": 5, "right": right_pad, "top": 10, "bottom": 5})
    if title:
        chart = chart.properties(title=alt.Title(title, anchor="start", offset=14))
    chart = (chart.configure(background="transparent", font="Inter, sans-serif")
             .configure_view(stroke=None)
             .configure_axis(labelColor=NAVY, titleColor=MUTED, gridColor=GRID, domainColor=RULE,
                             tickColor=RULE, titleFontWeight="normal", labelFontSize=12, titleFontSize=12)
             .configure_axisX(grid=False)
             .configure_title(color=NAVY, fontSize=15, fontWeight=600, font="Inter, sans-serif")
             .configure_legend(orient="bottom", title=None, labelColor=NAVY))
    st.altair_chart(chart, theme=None, width="stretch")
