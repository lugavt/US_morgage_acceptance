import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (LABELS, MODEL_INFO, MODELS, RECOMMENDED, app_ref, applicant_card, applicant_picker, assets_ready,
                    colour_scale, current_threshold, describe, footer, in_text, load, md, models_present,
                    page_header, pct, rule_label, show_chart, so_what)

# How each scenario factor is shown: a unit-bearing axis title, a value transform and a d3 axis format.
FACTORS = {
    "income": ("Income ($)", lambda v: v * 1000, "$,.0f"),
    "loan_amount": ("Loan amount ($)", lambda v: v, "$,.0f"),
    "combined_loan_to_value_ratio": ("Loan-to-value (%)", lambda v: v, ".0f"),
    "debt_to_income_ratio": ("Debt-to-income (%)", lambda v: v, ".0f"),
}
EXPECTED = {"income": 1, "loan_amount": 0, "combined_loan_to_value_ratio": -1, "debt_to_income_ratio": -1}


def response_sentences(rows: pd.DataFrame, feature: str) -> list[str]:
    out = []
    for key in [k for k in MODEL_INFO if k in set(rows["model"])]:
        r = rows[rows["model"] == key].sort_values("value")
        lo, hi = r.iloc[0], r.iloc[-1]
        move = hi["score"] - lo["score"]
        name = in_text(key)
        span = f"from {describe(lo['value'], feature)} to {describe(hi['value'], feature)}"
        if abs(move) < 0.005:
            out.append(f"Raising {LABELS[feature].lower()} {span} barely changes {name}'s approval score "
                       f"({pct(lo['score'], 0)}).")
            continue
        out.append(f"Raising {LABELS[feature].lower()} {span} {'raises' if move > 0 else 'lowers'} {name}'s "
                   f"approval score from {pct(lo['score'], 0)} to {pct(hi['score'], 0)}.")
        steps = np.diff(r["score"].to_numpy())
        against = np.where(np.sign(steps) == -np.sign(move))[0]
        against = [i for i in against if abs(steps[i]) >= 0.01]
        if against:
            i = max(against, key=lambda j: abs(steps[j]))
            out.append(f"{name[0].upper() + name[1:]}'s response is not smooth between {describe(r.iloc[i]['value'], feature)} and "
                       f"{describe(r.iloc[i + 1]['value'], feature)}, where the score moves the other way by "
                       f"{abs(steps[i]) * 100:.1f} points.")
    return out


if not assets_ready():
    st.stop()
t = current_threshold()
whatif = load("whatif")
header = st.container()  # filled below, once the application is chosen
a = applicant_picker()

if whatif is None:
    page_header("Applicant scenarios", "Scenarios are not included in this review run.", "",
                ["Performance", "Interpretability"], where=header)
    footer()
    st.stop()

features = [f for f in FACTORS if f in set(whatif["feature"])]
feature = st.session_state.get("scenario_factor", features[0])
rows = whatif[(whatif["row_nr"] == a.name) & (whatif["feature"] == feature)]
focus = RECOMMENDED if RECOMMENDED in set(rows["model"]) else models_present(rows)[0]
f_rows = rows[rows["model"] == focus]
spread = f_rows["score"].max() - f_rows["score"].min()
page_header("Applicant scenarios",
            f"For {app_ref(a.name)}, moving {LABELS[feature].lower()} across its usual range changes "
            f"{in_text(focus)}'s approval score by up to {spread * 100:.0f} points.",
            "Move one factor from its 5th to its 95th percentile, everything else held equal.",
            ["Performance", "Interpretability"], where=header)

applicant_card(a)
st.radio("Factor", features, key="scenario_factor", format_func=lambda f: FACTORS[f][0], horizontal=True)

title, transform, fmt = FACTORS[feature]
plot = rows.assign(Model=rows["model"].map(MODELS), x=rows["value"].map(transform))
lines = alt.Chart(plot).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X("x:Q", title=title, axis=alt.Axis(format=fmt)),
    y=alt.Y("score:Q", title="Approval score", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
    color=colour_scale(plot["Model"]),
    tooltip=["Model", alt.Tooltip("x:Q", title=title, format=fmt),
             alt.Tooltip("score:Q", title="Approval score", format=".1%")])
chart = lines + rule_label(t, f"Approval cutoff {pct(t)}", axis="y")
if pd.notna(a[feature]):
    chart = chart + rule_label(transform(a[feature]), "This applicant", colour="#5F6673", dash=(1, 0), label_at=0.03)
show_chart(chart, f"Approval score as {LABELS[feature].lower()} changes")

sentences = response_sentences(rows, feature)
st.markdown(md("\n".join(f"- {s}" for s in sentences)))
st.caption("One factor moves at a time: a larger loan amount does not change the loan-to-value here.")

expected = EXPECTED[feature]
moves = {k: (lambda r: r["score"].iloc[-1] - r["score"].iloc[0])(rows[rows["model"] == k].sort_values("value"))
         for k in models_present(rows)}
flat = [in_text(k) for k, mv in moves.items() if abs(mv) < 0.005]
if expected:
    wrong = [in_text(k) for k, mv in moves.items() if abs(mv) >= 0.005 and np.sign(mv) == -expected]
    text = ("Every model that responds moves in the direction a credit officer would expect."
            if not wrong else f"{', '.join(wrong)} moves against expectation here; a model risk reviewer would "
                              "ask why.")
    if flat:
        text += f" {', '.join(flat)} barely responds to this factor."
    text = text[0].upper() + text[1:]
    so_what(text)
else:
    so_what("A larger loan has no single expected direction; read it together with loan-to-value.")
footer()
