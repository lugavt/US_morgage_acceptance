import altair as alt
import pandas as pd
import streamlit as st

from common import (BURGUNDY, glossary_link, LABELS, MODEL_INFO, NAVY, RAISES, app_ref, applicant_card, applicant_picker,
                    assets_ready, current_threshold, describe, footer, load, md, model_scores, outcome, page_header,
                    pct, show_chart, so_what)


NUMERIC = {"loan_amount", "combined_loan_to_value_ratio", "loan_term", "intro_rate_period", "property_value",
           "total_units", "income", "debt_to_income_ratio"}


def reason_sentence(row: pd.Series, a: pd.Series) -> str:
    feature, value = row["feature"], a[row["feature"]]
    if feature == "has_co_applicant":
        factor = "Having a co-applicant" if value else "No co-applicant"
    elif feature in NUMERIC:
        factor = f"{LABELS[feature]} of {describe(value, feature)}"
    else:
        factor = f"{LABELS[feature]} ({describe(value, feature)})"
    verb = "raises" if row["effect"] > 0 else "lowers"
    return f"{factor} {verb} the approval score by {abs(row['effect']) * 100:.1f} points."


if not assets_ready():
    st.stop()
t = current_threshold()
header = st.container()  # filled below, once the application is chosen
a = applicant_picker()
scores = model_scores(a)
reasons = load("reasons")
reasons = reasons[reasons["row_nr"] == a.name] if reasons is not None else None

denied = [k for k, p in scores.items() if p < t]
top_lowering = None
if reasons is not None and not reasons.empty:
    lowering = reasons[reasons["effect"] < 0]
    if not lowering.empty:
        top_lowering = LABELS[lowering.loc[lowering["effect"].idxmin(), "feature"]].lower()
takeaway = (f"{app_ref(a.name)}: {len(scores) - len(denied)} of {len(scores)} models approve it at the "
            f"{pct(t)} cutoff" + (f"; {top_lowering} lowers its score the most." if top_lowering else "."))
page_header("Decision explanation", takeaway,
            "Pick an application to see each model's decision and its principal reasons.",
            ["Interpretability", "Fairness"], where=header)

applicant_card(a)
st.caption(f"Cutoff {pct(t)}, set on the Approval policy page. Regulation B requires the specific principal "
           "reasons for every denial.")

cols = st.columns(max(len(scores), 1))
for col, (key, p) in zip(cols, scores.items()):
    info = MODEL_INFO[key]
    with col.container(border=True):
        st.markdown(f"**{info['name']}**")
        st.caption(f"{info['blurb']}, {info['technique']}")
        colour = NAVY if p >= t else BURGUNDY
        st.markdown(f"<div style='color:{colour};font-size:1.1rem;font-weight:600;margin:0.3rem 0'>"
                    f"{outcome(p, t)}</div>", unsafe_allow_html=True)
        st.markdown(f"Approval score: **{pct(p)}**")
        r = reasons[reasons["model"] == key] if reasons is not None else None
        if r is None or r.empty:
            st.caption("Reasons not included in this review run.")
            continue
        if p < t:
            top = r[r["effect"] < 0].nsmallest(4, "effect")
            st.markdown("**Principal reasons for denial**")
        else:
            top = r[r["effect"] > 0].nlargest(4, "effect")
            st.markdown("**Main factors behind approval**")
        st.markdown(md("\n".join(f"{i}. {reason_sentence(row, a)}" for i, (_, row) in enumerate(top.iterrows(), 1))
                    or "No single factor stands out."))
        chart = r.reindex(r["effect"].abs().sort_values(ascending=False).index).head(8)
        chart = chart.assign(Factor=chart["feature"].map(LABELS), points=chart["effect"] * 100,
                             Direction=chart["effect"].map(lambda e: "Raises the score" if e > 0 else "Lowers the score"))
        bars = alt.Chart(chart).mark_bar().encode(
            x=alt.X("points:Q", title="Change in approval score (points)"),
            y=alt.Y("Factor:N", sort=None, title=None),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises the score", "Lowers the score"],
                                                           range=[RAISES, BURGUNDY]), legend=None),
            tooltip=["Factor", "Direction", alt.Tooltip("points:Q", title="Points", format="+.1f")])
        st.markdown("**What drove this score**")
        show_chart(bars, height=240, right_pad=10)

st.caption("Effect of each factor: the change in the score when that factor alone takes typical values. "
           "Logistic regression uses 12 of the 30 factors.")
glossary_link("How reason codes are computed (Glossary)")

if scores:
    agree = len(set(outcome(p, t) for p in scores.values())) == 1
    so_what("All three models reach the same decision here." if agree else "The models disagree on this application.")
footer()
