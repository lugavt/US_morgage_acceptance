import altair as alt
import pandas as pd
import streamlit as st

from common import (glossary_link, MODEL_INFO, MODELS, RECOMMENDED, app_ref, applicant_card, applicant_picker, assets_ready,
                    colour_scale, consistency_rating, current_threshold, footer, in_text, load, models_present,
                    page_header, pct, rule_label, show_chart, so_what, unchanged_share)

if not assets_ready():
    st.stop()
t = current_threshold()
refits = load("refits")
header = st.container()  # filled below, once the application is chosen
a = applicant_picker()

if refits is None:
    page_header("Decision consistency", "Retraining results are not included in this review run.", "",
                ["Stability"], where=header)
    footer()
    st.stop()

rows = refits[refits["row_nr"] == a.name]
present = models_present(rows)
n_retrains = rows["refit"].nunique()
focus = RECOMMENDED if RECOMMENDED in present else present[0]
approving = {k: int((rows[rows["model"] == k]["score"] >= t).sum()) for k in present}
page_header("Decision consistency",
            f"{approving[focus]} of {n_retrains} retrains of {in_text(focus)} approve {app_ref(a.name)} at the "
            f"{pct(t)} cutoff.",
            f"Each model was retrained {n_retrains} times on a bootstrap sample. Pick an application to see whether "
            "its decision holds.", ["Stability"], where=header)

applicant_card(a)

cols = st.columns(max(len(present), 1))
for col, key in zip(cols, present):
    r = rows[rows["model"] == key]["score"]
    with col.container(border=True):
        st.markdown(f"**{MODEL_INFO[key]['name']}**")
        st.metric("Retrains that approve", f"{approving[key]} of {n_retrains}")
        st.caption(f"Score range across retrains: {pct(r.min())} to {pct(r.max())}")

plot = rows.assign(Model=rows["model"].map(MODELS))
ticks = alt.Chart(plot).mark_tick(thickness=2, size=26).encode(
    x=alt.X("score:Q", title="Approval score", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("Model:N", title=None, sort=[MODELS[k] for k in present]),
    color=colour_scale(plot["Model"], legend=None),
    tooltip=["Model", alt.Tooltip("score:Q", title="Approval score", format=".1%")])
show_chart(ticks + rule_label(t, f"Approval cutoff {pct(t)}"), f"Approval score across {n_retrains} retrains",
           height=60 + 45 * len(present))
st.caption("This measures retraining, not drift over time; drift is monitored after go-live.")

st.subheader("Across the applications in this review")
shares = unchanged_share(t)
applicants = load("applicants")
random_rows = applicants.loc[applicants["picked_because"].str.contains("lender"), "row_nr"]
spread = (refits[refits["row_nr"].isin(random_rows)].groupby(["model", "row_nr"])["score"]
          .agg(lambda s: s.max() - s.min()).groupby(level="model").mean())
summary = pd.DataFrame([{"Model": MODEL_INFO[k]["name"],
                         "Decisions unchanged across all retrains": pct(shares[k], 0),
                         "Average score range across retrains": f"{spread[k] * 100:.1f} points",
                         "Rating": consistency_rating(shares[k])} for k in models_present(refits)])
st.dataframe(summary, hide_index=True, width="stretch",
             column_config={c: st.column_config.TextColumn(c, alignment="right")
                            for c in ["Decisions unchanged across all retrains", "Average score range across retrains"]})
st.caption(f"The {len(random_rows)} randomly sampled applications, at {pct(t)}. Close calls are left out.")
glossary_link("Stability measures in the Glossary")

focus_share = shares.get(focus)
so_what(f"{pct(1 - focus_share, 0)} of sampled {in_text(focus)} decisions change in at least one retrain, so SR 11-7 "
        "monitoring of retrained models is a condition for go-live.")
footer()
