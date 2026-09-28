import altair as alt
import pandas as pd
import streamlit as st

from common import (ATTRIBUTE_LABELS, BURGUNDY, LENDER, MODEL_INFO, MODELS, MUTED, RECOMMENDED, REFERENCE_LABELS,
                    assets_ready, at_threshold, bn, break_even, colour_scale, fairness, footer,
                    glossary_link, group_label, in_text, load, meta, models_present, page_header, pct, pnl,
                    rule_label, show_chart, so_what)

if not assets_ready():
    st.stop()
m = meta()
grid = load("grid")
present = models_present(grid)
focus = RECOMMENDED if RECOMMENDED in present else present[0]
focus_name, focus_short = in_text(focus), MODELS[focus]

# Assumptions are read first so the header can state the result. Widget values are re-assigned so they
# survive a visit to another page.
ss = st.session_state
for key, default in [("margin_pct", m["margin"] * 100), ("lgd_pct", m["lgd"] * 100), ("d_pct", 100.0),
                     ("manual_on", False)]:
    ss[key] = ss.get(key, default)
margin, lgd, d = ss["margin_pct"] / 100, ss["lgd_pct"] / 100, ss["d_pct"] / 100
t_be = break_even(margin, lgd, d)
if ss["manual_on"]:
    ss["manual_cutoff"] = ss.get("manual_cutoff", round(t_be * 200) / 2)
t = ss["manual_cutoff"] / 100 if ss["manual_on"] else t_be
ss["threshold"] = t

overall = grid[grid["attribute"] == "all"]
now = at_threshold(overall, t).set_index("model")
oracle = margin * m["amt_pos_total"]
# At the base-case assumptions, use the exact figures at the break-even cutoff (as on the Executive summary)
# rather than the nearest 0.001 step of the precomputed grid, so every page shows the same numbers.
base_case = not ss["manual_on"] and (margin, lgd, d) == (m["margin"], m["lgd"], 1.0)


def approval_and_pnl(key: str) -> tuple[float, float]:
    if base_case and key in m["models"]:
        return m["models"][key]["approval_rate"], m["models"][key]["pnl"]
    return now.loc[key, "approved"] / now.loc[key, "n"], pnl(now.loc[key], margin, lgd, d)


f_approval, f_pnl = approval_and_pnl(focus)

page_header(
    "Approval policy",
    f"At a cutoff of {pct(t)}, {focus_name} approves {pct(f_approval)} of applications and "
    f"earns {bn(f_pnl)}, against {bn(oracle)} for the lender's own decisions.",
    "Everything below updates with the assumptions.",
    ["Financial", "Fairness"],
)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.slider("Net interest margin (%)", 1.0, 5.0, step=0.5, key="margin_pct", format="%.1f%%")
    c2.slider("Loss rate on a bad approval (%)", 10.0, 60.0, step=5.0, key="lgd_pct", format="%.0f%%")
    c3.slider("Default share d (%)", 5.0, 100.0, step=5.0, key="d_pct", format="%.0f%%",
              help="Share of lender-denied applicants assumed to default. 100% is the worst case.")
    c1, c2 = st.columns([1, 2])
    c1.metric("Break-even cutoff", pct(t_be))
    c2.toggle("Set the cutoff directly", key="manual_on")
    if ss["manual_on"]:
        c2.slider("Cutoff (%)", 30.0, 99.0, step=0.5, key="manual_cutoff", format="%.1f%%")

rows = [{"Model": MODEL_INFO[k]["name"], "Approval rate": pct(approval_and_pnl(k)[0]), "P&L": bn(approval_and_pnl(k)[1]),
         "Share of the lender's P&L": pct(approval_and_pnl(k)[1] / oracle, 0)} for k in present]
rows.append({"Model": LENDER, "Approval rate": pct(m["lender_approval_rate"]), "P&L": bn(oracle),
             "Share of the lender's P&L": "100%"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={c: st.column_config.TextColumn(c, alignment="right")
                            for c in ["Approval rate", "P&L", "Share of the lender's P&L"]})

curve = overall.assign(Model=overall["model"].map(MODELS), cutoff=overall["threshold"],
                       profit=pnl(overall, margin, lgd, d) / 1e9)
best = curve[curve["model"] == focus].nlargest(1, "profit").iloc[0]
lines = alt.Chart(curve).mark_line(strokeWidth=2).encode(
    x=alt.X("cutoff:Q", title="Cutoff", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0.3, 1])),
    y=alt.Y("profit:Q", title="P&L ($bn)"),
    color=colour_scale(curve["Model"]),
    tooltip=["Model", alt.Tooltip("cutoff:Q", title="Cutoff", format=".1%"),
             alt.Tooltip("profit:Q", title="P&L ($bn)", format=",.2f")])
show_chart(lines + rule_label(t, f"Current cutoff {pct(t)}"),
           f"{MODEL_INFO[focus]['name']}'s P&L peaks at a cutoff of {pct(best['cutoff'])} under these assumptions")

st.subheader("Fairness at this cutoff")
c1, c2 = st.columns(2)
attribute = c1.radio("Compare by", list(m["attributes"]), format_func=ATTRIBUTE_LABELS.get, horizontal=True)
views = {"Disparate impact ratio": "impact_ratio", "Approval rate": "approval", "True positive rate": "tpr"}
view = c2.segmented_control("Show", list(views), default="Disparate impact ratio") or "Disparate impact ratio"
reference = m["attributes"][attribute]
by_group = grid[grid["attribute"] == attribute]
table = fairness(at_threshold(by_group, t), reference)
wide = table.pivot(index="group", columns="model", values=views[view])
wide = wide[[MODELS[k] for k in present] + ([LENDER] if views[view] != "tpr" else [])]
shown = wide.map(lambda v: f"{v:.2f} ▼" if v < 0.8 else f"{v:.2f}") if views[view] == "impact_ratio" else wide.map(pct)
shown.insert(0, "Applications", table.drop_duplicates("group").set_index("group")["n"].map("{:,.0f}".format))
shown.index = shown.index.map(group_label)
shown = shown.reset_index().rename(columns={"group": "Group"})
st.dataframe(shown, hide_index=True, width="stretch",
             column_config={c: st.column_config.TextColumn(c, alignment="right") for c in shown.columns[1:]})
st.caption(f"Reference group: {REFERENCE_LABELS[reference]}. ▼ below the four-fifths line (0.80). "
           "Groups under 1,000 applications are not shown.")
glossary_link()

lender_ratio = table[(table["model"] == LENDER) & (table["group"] != reference)].set_index("group")["impact_ratio"]
group = st.selectbox("Follow one group across cutoffs", list(lender_ratio.sort_values().index), format_func=group_label)
per_t = pd.concat([fairness(r, reference).assign(threshold=th) for th, r in by_group.groupby("threshold")])
# Above 0.97 so few applications are approved that group ratios are mostly noise.
per_t = per_t[(per_t["group"] == group) & (per_t["threshold"] <= 0.97)].rename(columns={"model": "Model"})
models_only = per_t[per_t["Model"] != LENDER]
benchmark = lender_ratio[group]
chart = (alt.Chart(models_only).mark_line(strokeWidth=2).encode(
            x=alt.X("threshold:Q", title="Cutoff", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0.3, 1])),
            y=alt.Y("impact_ratio:Q", title="Disparate impact ratio"),
            color=colour_scale(models_only["Model"]),
            tooltip=["Model", alt.Tooltip("threshold:Q", title="Cutoff", format=".1%"),
                     alt.Tooltip("impact_ratio:Q", title="Ratio", format=".2f")])
         + rule_label(benchmark, f"Lender's own decisions ({benchmark:.2f})", axis="y", colour=MUTED)
         + rule_label(0.8, "Four-fifths line (0.80)", axis="y", colour=BURGUNDY, dash=(2, 2), below=benchmark >= 0.8)
         + rule_label(t, "Current cutoff"))
focus_curve = models_only[models_only["Model"] == focus_short].set_index("threshold")["impact_ratio"]
t_grid = focus_curve.index[abs(focus_curve.index - t).argmin()]
lower = focus_curve.index[abs(focus_curve.index - max(0.30, t - 0.10)).argmin()]
direction = "rises" if focus_curve[lower] > focus_curve[t_grid] else "falls"
show_chart(chart, f"{group_label(group)}: {focus_name}'s ratio {direction} as the cutoff comes down")

lower_row = at_threshold(overall, lower).set_index("model").loc[focus]
so_what(f"A {pct(lower)} cutoff would move {focus_name}'s ratio for {group_label(group)} applicants from "
        f"{focus_curve[t_grid]:.2f} to {focus_curve[lower]:.2f}, at a P&L of {bn(pnl(lower_row, margin, lgd, d))}.")
footer()
