"""Glossary: every metric, measure and method used in the deck, the notebooks and the app.

Figures come from the same review data as the other pages (meta.json, grid.parquet, the evaluation
notebooks' CSVs). Figures not in that data are constants below, each with its source.
"""
import pandas as pd
import streamlit as st

from common import (ASSETS, DECK_STABILITY, LENDER, assets_ready, MODEL_INFO, at_threshold, bn, footer, load, md, meta, page_header, pct)

# Figures not in app/assets, with their source on origin/main (c459759)
PR_AUC = {"xgboost": (0.960, 0.825), "tabpfn": (0.949, 0.776)}  # models/xgboost_results.json; tabpfn.ipynb
STABILITY = DECK_STABILITY  # the deck's set, defined once in common.py
VAL_OPTIMAL_XGB = 0.920  # validation P&L sweep, recomputed 28 Sep with the review models

if not assets_ready():
    st.stop()
m = meta()
grid = load("grid")
overall = grid[grid["attribute"] == "all"]
keys = [k for k in MODEL_INFO if k in m["models"]]
names = {k: MODEL_INFO[k]["name"] for k in keys}


def listing(fn, digits: int = 3) -> str:
    """'XGBoost 0.903, logistic regression 0.765, TabPFN 0.876' for a per-model function."""
    return ", ".join(f"{names[k]} {fn(k):.{digits}f}" for k in keys if fn(k) is not None)


def confusion(key: str, t: float) -> dict:
    r = at_threshold(overall[overall["model"] == key], t).iloc[0]
    tp = r["approved_pos"]
    fp = r["approved"] - tp
    fn = r["n_pos"] - tp
    tn = r["n"] - r["n_pos"] - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": r["n"]}


def at_half(stat: str) -> str:
    out = []
    for k in keys:
        c = confusion(k, 0.5)
        value = {"accuracy": (c["tp"] + c["tn"]) / c["n"], "precision": c["tp"] / (c["tp"] + c["fp"]),
                 "recall": c["tp"] / (c["tp"] + c["fn"]), "specificity": c["tn"] / (c["tn"] + c["fp"]),
                 "fpr": c["fp"] / (c["tn"] + c["fp"])}
        value["f1"] = 2 * value["precision"] * value["recall"] / (value["precision"] + value["recall"])
        out.append(f"{names[k]} {value[stat]:.3f}")
    return "At the 0.5 cutoff on the test set: " + ", ".join(out) + "."


fair = pd.read_csv(ASSETS / "framework" / "fairness_detail.csv", dtype=str) \
    if (ASSETS / "framework" / "fairness_detail.csv").exists() else None
interp = pd.read_csv(ASSETS / "framework" / "interpretability_summary.csv") \
    if (ASSETS / "framework" / "interpretability_summary.csv").exists() else None


def fair_value(model: str, attribute: str, column: str) -> str:
    if fair is None:
        return "n/a"
    row = fair[(fair["model"] == model) & (fair["attribute"] == attribute)]
    return row[column].iloc[0] if len(row) else "n/a"


def race_pairs(column: str, scale: float = 1.0, digits: int = 2) -> str:
    return ", ".join(f"{names[k]} {float(fair_value(k, 'derived_race', column)) * scale:.{digits}f}"
                     for k in keys if fair_value(k, "derived_race", column) != "n/a")


fidelity = ", ".join(f"{names[k]} {interp.set_index('model').loc[k, 'surrogate fidelity (R²)']:.2f}"
                     for k in keys) if interp is not None else "n/a"
oracle = m["margin"] * m["amt_pos_total"]
everyone = m["margin"] * m["amt_pos_total"] - m["lgd"] * m["amt_neg_total"]
best = max(keys, key=lambda k: m["models"][k]["pnl"])
runner_up = max((k for k in keys if k != best), key=lambda k: m["models"][k]["pnl"])

CATEGORIES = ["Data and setup", "Statistical performance", "Economic performance", "Interpretability",
              "Stability", "Fairness", "Statistical testing", "Regulatory context"]


def E(cat, term, definition, formula=None, project=None, why=None, builds=(), abbr=None, extra=None):
    return {"cat": cat, "term": term, "abbr": abbr, "definition": definition,
            "formula": [formula] if isinstance(formula, str) else list(formula or []),
            "project": project, "why": why, "builds": list(builds), "extra": extra}


def worked_example():
    """The centrepiece entry: from one loan's expected profit to the portfolio P&L and the 0.921 cutoff."""
    st.markdown("**1. Setup.** For application *i* with loan amount $L_i$ (in dollars, as HMDA reports it) and model "
                r"score $p_i = \hat{P}(\text{approved})$: approving a loan the lender approved earns the margin $m$ "
                r"of $L_i$; approving one the lender denied loses $\text{LGD}$ of $L_i$.")
    st.latex(r"m = 3\%, \qquad \text{LGD} = 35\%")
    st.markdown("**2. Expected profit of approving one application.**")
    st.latex(r"\mathbb{E}[\pi_i] = p_i\, m\, L_i \;-\; (1 - p_i)\,\text{LGD}\, L_i")
    st.markdown(r"**3. Break-even.** Approve when $\mathbb{E}[\pi_i] > 0$. The loan amount $L_i$ cancels, so the "
                "cutoff is the same for every loan size:")
    st.latex(r"p_i > t^* = \frac{\text{LGD}}{m + \text{LGD}} = \frac{0.35}{0.38} \approx 0.921")
    st.markdown(r"**4. Portfolio P&L at a cutoff $t$**, with $y_i = 1$ if the lender approved. We report it in "
                r"\$bn (divide by $10^9$). This is the formula in `notebooks/app_assets.ipynb`.")
    st.latex(r"\text{P\&L}(t) = \sum_{i:\,p_i \ge t} L_i \big[\, m\, y_i - \text{LGD}\,(1 - y_i) \big]")
    st.markdown(r"**5. Benchmarks.** The lender's own decisions earn $m \sum_i L_i y_i$. Approving everyone is "
                r"$t = 0$.")
    rows = [{"Strategy": names[k], "Approval rate": pct(m["models"][k]["approval_rate"]),
             "P&L ($bn)": f"{m['models'][k]['pnl'] / 1e9:+.2f}"} for k in keys]
    rows += [{"Strategy": LENDER, "Approval rate": pct(m["lender_approval_rate"]), "P&L ($bn)": f"{oracle / 1e9:+.2f}"},
             {"Strategy": "Approve everyone", "Approval rate": "100.0%", "P&L ($bn)": f"{everyone / 1e9:+.2f}"}]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(md(f"**6. Validation check.** A sweep over cutoffs on the validation set peaks at "
                   f"{VAL_OPTIMAL_XGB:.3f} for XGBoost, next to 0.921, as expected from calibrated probabilities."))
    st.markdown("**7. Caveat.** HMDA records decisions, not defaults, so treating every denial as a loss is the "
                r"worst case. If only a share $d$ of denied applicants would default:")
    st.latex(r"t^*(d) = \frac{\text{LGD}\, d}{m + \text{LGD}\, d}")
    st.dataframe(pd.DataFrame({"d": ["100%", "50%", "30%", "20%", "10%"],
                               "Break-even cutoff": [f"{0.35 * d / (0.03 + 0.35 * d):.3f}"
                                                     for d in (1, 0.5, 0.3, 0.2, 0.1)]}),
                 hide_index=True)


G = [
    # 1. Data and setup
    E("Data and setup", "HMDA", "The US Home Mortgage Disclosure Act public Loan/Application Register: one row per "
      "mortgage application, with the lender's decision, loan terms and applicant demographics.",
      abbr="Home Mortgage Disclosure Act",
      project="2025 file: 13,543,606 raw rows, 9,132,956 after cleaning. It stands in for ApexLend's own book.",
      why="A market-wide proxy. Results show how models behave, not ApexLend's own economics."),
    E("Data and setup", "Target", "What the models predict: whether the lender approved the application "
      "(originated, or approved but not accepted) or denied it.",
      formula=r"y_i = 1 \text{ if approved}, \quad y_i = 0 \text{ if denied}",
      project=f"The lender approved {pct(m['lender_approval_rate'])} of test applications.",
      why="The models learn the lender's decisions, not loan default, so they inherit any bias in those decisions.",
      builds=["HMDA"]),
    E("Data and setup", "Leakage", "A column that reveals the outcome, or is only filled in after the decision, "
      "and so makes a model look better than it could be in use.",
      project="Two passes of testing removed 12 columns, for example initially_payable_to_institution.",
      why="A leaked model would approve applicants in production on information it will not have."),
    E("Data and setup", "Protected attributes", "Characteristics the law protects from discrimination, such as "
      "race, ethnicity, sex and age.",
      project="Sixteen demographic and geographic columns are kept out of every model and used only in the fairness "
              "review. The lender's identity is also excluded as a proxy for race.",
      why="Excluding them is necessary but not sufficient: other inputs can carry the same information."),
    E("Data and setup", "Train, validation and test sets", "Three disjoint parts of the data: the model learns on "
      "train, choices such as the cutoff are tuned on validation, and results are reported on test.",
      project="6,394,039 / 1,369,931 / 1,368,986 applications (about 70%, 15%, 15%).",
      why="Test figures estimate performance on applications the model has never seen."),
    E("Data and setup", "Stratified split", "A split that keeps the same mix of chosen groups in every part.",
      project="Stratified by outcome, race, ethnicity, sex and age, so small groups appear in all three parts.",
      builds=["Train, validation and test sets", "Protected attributes"]),
    E("Data and setup", "Reference group", "The group every other group is compared with in a fairness ratio.",
      project="White; not Hispanic or Latino; male; aged 35 to 44.",
      builds=["Protected attributes"]),
    E("Data and setup", "Loan-to-income", "An engineered input: loan amount divided by annual income.",
      formula=r"\text{LTI}_i = \frac{L_i}{1000 \times \text{income}_i}",
      project="Added for XGBoost and TabPFN. HMDA reports income in thousands of dollars."),
    E("Data and setup", "Feature set", "The inputs a model may use.",
      project="Logistic regression uses 12 inputs. XGBoost and TabPFN use the same 31 (8 numeric, 22 categorical, "
              "plus loan-to-income), so the gap between those two comes from the learner alone.",
      why="Part of logistic regression's weaker accuracy comes from its smaller input set.",
      builds=["Leakage", "Protected attributes", "Loan-to-income"]),
    E("Data and setup", "Calibration", "A model is calibrated when its scores match observed frequencies: among "
      "applications scored 0.9, about 90% were approved.",
      formula=r"\Pr(y = 1 \mid p = s) \approx s",
      why="The 0.921 cutoff is only meaningful if scores are calibrated."),
    E("Data and setup", "Isotonic calibration", "A monotone step function fitted on held-out data that maps raw "
      "scores to calibrated probabilities.",
      project="Applied to XGBoost, fitted on 10% of its training data.",
      builds=["Calibration"]),
    E("Data and setup", "Context rows", "The labelled examples TabPFN reads at prediction time instead of learning "
      "weights.",
      project="A stratified sample of 9,898 training applications.",
      builds=["Stratified split"]),
    E("Data and setup", "In-context learning", "Predicting from examples supplied with the query, by a network "
      "pre-trained on many synthetic datasets, with no training on the client's data.",
      project="How TabPFN works. Its scores move slightly with the other applications scored in the same batch.",
      builds=["Context rows"]),

    # 2. Statistical performance
    E("Statistical performance", "Positive class", "The outcome a model scores the probability of.",
      project="Approval (y = 1). The denial class is the minority, about 22.5% of applications."),
    E("Statistical performance", "Decision threshold", "The score above which an application is approved.",
      formula=r"\hat{y}_i = \mathbb{1}\{p_i \ge t\}",
      project="0.5 for the statistical comparison; 0.921 (the break-even cutoff) for P&L and fairness.",
      builds=["Positive class"]),
    E("Statistical performance", "Confusion matrix", "The four counts of a thresholded classifier: true positives "
      "(TP), false positives (FP), true negatives (TN) and false negatives (FN).",
      formula=r"\begin{array}{c|cc} & \hat{y}=1 & \hat{y}=0 \\ \hline y=1 & TP & FN \\ y=0 & FP & TN \end{array}",
      project="Here a false positive is an approval of an application the lender denied.",
      builds=["Decision threshold", "Positive class"], abbr="TP, FP, TN, FN"),
    E("Statistical performance", "Accuracy", "Share of all applications classified correctly.",
      formula=r"\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}", project=at_half("accuracy"),
      why="Misleading when classes are unbalanced: approving everyone scores 77.5%.", builds=["Confusion matrix"]),
    E("Statistical performance", "Precision", "Share of approvals that the lender also approved.",
      formula=r"\text{Precision} = \frac{TP}{TP + FP}", project=at_half("precision"), builds=["Confusion matrix"]),
    E("Statistical performance", "Sensitivity", "Share of lender-approved applications the model approves.",
      abbr="recall, true positive rate, TPR", formula=r"\text{TPR} = \frac{TP}{TP + FN}", project=at_half("recall"),
      builds=["Confusion matrix"]),
    E("Statistical performance", "Specificity", "Share of lender-denied applications the model also denies.",
      abbr="true negative rate, TNR", formula=r"\text{TNR} = \frac{TN}{TN + FP}", project=at_half("specificity"),
      why="The share of bad approvals avoided.", builds=["Confusion matrix"]),
    E("Statistical performance", "False positive rate", "Share of lender-denied applications the model approves.",
      abbr="FPR", formula=r"\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{TNR}", project=at_half("fpr"),
      builds=["Specificity"]),
    E("Statistical performance", "F1-score", "Harmonic mean of precision and sensitivity.",
      formula=r"F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}",
      project=at_half("f1"), builds=["Precision", "Sensitivity"]),
    E("Statistical performance", "ROC curve", "True positive rate plotted against false positive rate as the "
      "threshold moves from 1 to 0.", builds=["Sensitivity", "False positive rate", "Decision threshold"]),
    E("Statistical performance", "ROC-AUC", "Area under the ROC curve: the probability that a randomly chosen "
      "approved application scores higher than a randomly chosen denied one. 0.5 is a coin toss, 1.0 is perfect.",
      abbr="AUC", formula=r"\text{AUC} = \Pr(p_i > p_j \mid y_i = 1,\, y_j = 0)",
      project="On the test set: " + listing(lambda k: m["models"][k]["auc"]) + ".",
      why="Threshold-free, so it is the fair comparison of ranking power.", builds=["ROC curve"]),
    E("Statistical performance", "Gini", "A rescaled AUC used in credit scoring.",
      formula=r"\text{Gini} = 2\,\text{AUC} - 1",
      project=listing(lambda k: 2 * m["models"][k]["auc"] - 1) + ".", builds=["ROC-AUC"]),
    E("Statistical performance", "PR curve", "Precision plotted against recall as the threshold moves.",
      builds=["Precision", "Sensitivity", "Decision threshold"]),
    E("Statistical performance", "PR-AUC", "Area under the PR curve, computed for one class at a time. For the "
      "minority class it is stricter than ROC-AUC.",
      project=f"Approval class: XGBoost {PR_AUC['xgboost'][0]:.3f}, TabPFN {PR_AUC['tabpfn'][0]:.3f}. Denial "
              f"class: XGBoost {PR_AUC['xgboost'][1]:.3f}, TabPFN {PR_AUC['tabpfn'][1]:.3f}.",
      why="The denial class is where the models separate.", builds=["PR curve", "Positive class"]),
    E("Statistical performance", "Brier score", "Mean squared gap between the score and the outcome. Lower is "
      "better; it rewards calibration and sharpness together.",
      formula=r"\text{Brier} = \frac{1}{n}\sum_i (p_i - y_i)^2",
      project=listing(lambda k: m["models"][k]["brier"]) + ".", builds=["Calibration"]),
    E("Statistical performance", "Log loss", "Average negative log-likelihood of the outcomes under the scores. "
      "It punishes confident mistakes heavily.",
      formula=r"\text{LogLoss} = -\frac{1}{n}\sum_i \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big]",
      project="TabPFN 0.321 on the test set (tabpfn notebook).", builds=["Calibration"]),

    # 3. Economic performance
    E("Economic performance", "Net interest margin", "Profit on a good loan, as a share of the loan amount.",
      abbr="m", project="3%, an industry assumption, not measured from the data."),
    E("Economic performance", "Loss rate on a bad approval", "Loss on approving an application the lender denied, "
      "as a share of the loan amount.", abbr="LGD",
      project="35%, an industry assumption. HMDA has no defaults, so a lender denial stands in for a bad loan."),
    E("Economic performance", "Break-even threshold", "The score above which approving is expected to make money.",
      abbr="0.921", extra=worked_example,
      builds=["Net interest margin", "Loss rate on a bad approval", "Decision threshold", "Calibration"]),
    E("Economic performance", "P&L", "Profit and loss of a set of decisions, in $bn, on the test set.",
      formula=r"\text{P\&L}(t) = \sum_{i:\,p_i \ge t} L_i \big[\, m\, y_i - \text{LGD}\,(1 - y_i) \big]",
      project="At 0.921: " + ", ".join(f"{names[k]} {bn(m['models'][k]['pnl'])}" for k in keys) + ".",
      builds=["Net interest margin", "Loss rate on a bad approval", "Decision threshold"]),
    E("Economic performance", "Lender's own decisions", "The benchmark: the P&L of approving exactly what the lender "
      "approved, the most a model can earn by reproducing its decisions.",
      formula=r"\text{P\&L}_{\text{lender}} = m \sum_i L_i\, y_i",
      project=f"{bn(oracle)} at {pct(m['lender_approval_rate'])} approval.", builds=["P&L"]),
    E("Economic performance", "Approve-everyone baseline", "The P&L of approving every application (cutoff 0).",
      project=f"{bn(everyone)}: denied applications cost more than approved ones earn.", builds=["P&L"]),
    E("Economic performance", "Share of the lender's P&L captured", "A model's P&L divided by the lender's own.",
      formula=r"\text{share} = \text{P\&L}_{\text{model}} \,/\, \text{P\&L}_{\text{lender}}",
      project=listing(lambda k: m["models"][k]["pnl"] / oracle, 3) + f". {names[best]} earns "
              f"{m['models'][best]['pnl'] / m['models'][runner_up]['pnl'] - 1:.0%} more than {names[runner_up]}.",
      builds=["P&L", "Lender's own decisions"]),
    E("Economic performance", "Validation-optimal threshold", "The cutoff with the highest P&L on the validation set.",
      project=f"{VAL_OPTIMAL_XGB:.3f} for XGBoost, next to the 0.921 break-even value.",
      builds=["Break-even threshold", "Train, validation and test sets"]),
    E("Economic performance", "Default share", "The share of lender-denied applicants assumed to default. 100% "
      "is the worst case and gives 0.921; a lower share lowers the cutoff.", abbr="d",
      formula=r"t^*(d) = \frac{\text{LGD}\, d}{m + \text{LGD}\, d}",
      project="An input on the Approval policy page.", builds=["Break-even threshold"]),

    # 4. Interpretability
    E("Interpretability", "Local and global explanation", "A local explanation accounts for one decision; a global "
      "one describes the model's behaviour overall."),
    E("Interpretability", "Shapley value", "From cooperative game theory: a player's average marginal contribution "
      "over all orders in which players can join.",
      formula=r"\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}"
              r"\big[v(S \cup \{j\}) - v(S)\big]"),
    E("Interpretability", "SHAP", "Shapley values applied to one prediction: each input's contribution to the gap "
      "between this score and the average score. The contributions add up exactly.",
      abbr="SHapley Additive exPlanations",
      formula=r"f(x) = \mathbb{E}[f(X)] + \sum_j \phi_j(x)",
      project="200 test rows, 100 background rows. Debt-to-income ranks first for XGBoost and TabPFN, loan purpose "
              "for logistic regression.",
      why="Supports the specific principal reasons required for a denial.",
      builds=["Shapley value", "Local and global explanation"]),
    E("Interpretability", "SHAP explainers", "TreeExplainer computes SHAP exactly from a tree model's structure; "
      "LinearExplainer exactly for a linear model; KernelExplainer approximates it for any model by sampling.",
      project="Exact for XGBoost and logistic regression; approximate (KernelExplainer) for TabPFN.",
      why="An exact reason is easier to defend to an examiner.", builds=["SHAP", "Background rows"]),
    E("Interpretability", "Background rows", "The reference applications SHAP averages over to represent "
      "'typical' input values.", project="100 rows.", builds=["SHAP"]),
    E("Interpretability", "XPER", "A Shapley decomposition of a performance metric (here AUC) rather than of a "
      "prediction: how much each input adds to the model's ranking power.",
      project="Capped at 20,000 rows, top 5 inputs; debt-to-income adds most for XGBoost and TabPFN.",
      builds=["Shapley value", "ROC-AUC"]),
    E("Interpretability", "Permutation importance", "The fall in AUC when one input's values are shuffled.",
      formula=r"I_j = \text{AUC} - \text{AUC}_{\,x_j \text{ shuffled}}",
      project="5 repeats per input. Shuffling debt-to-income costs 0.10 to 0.14 of AUC for every model.",
      builds=["ROC-AUC"]),
    E("Interpretability", "Fidelity R²", "How well a surrogate reproduces the model's scores.",
      formula=r"R^2 = 1 - \frac{\sum_i (p_i - \hat{p}_i)^2}{\sum_i (p_i - \bar{p})^2}"),
    E("Interpretability", "Global surrogate tree", "A shallow decision tree fitted to a model's own scores (not "
      "the outcomes), read as a summary of the model.",
      project=f"Depth 4 (top three levels drawn). Fidelity: {fidelity}.",
      why="A surrogate only counts as evidence if its fidelity is high.", builds=["Fidelity R²"]),
    E("Interpretability", "PDP", "Partial dependence: the average score as one input varies, others as observed.",
      abbr="partial dependence plot", formula=r"\text{PD}_j(v) = \frac{1}{n}\sum_i f(x_i \text{ with } x_{ij} = v)",
      project="Debt-to-income, loan-to-value and income."),
    E("Interpretability", "ICE", "Individual conditional expectation: one PDP curve per applicant.",
      project="50 applicant curves per plot.", why="Shows whether an effect differs from person to person.",
      builds=["PDP"]),
    E("Interpretability", "LIME", "Local interpretable model-agnostic explanations: a simple linear model fitted "
      "to perturbed copies of one application.",
      project="The highest, lowest and borderline scores; continuous inputs only.",
      why="Checks whether two explanation methods agree.", builds=["Local and global explanation"]),
    E("Interpretability", "Reason codes in this app", "The app's per-applicant factors: the change in the score "
      "when one input alone is replaced by the values of 20 typical applications (a drop-one-factor effect).",
      formula=r"e_j(x) = f(x) - \frac{1}{20}\sum_{r=1}^{20} f(x \text{ with } x_j = r_j)",
      project="Same method for all three models so they compare directly. Unlike SHAP, the effects do not add up "
              "to the score; SHAP remains the reference for adverse action reasons.",
      builds=["SHAP", "Adverse action notice"]),

    # 5. Stability
    E("Stability", "Bootstrap refit", "Retraining a model on a sample drawn with replacement from the training data.",
      project="30 refits for XGBoost and logistic regression, 10 for TabPFN, each scored on 20,000 test rows "
              "(5,000 for TabPFN)."),
    E("Stability", "Importance vector", "One number per input measuring how much the model uses it, on a shared "
      "list of inputs so different models can be compared.", builds=["Permutation importance"]),
    E("Stability", "Bootstrap importance distance", "The L2 distance between a refit's importance vector and the "
      "baseline fit's.", formula=r"d = \lVert \varphi(f_1) - \varphi(f_2) \rVert_2",
      why="Raw distances are in each model's own units and do not compare across models.",
      builds=["Importance vector", "Bootstrap refit"]),
    E("Stability", "Rank distance", "The L2 distance between the rankings of the inputs (1 = most important), "
      "which puts all models on the same scale.",
      formula=r"d_{\text{rank}} = \lVert r(\varphi_1) - r(\varphi_2) \rVert_2",
      project="Mean over refits: " + ", ".join(f"{names[k]} {STABILITY['rank'][k]:.1f}" for k in keys) + ".",
      why="Measures whether the ranking of what drives decisions survives retraining.",
      builds=["Importance vector"]),
    E("Stability", "AUC spread", "The standard deviation of test AUC across refits.",
      project="Standard deviation: " + ", ".join(f"{names[k]} {STABILITY['auc_sd'][k]:.4f}" for k in keys) + ".",
      builds=["ROC-AUC", "Bootstrap refit"]),
    E("Stability", "D1 to D2", "Fit on 50% of the training data, then on 100%, and compare the importance vectors. "
      "No resampling: it asks whether more data changes the drivers.",
      project="Rank distance: " + ", ".join(f"{names[k]} {STABILITY['d1d2'][k]:.1f}" for k in keys) + ".",
      builds=["Importance vector", "Rank distance"]),
    E("Stability", "Parameter distance", "The L2 distance between the coefficient vectors of two logistic regression "
      "fits. Trees and TabPFN have no comparable parameter vector.",
      formula=r"\lVert \theta_{\text{base}} - \theta_{\text{boot}} \rVert_2",
      project=f"Median about {STABILITY['param_median']:.2f} (mean {STABILITY['param_mean']:.2f}).",
      builds=["Bootstrap refit"]),

    # 6. Fairness
    E("Fairness", "Selection rate", "Share of a group's applications that are approved.",
      formula=r"\text{SR}_g = \Pr(\hat{y} = 1 \mid G = g)"),
    E("Fairness", "Independence", "Fairness definition: the decision does not depend on the group.",
      formula=r"\hat{y} \perp G", builds=["Selection rate"]),
    E("Fairness", "TPR and FPR by group", "True and false positive rates computed separately for each group.",
      builds=["Sensitivity", "False positive rate"]),
    E("Fairness", "Separation", "Fairness definition: equal error rates across groups once the true outcome is known.",
      formula=r"\hat{y} \perp G \mid y", builds=["TPR and FPR by group"]),
    E("Fairness", "Calibration by group", "Calibration checked separately for each group: the same score should mean "
      "the same approval rate.", project="At the same score, Black applicants were approved less often.",
      builds=["Calibration"]),
    E("Fairness", "Sufficiency", "Fairness definition: at the same score, the outcome does not depend on the group.",
      formula=r"y \perp G \mid p", builds=["Calibration by group"]),
    E("Fairness", "Demographic parity ratio", "Lowest group selection rate divided by the highest.",
      formula=r"\text{DPR} = \min_g \text{SR}_g \,/\, \max_g \text{SR}_g", builds=["Selection rate"]),
    E("Fairness", "Disparate impact ratio", "A group's selection rate divided by the reference group's.",
      abbr="DIR, adverse impact ratio",
      formula=r"\text{DIR}_g = \text{SR}_g \,/\, \text{SR}_{\text{ref}}",
      project="Black vs White at 0.921: " + race_pairs("DIR (pair)") + ".",
      builds=["Selection rate", "Reference group"]),
    E("Fairness", "Four-fifths rule", "A disparate impact ratio below 0.8 is a screening signal that triggers "
      "review. It is not a legal line.", formula=r"\text{DIR}_g < 0.8",
      project="All three models fall below it on race.", builds=["Disparate impact ratio"]),
    E("Fairness", "Equalized odds difference", "The largest gap between groups in TPR or FPR.",
      abbr="EOD", formula=r"\text{EOD} = \max\big(|\Delta\text{TPR}|,\, |\Delta\text{FPR}|\big)",
      project="Black vs White: " + race_pairs("EOD (pair)", digits=3) + ".",
      why="Inherits any bias in the lender's past decisions.", builds=["TPR and FPR by group"]),
    E("Fairness", "Conditional parity", "Selection rates compared within risk bands, to test whether risk explains "
      "a gap.", project="Five bands of debt-to-income or loan-to-value. The race gap persists in every band for "
                        "XGBoost and TabPFN.",
      builds=["Selection rate", "Chi-square test of independence", "Fisher's method"]),
    E("Fairness", "Fairness PDP", "A PDP drawn separately for each group, showing where along an input the gap sits.",
      abbr="FPDP", project="Across debt-to-income and loan-to-value.", builds=["PDP", "Selection rate"]),
    E("Fairness", "Shared decision rule", "Every model is judged at the same cutoff, so fairness gaps compare "
      "like with like.", project="All three models at 0.921, on the full test set.",
      builds=["Break-even threshold"]),

    # 7. Statistical testing
    E("Statistical testing", "Null hypothesis", "The default claim a test tries to reject, for example 'approval "
      "does not depend on race'.", abbr="H0"),
    E("Statistical testing", "p-value", "The probability of data at least as extreme as observed if the null "
      "hypothesis were true.", builds=["Null hypothesis"]),
    E("Statistical testing", "Chi-square test of independence", "Tests whether two categorical variables, such as "
      "group and decision, are independent.",
      formula=r"\chi^2 = \sum \frac{(O - E)^2}{E}",
      project="Rejects for every attribute and model (p < 0.0001).", builds=["p-value"]),
    E("Statistical testing", "Fisher's method", "Combines several p-values into one test.",
      formula=r"X^2 = -2\sum_{k=1}^{K} \ln p_k \sim \chi^2_{2K}",
      project="Combines the per-band tests of conditional parity.", builds=["p-value"]),
    E("Statistical testing", "Confidence interval", "A range that contains the true value in a stated share of "
      "repeated samples (90% here)."),
    E("Statistical testing", "TOST equivalence test", "Two one-sided tests. The null is flipped: a gap counts as "
      "unfair until its confidence interval sits inside ±δ.",
      formula=r"H_0: |\Delta| \ge \delta \quad \text{vs} \quad H_1: |\Delta| < \delta, \qquad \delta = 0.02",
      project="Black vs White approval gap (points): " + race_pairs("approval gap (P − R)", 100, 1)
              + ". None is equivalent within 2 points.",
      builds=["Confidence interval", "Null hypothesis"], abbr="two one-sided tests"),
    E("Statistical testing", "Significance at 1.37M rows", "With this many rows, a test rejects on gaps of no "
      "practical size, so p-values stop being informative.",
      why="TOST, which asks whether a gap is small enough to ignore, drives the fairness verdict.",
      builds=["p-value", "TOST equivalence test"]),

    # 8. Regulatory context
    E("Regulatory context", "ECOA and Regulation B", "The Equal Credit Opportunity Act and its implementing "
      "regulation: no credit discrimination on race, color, religion, national origin, sex, marital status or age.",
      abbr="ECOA, Reg B"),
    E("Regulatory context", "Adverse action notice", "The notice a lender must send when it denies credit, stating "
      "the specific principal reasons (usually up to four).", builds=["ECOA and Regulation B"]),
    E("Regulatory context", "Disparate impact and disparate treatment", "Treatment: using a protected attribute "
      "directly. Impact: a neutral practice that falls harder on a protected group without business justification.",
      project="No model sees a protected attribute, but gaps can travel through correlated inputs.",
      builds=["ECOA and Regulation B", "Disparate impact ratio"]),
    E("Regulatory context", "Age rule for applicants aged 62 and over", "Regulation B lets an empirically derived "
      "scoring system use age, provided applicants aged 62 or over are not scored less favourably for it.",
      builds=["ECOA and Regulation B"]),
    E("Regulatory context", "SR 11-7", "US banking supervisors' guidance on model risk management: conceptual "
      "soundness, ongoing monitoring and effective challenge."),
    E("Regulatory context", "Champion and challenger", "The model in production (champion) and an alternative run "
      "alongside it to test whether it should be replaced.", builds=["SR 11-7"]),
]

# Every "Builds on" term must have its own entry.
_terms = {e["term"] for e in G}
assert all(b in _terms for e in G for b in e["builds"]), [b for e in G for b in e["builds"] if b not in _terms]


def set_query(term: str) -> None:
    st.session_state["glossary_q"] = term


page_header("Glossary", "Every metric, measure and method used in this review, with its formula and our figures.",
            "Search, or open a category. Links under Builds on jump to the definitions an entry depends on.",
            [c for c in ["Financial", "Performance", "Fairness", "Interpretability", "Stability"]])
query = st.text_input("Search the glossary", key="glossary_q", placeholder="e.g. ROC-AUC, TOST, 0.921").strip()
st.markdown(" · ".join(f"[{c}](#{c.lower().replace(' ', '-')})" for c in CATEGORIES))

matches = [e for e in G if not query or query.lower() in " ".join(
    str(v) for v in (e["term"], e["abbr"], e["definition"], e["project"], e["why"])).lower()]
if not matches:
    st.info("No entry matches this search.")
for cat in CATEGORIES:
    entries = [e for e in matches if e["cat"] == cat]
    if not entries:
        continue
    st.header(cat, anchor=cat.lower().replace(" ", "-"))
    for e in entries:
        label = e["term"] + (f" ({e['abbr']})" if e["abbr"] else "")
        with st.expander(label, expanded=bool(query) and len(matches) <= 3):
            st.markdown(md(e["definition"]))
            for f in e["formula"]:
                st.latex(f)
            if e["extra"]:
                e["extra"]()
            if e["project"]:
                st.markdown(md(f"**In our project.** {e['project']}"))
            if e["why"]:
                st.markdown(md(f"**Why it matters for ApexLend.** {e['why']}"))
            if e["builds"]:
                cols = st.columns(len(e["builds"]) + 1)
                cols[0].caption("Builds on")
                for col, b in zip(cols[1:], e["builds"]):
                    col.button(b, key=f"{e['term']}>{b}", type="tertiary", on_click=set_query, args=(b,))
footer()
