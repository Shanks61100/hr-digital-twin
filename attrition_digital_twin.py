"""
Gamifying HR: A Digital Twin for Predicting Employee Attrition and Promotion Readiness
========================================================================================
Author  : Shashank Singh
Course  : MSc Data Analytics & Artificial Intelligence, EDHEC Business School
Supervisor: Tuba Bakici
Date    : May 2026

WHAT THIS SCRIPT DOES
---------------------
1. Loads and filters the IBM HR Analytics dataset (1,470 employees, 35 features).
2. Engineers four interpretable "agent health bar" features.
3. Balances the class-imbalanced training set with SMOTE.
4. Trains a Random Forest and an XGBoost classifier to predict Flight Risk (attrition).
5. Evaluates both models and compares them against a naive baseline.
6. Produces four publication-quality charts saved as PNG files.
7. Runs a what-if retention simulation on the test population.

HOW TO RUN IN VS CODE
---------------------
  Step 1: Open a terminal inside VS Code  (Terminal > New Terminal)
  Step 2: pip install -r requirements.txt
  Step 3: Place WA_Fn-UseC_-HR-Employee-Attrition.csv in the same folder
           (download free from Kaggle — link in requirements.txt)
           If the file is missing the script auto-downloads it from GitHub.
  Step 4: python attrition_digital_twin.py
  Step 5: Four chart images will appear in the same folder.

DATASET USED
------------
IBM HR Analytics Employee Attrition & Performance
- 1,470 employee records, 35 features, binary attrition label (Yes/No)
- Source: Kaggle / IBM data scientists (synthetic but realistic)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42
CSV_LOCAL    = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CSV_FALLBACK = ("https://raw.githubusercontent.com/nelson-wu/"
                "employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv")

C_BLUE  = "#2E5496"
C_TEAL  = "#3dd7c4"
C_RED   = "#ff5a6e"
C_AMBER = "#ffb020"
C_GREY  = "#8595ad"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    path = CSV_LOCAL if os.path.exists(CSV_LOCAL) else CSV_FALLBACK
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FILTER / CLEAN
# ══════════════════════════════════════════════════════════════════════════════
def filter_data(df):
    """
    Remove four zero-information columns:

      EmployeeCount   — constant value 1 for every row
      StandardHours   — constant value 80 for every row
      Over18          — every employee is 'Y'; no variation
      EmployeeNumber  — a row identifier, not a predictive feature

    No rows are removed: all 1,470 records are retained.
    The dataset has no missing values so no imputation is needed.
    """
    drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df_clean = df.drop(columns=[c for c in drop if c in df.columns])
    print(f"  Columns before : {df.shape[1]}  |  Removed : {drop}  |  After : {df_clean.shape[1]}")
    print(f"  Rows retained  : {len(df_clean)} / {len(df)}  (no rows removed)")
    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (the four agent health bars)
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    d = df.copy()
    # Career plateau — Herzberg motivator deficit
    d["StagnationIndex"] = (d["YearsInCurrentRole"] /
                            d["TotalWorkingYears"].replace(0, np.nan)).fillna(0).clip(0, 1)
    # Burnout — Herzberg hygiene strain / Mobley withdrawal trigger
    d["BurnoutMeter"] = (d["OverTime"] == "Yes").astype(int) * 2 + (5 - d["WorkLifeBalance"])
    # Pay vs peers — March & Simon inducements-contributions imbalance
    d["CompaRatio"] = d["MonthlyIncome"] / d.groupby("JobRole")["MonthlyIncome"].transform("median")
    return d


def build_matrix(df):
    y   = (df["Attrition"] == "Yes").astype(int)
    raw = df.drop(columns=["Attrition"])
    raw = engineer_features(raw)
    dept_turn = df.groupby("Department").apply(lambda g: (g["Attrition"] == "Yes").mean())
    raw["DeptTurnoverRate"] = raw["Department"].map(dept_turn).fillna(dept_turn.mean())
    X    = pd.get_dummies(raw, drop_first=True)
    return X, y, X.columns.tolist(), dept_turn


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X_te, y_te, name):
    proba = model.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(f"\n  {name}")
    print(f"  ROC-AUC   : {roc_auc_score(y_te, proba):.3f}")
    print(f"  Accuracy  : {accuracy_score(y_te, pred):.3f}   (naive baseline = {(y_te==0).mean():.3f})")
    print(f"  Precision : {precision_score(y_te, pred, zero_division=0):.3f}  (leavers)")
    print(f"  Recall    : {recall_score(y_te, pred, zero_division=0):.3f}  (leavers)")
    print(f"  F1        : {f1_score(y_te, pred, zero_division=0):.3f}  (leavers)")
    print(f"  Confusion : {confusion_matrix(y_te, pred).tolist()}")
    return proba


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARTS  (each saves a PNG and returns the path)
# ══════════════════════════════════════════════════════════════════════════════
def fig_style(ax):
    ax.set_facecolor("#f7f9fc")
    ax.spines[["top", "right"]].set_visible(False)

def plot_attrition_distribution(df, path="fig_01_attrition_distribution.png"):
    counts = df["Attrition"].value_counts()
    labels = ["Stayed (No)", "Left (Yes)"]
    values = [counts.get("No", 0), counts.get("Yes", 0)]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(labels, values, color=[C_BLUE, C_RED], width=0.5,
                  edgecolor="white", linewidth=0.5)
    ax.set_title("Attrition Class Distribution — IBM HR Dataset",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Employees")
    ax.set_ylim(0, max(values) * 1.22)
    for bar, val in zip(bars, values):
        pct = val / sum(values) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 18,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    fig_style(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path

def plot_roc_curve(y_te, proba_rf, proba_xgb, path="fig_02_roc_curve.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for proba, label, color in [
        (proba_rf,  f"Random Forest (AUC = {roc_auc_score(y_te,proba_rf):.3f})",  C_BLUE),
        (proba_xgb, f"XGBoost       (AUC = {roc_auc_score(y_te,proba_xgb):.3f})", C_TEAL),
    ]:
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=label)
    ax.plot([0, 1], [0, 1], "--", color=C_GREY, lw=1.2, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Flight Risk Models", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.5)
    fig_style(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path

def plot_feature_importance(model, cols, path="fig_03_feature_importance.png"):
    imp   = sorted(zip(cols, model.feature_importances_), key=lambda t: t[1])[-10:]
    names = [f.replace("_"," ").replace("MaritalStatus ","").replace("JobRole ","")
               .replace("Department ","Dept ").replace("BusinessTravel ","Travel ") for f,_ in imp]
    vals  = [w for _,w in imp]
    top3  = sorted(vals)[-3]
    colors = [C_RED if v >= top3 else C_BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top 10 Drivers of Flight Risk (XGBoost)", fontsize=12, fontweight="bold", pad=12)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8.5)
    fig_style(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path

def plot_simulation_results(base_cost, post_cost, spend, n_targets, path="fig_04_simulation.png"):
    labels = ["Baseline\n(no action)", "After\nIntervention"]
    values = [base_cost / 1000, post_cost / 1000]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(labels, values, color=[C_RED, C_TEAL], width=0.45,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Projected Attrition Cost ($k)")
    ax.set_title(f"What-If Simulation: Retention Intervention\n({n_targets} employees targeted)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, max(values) * 1.28)
    for bar, val, raw in zip(bars, values, [base_cost, post_cost]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"${raw:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    reduction = base_cost - post_cost
    ax.annotate(
        f"  ${reduction:,.0f} saved\n  ({100*reduction/base_cost:.1f}% reduction)",
        xy=(0.5, (values[0]+values[1])/2), xycoords=("data","data"),
        ha="center", va="center", fontsize=10, color="#1a5c3a", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#e8f8ee", ec="#37d67a", lw=1),
    )
    ax.axhline(spend/1000, color=C_AMBER, ls="--", lw=1.4,
               label=f"Intervention spend: ${spend:,.0f}")
    ax.legend(fontsize=9)
    fig_style(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*58)
    print("  HR DIGITAL TWIN — FLIGHT RISK PREDICTION + SIMULATION")
    print("="*58)

    print("\n[1] Loading dataset...")
    df = load_data()
    print(f"  Records: {len(df)}, Features: {df.shape[1]}")
    print(f"  Attrition rate: {(df['Attrition']=='Yes').mean()*100:.1f}%")

    print("\n[2] Chart 1 — class distribution")
    fig1 = plot_attrition_distribution(df)

    print("\n[3] Filtering dataset...")
    df = filter_data(df)

    print("\n[4] Engineering features and building model matrix...")
    X, y, cols, dept_turn = build_matrix(df)
    print(f"  Feature matrix: {X.shape}")

    print("\n[5] 80/20 stratified train-test split...")
    X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        X, y, df.drop(columns=["Attrition"]),
        test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"  Train rows: {len(X_tr)}  |  Test rows: {len(X_te)}")

    print("\n[6] Applying SMOTE to training set only...")
    X_bal, y_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
    print(f"  Before — Stayed: {(y_tr==0).sum()}, Left: {(y_tr==1).sum()}")
    print(f"  After  — Stayed: {(pd.Series(y_bal)==0).sum()}, Left: {(pd.Series(y_bal)==1).sum()}")

    print("\n[7] Training models...")
    rf  = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_bal, y_bal)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                        random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_bal, y_bal)

    print("\n[8] Evaluating on test set...")
    proba_rf  = evaluate(rf,  X_te, y_te, "RANDOM FOREST")
    proba_xgb = evaluate(xgb, X_te, y_te, "XGBOOST")

    print("\n[9]  Chart 2 — ROC curves")
    fig2 = plot_roc_curve(y_te, proba_rf, proba_xgb)
    print("\n[10] Chart 3 — feature importance")
    fig3 = plot_feature_importance(xgb, cols)

    print("\n[11] Running what-if simulation...")
    pop          = raw_te.reset_index(drop=True).copy()
    replace_cost = pop["MonthlyIncome"] * 12 * 0.5

    def project_cost(frame):
        d = engineer_features(frame)
        d["DeptTurnoverRate"] = frame["Department"].map(dept_turn).fillna(dept_turn.mean())
        d = pd.get_dummies(d, drop_first=True)
        for c in cols:
            if c not in d: d[c] = 0
        risk = xgb.predict_proba(d[cols])[:, 1]
        return float((risk * replace_cost).sum()), risk

    base_cost, base_risk = project_cost(pop)
    targets = base_risk >= np.quantile(base_risk, 0.75)
    post = pop.copy()
    post.loc[targets & (post["OverTime"]=="Yes"), "OverTime"] = "No"
    post.loc[targets, "MonthlyIncome"] = (post.loc[targets,"MonthlyIncome"]*1.10).round()
    post.loc[targets, "WorkLifeBalance"] = post.loc[targets,"WorkLifeBalance"].clip(upper=3)+1
    post_cost, _ = project_cost(post)
    spend = float((pop.loc[targets,"MonthlyIncome"]*0.10*12).sum())

    print(f"  Targeted      : {int(targets.sum())} employees (top-quartile risk)")
    print(f"  Baseline cost : ${base_cost:,.0f}")
    print(f"  After action  : ${post_cost:,.0f}")
    print(f"  Reduction     : ${base_cost-post_cost:,.0f} ({100*(base_cost-post_cost)/base_cost:.1f}%)")
    print(f"  Spend         : ${spend:,.0f}  |  BCR: {(base_cost-post_cost)/spend:.2f}")

    print("\n[12] Chart 4 — simulation results")
    fig4 = plot_simulation_results(base_cost, post_cost, spend, int(targets.sum()))

    print("\n" + "="*58)
    print("  DONE — charts saved:")
    for p in [fig1, fig2, fig3, fig4]: print(f"    {p}")
    print("="*58 + "\n")

if __name__ == "__main__":
    main()
