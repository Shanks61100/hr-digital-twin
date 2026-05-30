"""
HR Digital Twin — Interactive Character Roster (Streamlit App)
================================================================
Author  : Shashank Singh
Course  : MSc Data Analytics & Artificial Intelligence, EDHEC Business School
Supervisor: Tuba Bakici

HOW TO RUN LOCALLY
------------------
1. Make sure you have run attrition_digital_twin.py at least once
   (it must have saved the trained model files in the same folder).
2. Install Streamlit if you haven't already:
       pip install streamlit
3. Run:
       streamlit run app.py
4. Your browser will open automatically at http://localhost:8501

HOW TO SHARE WITH YOUR PROFESSOR (free public URL)
---------------------------------------------------
1. Push this folder to GitHub (or just this file + the CSV).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click "New app", point it at this file.
4. Within 2 minutes you get a public URL like:
       https://yourname-hr-digital-twin.streamlit.app
5. Send that URL to your professor — she clicks it, no installation needed.

WHAT EACH FEATURE MEANS (thesis reference)
-------------------------------------------
JobLevel       : Career level from 1 (junior / entry) to 5 (director / executive).
                 In the game this is the character's LEVEL — how far they have
                 progressed up the career ladder, not an XP accumulation.
                 A Level 1 character has room to grow; a Level 5 is at the top.

FlightRisk     : The XGBoost model's predicted probability (0-100%) that this
                 employee will leave voluntarily within the simulation horizon.
                 Rendered as the character's HP bar — a depleted HP bar means
                 the character is about to leave the party.

PromotionReady : Engineered proxy combining PerformanceRating, JobLevel, and the
                 gap between YearsInCurrentRole and YearsSinceLastPromotion.
                 Rendered as the XP bar — how close the character is to levelling
                 up. When XP is full and HP is healthy, promote them.

BurnoutMeter   : (OverTime == Yes) * 2 + (5 - WorkLifeBalance).
                 Rendered as the Fatigue meter — high fatigue means the character
                 is being overworked. Relieving overtime drops this bar and,
                 in turn, reduces FlightRisk.

StagnationIndex: YearsInCurrentRole / TotalWorkingYears, clipped to [0, 1].
                 Rendered as the Stuck indicator — a character who has spent most
                 of their career in the same role without promotion is stagnating.
                 Stagnation feeds FlightRisk over time.

CompaRatio     : MonthlyIncome / median(MonthlyIncome for the same JobRole).
                 A ratio below 1.0 means this character is underpaid relative to
                 peers. Rendered as the Gold bar — below-market pay is one of
                 Herzberg's hygiene deficits and a direct predictor of departure.

DeptTurnover   : The historical attrition rate of the character's department.
                 A proxy for Manager Friction — a team that loses people
                 frequently signals a leadership problem that affects everyone
                 in that group.
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Digital Twin — Character Roster",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Chakra Petch', sans-serif; }
.stApp { background: #070b13; color: #e9eff8; }
h1,h2,h3 { font-family: 'Chakra Petch', sans-serif; }
.card {
    background: linear-gradient(180deg,#111827,#0c1320);
    border: 1px solid #1e2a3e;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.2s;
}
.card:hover { border-color: #3dd7c4; }
.card.selected { border-color: #3dd7c4; box-shadow: 0 0 0 1px #3dd7c4, 0 6px 20px #3dd7c420; }
.card.danger { border-color: #5a2230; background: linear-gradient(180deg,#1a0a10,#0c1320); }
.card.star { border-color: #2a3a6a; background: linear-gradient(180deg,#0f1830,#0c1320); }
.tag { font-family: 'Space Mono'; font-size: 10px; font-weight: 700;
       padding: 2px 8px; border-radius: 5px; letter-spacing: 0.8px;
       display: inline-block; margin-bottom: 6px; }
.tag-risk { background: #3a1620; color: #ff5267; }
.tag-star { background: #1a2348; color: #5fd0ff; }
.tag-ok   { background: #13301f; color: #37d67a; }
.emp-name { font-size: 16px; font-weight: 700; letter-spacing: 0.3px; }
.emp-role { font-family: 'Space Mono'; font-size: 10px; color: #7e8ca6; margin-bottom: 8px; }
.mini-bar-wrap { margin: 3px 0; }
.mini-label { font-family: 'Space Mono'; font-size: 9px; color: #7e8ca6;
              display: flex; justify-content: space-between; }
.stat-section { margin: 12px 0; }
.stat-name { font-family: 'Space Mono'; font-size: 11px; color: #7e8ca6;
             letter-spacing: 0.5px; margin-bottom: 4px;
             display: flex; justify-content: space-between; }
.verdict { border-radius: 9px; padding: 12px 14px; font-family: 'Space Mono';
           font-size: 11px; line-height: 1.6; margin-top: 14px; }
.v-risk { background: #2a0f16; border: 1px solid #5a2230; color: #ffb3bd; }
.v-star { background: #121a38; border: 1px solid #2a3a6a; color: #bcd0ff; }
.v-ok   { background: #0f2418; border: 1px solid #1f4a30; color: #a8e8c2; }
.feature-box { background: #0c1320; border: 1px solid #1e2a3e; border-radius: 9px;
               padding: 10px 13px; margin: 6px 0; font-family: 'Space Mono'; font-size: 10px; color: #7e8ca6; }
.feature-box b { color: #3dd7c4; }
.metric-val { font-size: 28px; font-weight: 700; color: #e9eff8; font-family: 'Chakra Petch'; }
.metric-lbl { font-family: 'Space Mono'; font-size: 10px; color: #7e8ca6; letter-spacing: 1px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ── Model training (cached so it only runs once) ───────────────────────────────
@st.cache_resource
def train_model():
    CSV_LOCAL    = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    CSV_FALLBACK = ("https://raw.githubusercontent.com/nelson-wu/"
                    "employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    path = CSV_LOCAL if os.path.exists(CSV_LOCAL) else CSV_FALLBACK
    df   = pd.read_csv(path); df.columns=[c.strip() for c in df.columns]

    y    = (df["Attrition"]=="Yes").astype(int)
    raw  = df.drop(columns=[c for c in
           ["EmployeeCount","StandardHours","Over18","EmployeeNumber","Attrition"]
           if c in df.columns])

    dept_turn = df.groupby("Department").apply(lambda g:(g["Attrition"]=="Yes").mean())

    def engineer(d):
        d = d.copy()
        d["StagnationIndex"] = (d["YearsInCurrentRole"]/d["TotalWorkingYears"].replace(0,np.nan)).fillna(0).clip(0,1)
        d["BurnoutMeter"]    = (d["OverTime"]=="Yes").astype(int)*2+(5-d["WorkLifeBalance"])
        d["CompaRatio"]      = d["MonthlyIncome"]/d.groupby("JobRole")["MonthlyIncome"].transform("median")
        d["DeptTurnoverRate"]= d["Department"].map(dept_turn).fillna(dept_turn.mean())
        return d

    X = pd.get_dummies(engineer(raw), drop_first=True)
    cols = X.columns.tolist()
    X_tr,_,y_tr,_ = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    X_bal,y_bal   = SMOTE(random_state=42).fit_resample(X_tr,y_tr)
    model = XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
                          subsample=0.9,colsample_bytree=0.9,eval_metric="logloss",
                          random_state=42,n_jobs=-1)
    model.fit(X_bal,y_bal)
    return model, cols, df, dept_turn, engineer


# ── Score all employees ────────────────────────────────────────────────────────
@st.cache_data
def score_all(_model, cols, _df, _dept_turn, _engineer):
    df, dept_turn, engineer = _df, _dept_turn, _engineer
    raw = df.drop(columns=[c for c in
          ["EmployeeCount","StandardHours","Over18","EmployeeNumber","Attrition"]
          if c in df.columns])
    eng = engineer(raw)
    X   = pd.get_dummies(eng, drop_first=True)
    for c in cols:
        if c not in X: X[c]=0
    proba = _model.predict_proba(X[cols])[:,1]

    results = df.copy()
    results["FlightRisk"]      = (proba*100).round(1)
    results["StagnationIndex"] = (eng["StagnationIndex"]*100).round(1)
    results["BurnoutMeter"]    = eng["BurnoutMeter"]
    results["BurnoutPct"]      = (eng["BurnoutMeter"]/6*100).clip(0,100).round(1)
    results["CompaRatio"]      = eng["CompaRatio"].round(2)
    results["DeptTurnoverRate"]= (eng["DeptTurnoverRate"]*100).round(1)

    # Promotion readiness proxy (0-100)
    def promo(row):
        p = (row["PerformanceRating"]-2)*22
        p += max(0,(row["YearsInCurrentRole"]-row["YearsSinceLastPromotion"]))*3
        p += (3-min(row["JobLevel"],3))*4
        p += min(row["YearsInCurrentRole"],8)*2
        return float(np.clip(p,3,98))
    results["PromoReadiness"] = results.apply(promo,axis=1).round(1)
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_color(v):
    return "#ff5267" if v>=55 else "#ffc24b" if v>=35 else "#37d67a"
def promo_color(v):
    return "#37d67a" if v>=70 else "#ffc24b" if v>=45 else "#ff5267"
def burn_color(v):
    return "#ff5267" if v>=60 else "#ffc24b" if v>=35 else "#37d67a"

def bar_html(pct, color):
    return f"""<div style="height:10px;background:#060a11;border-radius:6px;
               overflow:hidden;border:1px solid #0b1320;margin-top:3px">
               <div style="width:{min(100,pct):.0f}%;height:100%;background:{color};
               border-radius:6px;background-image:linear-gradient(90deg,transparent,rgba(255,255,255,.15));
               background-size:20px 20px;transition:width .4s"></div></div>"""

def level_label(lvl):
    return {1:"Junior",2:"Mid-level",3:"Senior",4:"Lead",5:"Director"}.get(int(lvl),"Unknown")


# ── App ────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Chakra Petch';letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
Digital Twin <span style="color:#3dd7c4">// Character Roster</span></h1>
<p style="font-family:'Space Mono';font-size:11px;color:#7e8ca6;margin-bottom:20px">
Every employee is a game character. Select one to open their full stat sheet.
Powered by a live XGBoost flight-risk model trained on IBM HR Analytics (1,470 employees, AUC 0.83).</p>
""", unsafe_allow_html=True)

with st.spinner("Training model on IBM HR dataset (runs once)..."):
    model, cols, df, dept_turn, engineer = train_model()
    results = score_all(model, cols, df, dept_turn, engineer)

# ── Scoreboard ─────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
avg_risk  = results["FlightRisk"].mean()
at_risk   = (results["FlightRisk"]>=55).sum()
pipe_pct  = (results["PromoReadiness"]>=65).mean()*100
proj_cost = (results["FlightRisk"]/100 * results["MonthlyIncome"]*12*0.5).sum()

with c1:
    st.markdown(f'<div class="metric-lbl">AVG FLIGHT RISK</div><div class="metric-val">{avg_risk:.1f}%</div>',unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-lbl">AT-RISK EMPLOYEES</div><div class="metric-val">{at_risk}</div>',unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-lbl">PIPELINE HEALTH</div><div class="metric-val">{pipe_pct:.0f}%</div>',unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-lbl">PROJECTED ATTRITION COST</div><div class="metric-val">${proj_cost:,.0f}</div>',unsafe_allow_html=True)

st.markdown("---")

# ── Filters ─────────────────────────────────────────────────────────────────────
f1,f2,f3,f4 = st.columns([2,2,2,2])
with f1:
    search = st.text_input("Search by name or role","",placeholder="e.g. Sales, Manager...")
with f2:
    dept_filter = st.selectbox("Department",["All"]+sorted(results["Department"].unique().tolist()))
with f3:
    risk_filter = st.selectbox("Risk level",["All","At Risk (55%+)","Watch (35-55%)","Stable (<35%)"])
with f4:
    sort_by = st.selectbox("Sort by",["Flight Risk (high first)","Promotion Readiness","Name","Job Level"])

# Apply filters
filt = results.copy()
if search:
    filt = filt[filt.apply(lambda r: search.lower() in r["JobRole"].lower()
                           or str(r.name).lower() in search.lower()
                           or search.lower() in r["Department"].lower(), axis=1)]
if dept_filter != "All":
    filt = filt[filt["Department"]==dept_filter]
if risk_filter == "At Risk (55%+)":
    filt = filt[filt["FlightRisk"]>=55]
elif risk_filter == "Watch (35-55%)":
    filt = filt[(filt["FlightRisk"]>=35)&(filt["FlightRisk"]<55)]
elif risk_filter == "Stable (<35%)":
    filt = filt[filt["FlightRisk"]<35]

sort_map = {"Flight Risk (high first)":"FlightRisk","Promotion Readiness":"PromoReadiness",
            "Name":"JobRole","Job Level":"JobLevel"}
filt = filt.sort_values(sort_map[sort_by], ascending=(sort_by=="Name"))

# ── View mode toggle ──────────────────────────────────────────────────────────
if "show_all" not in st.session_state:
    st.session_state.show_all = False

# Build the display set based on view mode
sorted_by_risk = filt.sort_values("FlightRisk", ascending=False)
top10    = sorted_by_risk.head(10)
bottom10 = sorted_by_risk.tail(10).sort_values("FlightRisk")  # safest first

if st.session_state.show_all or search or dept_filter != "All" or risk_filter != "All":
    display_df   = filt
    view_label   = f"Showing all {len(filt)} employees"
    is_split_view = False
else:
    display_df   = None   # handled separately in split view
    view_label   = f"Showing Top 10 (highest risk) and Bottom 10 (lowest risk) of {len(results)} employees"
    is_split_view = True

col_info, col_btn = st.columns([3,1])
with col_info:
    st.markdown(f'<p style="font-family:\'Space Mono\';font-size:11px;color:#7e8ca6;margin-bottom:8px">'
                f'{view_label}</p>', unsafe_allow_html=True)
with col_btn:
    if not (search or dept_filter != "All" or risk_filter != "All"):
        if st.session_state.show_all:
            if st.button("Show Top & Bottom 10", use_container_width=True):
                st.session_state.show_all = False
                st.rerun()
        else:
            if st.button("Show All 1,470 Employees", use_container_width=True):
                st.session_state.show_all = True
                st.rerun()

# ── Layout: roster + detail ───────────────────────────────────────────────────
roster_col, detail_col = st.columns([3,2])

# Session state for selected employee
if "selected_idx" not in st.session_state:
    first = top10.index[0] if is_split_view else filt.index[0] if len(filt)>0 else None
    st.session_state.selected_idx = first

def render_tile_row(df_slice, cols_grid):
    """Render one row of employee tiles."""
    for col_ui, (idx, emp) in zip(cols_grid, df_slice.iterrows()):
        fr = emp["FlightRisk"]; pr = emp["PromoReadiness"]
        lvl_name = level_label(emp["JobLevel"])
        with col_ui:
            if st.button(
                f"{emp['JobRole'][:18]}\n{emp['Department'][:14]}\nRisk: {fr:.0f}%  LV{int(emp['JobLevel'])}",
                key=f"btn_{idx}", use_container_width=True
            ):
                st.session_state.selected_idx = idx
            st.markdown(
                f'<div class="mini-bar-wrap">'
                f'{bar_html(fr, risk_color(fr))}'
                f'<div class="mini-label"><span>RISK</span>'
                f'<span style="color:{risk_color(fr)}">{fr:.0f}%</span></div>'
                f'</div>', unsafe_allow_html=True)

with roster_col:
    if is_split_view:
        # TOP 10 section
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <span style="background:#3a1620;color:#ff5267;font-family:'Space Mono';font-size:10px;
          font-weight:700;padding:3px 10px;border-radius:5px;letter-spacing:1px">TOP 10 AT RISK</span>
          <span style="font-family:'Space Mono';font-size:10px;color:#7e8ca6">Highest predicted flight risk — act on these first</span>
        </div>""", unsafe_allow_html=True)
        rows_top = [top10.iloc[i:i+4] for i in range(0, len(top10), 4)]
        for row_df in rows_top:
            cols_grid = st.columns(4)
            render_tile_row(row_df, cols_grid)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        # BOTTOM 10 section
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <span style="background:#13301f;color:#37d67a;font-family:'Space Mono';font-size:10px;
          font-weight:700;padding:3px 10px;border-radius:5px;letter-spacing:1px">BOTTOM 10 STABLE</span>
          <span style="font-family:'Space Mono';font-size:10px;color:#7e8ca6">Lowest predicted flight risk — your most secure employees</span>
        </div>""", unsafe_allow_html=True)
        rows_bot = [bottom10.iloc[i:i+4] for i in range(0, len(bottom10), 4)]
        for row_df in rows_bot:
            cols_grid = st.columns(4)
            render_tile_row(row_df, cols_grid)
    else:
        # Full list view
        rows = [display_df.iloc[i:i+4] for i in range(0, len(display_df), 4)]
        for row_df in rows:
            cols_grid = st.columns(4)
            render_tile_row(row_df, cols_grid)

with detail_col:
    sel_idx = st.session_state.selected_idx
    if sel_idx is not None and sel_idx in results.index:
        emp = results.loc[sel_idx]
        fr  = emp["FlightRisk"]; pr = emp["PromoReadiness"]
        bu  = emp["BurnoutPct"]; st_= emp["StagnationIndex"]
        cp  = emp["CompaRatio"]; dt = emp["DeptTurnoverRate"]
        lvl = int(emp["JobLevel"]); age=int(emp["Age"])

        # Status
        if fr>=55:   status,vcls,verdict = "AT RISK","v-risk",f"<b>This employee is at high risk of leaving</b> — flight risk is {fr:.0f}%. The biggest contributors are likely overtime, burnout, or below-market pay. An intervention targeting their top risk factor could significantly reduce this."
        elif pr>=70 and fr<35: status,vcls,verdict = "RISING STAR","v-star",f"<b>This employee is a rising star</b> — promotion readiness is {pr:.0f}% and flight risk is low at {fr:.0f}%. They are ready to level up. Delaying promotion increases the risk of losing them to an external offer."
        else:        status,vcls,verdict = "STABLE","v-ok",f"<b>This employee is stable</b> — flight risk sits at {fr:.0f}%. No immediate action required. Monitor burnout and stagnation over the coming weeks."

        # Tag colour
        tag_cls = "tag-risk" if fr>=55 else "tag-star" if (pr>=70 and fr<35) else "tag-ok"

        st.markdown(f"""
        <div style="margin-bottom:10px">
          <span class="tag {tag_cls}">{status}</span><br>
          <span style="font-size:20px;font-weight:700">{emp['JobRole']}</span><br>
          <span style="font-family:'Space Mono';font-size:11px;color:#7e8ca6">{emp['Department']}</span>
        </div>
        <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">
          <span class="tag" style="border:1px solid #1e2a3e;color:#3dd7c4">LEVEL {lvl} — {level_label(lvl)}</span>
          <span class="tag" style="border:1px solid #1e2a3e;color:#3dd7c4">AGE {age}</span>
          <span class="tag" style="border:1px solid #1e2a3e;color:{'#ff5267' if emp['OverTime']=='Yes' else '#37d67a'}">{'OVERTIME' if emp['OverTime']=='Yes' else 'NO OVERTIME'}</span>
          <span class="tag" style="border:1px solid #1e2a3e;color:#7e8ca6">{emp['MaritalStatus'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        # Stat bars
        stats = [
            ("FLIGHT RISK (HP)",    fr,  risk_color(fr),  "XGBoost predicted probability of leaving. Lower is better."),
            ("PROMOTION READINESS (XP)", pr, promo_color(pr),"Performance + time in role vs time since last promotion. Higher is better."),
            ("BURNOUT (FATIGUE)",   bu,  burn_color(bu),  "Overtime x2 + inverse work-life balance. Lower is better."),
            ("STAGNATION",         st_,  "#ffc24b" if st_>=60 else "#7e8ca6","Years in role / total career. High = career plateau."),
            ("PAY vs MARKET",    min(cp*100,100), "#37d67a" if cp>=0.95 else "#ffc24b","Salary vs median for same role. 100% = market rate."),
            ("TEAM TURNOVER RISK", dt,   risk_color(dt),  "Historical attrition rate in this department (manager friction proxy)."),
        ]
        for name, val, color, desc in stats:
            st.markdown(f"""
            <div class="stat-section">
              <div class="stat-name"><span>{name}</span>
              <span style="color:{color};font-weight:700">{val:.0f}{'%' if '%' not in name or True else ''}</span></div>
              {bar_html(val, color)}
              <div style="font-family:'Space Mono';font-size:9px;color:#5d6b82;margin-top:3px">{desc}</div>
            </div>""", unsafe_allow_html=True)

        # Key numbers
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0">
          <div class="feature-box"><b>Monthly Income</b><br>${int(emp['MonthlyIncome']):,}</div>
          <div class="feature-box"><b>Years at Company</b><br>{int(emp['YearsAtCompany'])} yrs</div>
          <div class="feature-box"><b>Years in Role</b><br>{int(emp['YearsInCurrentRole'])} yrs</div>
          <div class="feature-box"><b>Last Promoted</b><br>{int(emp['YearsSinceLastPromotion'])} yrs ago</div>
          <div class="feature-box"><b>Education</b><br>{emp['EducationField']}</div>
          <div class="feature-box"><b>Stock Options</b><br>Level {int(emp['StockOptionLevel'])}</div>
        </div>
        <div class="verdict {vcls}">{verdict}</div>
        """, unsafe_allow_html=True)

    else:
        st.markdown('<div style="text-align:center;color:#7e8ca6;font-family:\'Space Mono\';'
                    'font-size:12px;padding:60px 20px">Select an employee from the roster</div>',
                    unsafe_allow_html=True)

# ── Feature glossary at the bottom ─────────────────────────────────────────────
with st.expander("What does each stat mean?"):
    st.markdown("""
| Stat | Game metaphor | HR meaning | Theory |
|------|--------------|------------|--------|
| **Flight Risk (HP)** | Health Points — when HP hits zero, the character leaves | XGBoost probability of voluntary departure | Mobley (1977) turnover linkages |
| **Promotion Readiness (XP)** | Experience — fill the bar to level up | Performance + time-in-role vs last promotion | Herzberg motivators: achievement and advancement |
| **Burnout (Fatigue)** | Fatigue meter — depleted stamina leads to collapse | Overtime x2 + inverse work-life balance | Herzberg hygiene: working conditions |
| **Stagnation** | Stuck status effect | Career plateau: years in role vs total career | March and Simon (1958) inducements-contributions |
| **Pay vs Market (Gold)** | Gold / resources — below market is poverty | Compa-ratio vs role median | Herzberg hygiene: salary |
| **Team Turnover Risk** | Party morale | Dept-level attrition rate as manager friction proxy | Huebner ManagerID model (future build) |
| **Level** | Character level (1-5) | JobLevel: 1 = Junior, 2 = Mid, 3 = Senior, 4 = Lead, 5 = Director | Career progression stage |
    """)
    st.info("All stats are computed from the IBM HR Analytics dataset (1,470 employees) "
            "using the trained XGBoost model. No personal data is used or stored.")
