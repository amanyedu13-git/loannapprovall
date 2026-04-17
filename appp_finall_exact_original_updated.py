import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="LoanIQ — AI Loan Predictor", page_icon="🏦", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "result" not in st.session_state:
    st.session_state.result = None

@st.cache_resource
def load_model():
    if os.path.exists('best_model.pkl'):
        return joblib.load('best_model.pkl')
    return None

model = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f1117; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.landing-hero { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border: 1px solid #2a2f3e; border-radius: 20px; padding: 4rem 2rem; text-align: center; margin-bottom: 2rem; }
.landing-hero h1 { font-size: 3.5rem; color: #f0c040; margin-bottom: 0.5rem; }
.landing-hero p { color: #8892a4; font-size: 1.2rem; font-weight: 300; margin-bottom: 2rem; }
.feature-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 2rem 0; }
.feature-item { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 1.5rem; text-align: center; }
.feature-item .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-item h4 { color: #f0c040; margin-bottom: 0.3rem; font-size: 0.95rem; }
.feature-item p { color: #8892a4; font-size: 0.82rem; margin: 0; }
.stat-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
.stat-box { background: #0f1117; border: 1px solid #2a2f3e; border-radius: 10px; padding: 1.2rem; text-align: center; }
.stat-box .val { font-size: 1.8rem; font-weight: 700; color: #f0c040; }
.stat-box .lbl { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.05em; }
.card { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 1.8rem; margin-bottom: 1.5rem; }
.card h3 { color: #f0c040; font-size: 1rem; margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase; font-family: 'DM Sans', sans-serif; font-weight: 500; }
.app-header { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border: 1px solid #2a2f3e; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 2rem; text-align: center; }
.app-header h2 { color: #f0c040; font-size: 1.8rem; margin: 0; }
.app-header p { color: #8892a4; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.dash-approved { background: linear-gradient(135deg, #0d2b1f, #1a3a2a); border: 2px solid #2ecc71; border-radius: 20px; padding: 2.5rem; text-align: center; margin-bottom: 1.5rem; }
.dash-rejected { background: linear-gradient(135deg, #2b0d0d, #3a1a1a); border: 2px solid #e74c3c; border-radius: 20px; padding: 2.5rem; text-align: center; margin-bottom: 1.5rem; }
.dash-approved h1 { color: #2ecc71; font-size: 2.5rem; margin-bottom: 0.3rem; }
.dash-rejected h1 { color: #e74c3c; font-size: 2.5rem; margin-bottom: 0.3rem; }
.dash-approved p, .dash-rejected p { color: #aab; margin: 0.3rem 0; }
.metric-box { background: #0f1117; border: 1px solid #2a2f3e; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-box .value { font-size: 1.3rem; font-weight: 700; color: #f0c040; }
.metric-box .label { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.05em; }
.score-bar-bg { background: #0f1117; border-radius: 10px; height: 16px; margin-top: 8px; overflow: hidden; }
.score-bar-fill { height: 16px; border-radius: 10px; }
.score-section { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
.score-section h3 { color: #f0c040; font-family: 'DM Sans', sans-serif; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }
.tip-box { background: #1a1f2e; border-left: 3px solid #f0c040; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-top: 0.8rem; color: #8892a4; font-size: 0.9rem; }
.info-note { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem; color: #8892a4; font-size: 0.82rem; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label { color: #c8d0de !important; font-weight: 400; }
.stButton > button { background: linear-gradient(135deg, #f0c040, #e0a820); color: #0f1117; font-weight: 700; font-size: 1.1rem; border: none; border-radius: 10px; padding: 0.75rem 2rem; width: 100%; cursor: pointer; font-family: 'DM Sans', sans-serif; }
.stButton > button:hover { box-shadow: 0 8px 25px rgba(240, 192, 64, 0.3); }
</style>
""", unsafe_allow_html=True)


def landing_page():
    st.markdown("""
    <div class="landing-hero">
        <h1>🏦 LoanIQ</h1>
        <p>AI-powered loan approval predictor — know your chances in seconds</p>
        <div class="stat-row">
            <div class="stat-box"><div class="val">96.5%</div><div class="lbl">Model Accuracy</div></div>
            <div class="stat-box"><div class="val">15+</div><div class="lbl">Factors Analyzed</div></div>
            <div class="stat-box"><div class="val">4269</div><div class="lbl">Records Trained</div></div>
        </div>
    </div>
    <div class="feature-grid">
        <div class="feature-item"><div class="icon">🤖</div><h4>AI Prediction</h4><p>Random Forest with 98.9% ROC-AUC score</p></div>
        <div class="feature-item"><div class="icon">📊</div><h4>Bank-Style Scoring</h4><p>Real bank factors — CIBIL, FOIR, Employment & more</p></div>
        <div class="feature-item"><div class="icon">💡</div><h4>Smart Tips</h4><p>Personalized advice to improve your loan profile</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Check My Loan Eligibility"):
        st.session_state.page = "app"
        st.rerun()
    st.markdown('<div style="text-align:center; color:#4a5060; font-size:0.85rem; margin-top:2rem;">Built with ❤️ using Random Forest & Streamlit</div>', unsafe_allow_html=True)


def app_page():
    if st.button("← Back to Home"):
        st.session_state.page = "landing"
        st.session_state.result = None
        st.rerun()

    st.markdown('<div class="app-header"><h2>🏦 Loan Approval Predictor</h2><p>Fill in your details to get instant AI prediction</p></div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found! Please ensure best_model.pkl is in the same folder.")
        st.stop()

    # ── SECTION 1: Personal Details ──────────────────────────────────────
    st.markdown('<div class="card"><h3>👤 Personal Details</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with col3:
        dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
    col4, col5 = st.columns(2)
    with col4:
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    with col5:
        city_tier = st.selectbox("City / Location", ["Metro (Mumbai, Delhi, Bangalore…)", "Tier-2 (Pune, Jaipur, Surat…)", "Tier-3 / Rural"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 2: Employment Details ────────────────────────────────────
    st.markdown('<div class="card"><h3>💼 Employment Details</h3>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner", "Freelancer / Consultant"])
    with col7:
        company_type = st.selectbox("Company Type", ["Government / PSU", "MNC", "Listed Private Company", "Unlisted Private / Startup", "N/A (Self-Employed)"])
    col8, col9 = st.columns(2)
    with col8:
        job_years = st.number_input("Years in Current Job / Business", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
    with col9:
        industry = st.selectbox("Industry", ["IT / Software", "Banking / Finance", "Government / Defense", "Healthcare / Pharma", "Manufacturing", "Retail / Trade", "Real Estate", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 3: Financial Details ─────────────────────────────────────
    st.markdown('<div class="card"><h3>💰 Financial Details</h3>', unsafe_allow_html=True)
    col10, col11 = st.columns(2)
    with col10:
        monthly_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=1000000, value=50000, step=5000)
        income = monthly_income * 12
        st.caption(f"Annual Income: ₹{income:,.0f}")
    with col11:
        loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=100000, max_value=50000000, value=1500000, step=100000)
    col12, col13 = st.columns(2)
    with col12:
        existing_loans = st.number_input("Total Existing Loan Amount (₹)", min_value=0, max_value=10000000, value=0, step=50000)
    with col13:
        existing_emis = st.number_input("Total Existing Monthly EMIs (₹)", min_value=0, max_value=500000, value=0, step=1000,
                                         help="Sum of all current EMIs you are paying every month")
    cibil = st.slider("CIBIL Score", min_value=300, max_value=900, value=700, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 4: Loan Details ───────────────────────────────────────────
    st.markdown('<div class="card"><h3>📋 Loan Details</h3>', unsafe_allow_html=True)
    col14, col15, col16 = st.columns(3)
    with col14:
        loan_type = st.selectbox("Loan Type", ["Home Loan", "Personal Loan", "Car Loan", "Education Loan", "Business Loan", "Gold Loan", "Loan Against Property"])
    with col15:
        loan_tenure = st.selectbox("Loan Tenure", ["1 Year", "2 Years", "3 Years", "5 Years", "10 Years", "15 Years", "20 Years", "30 Years"])
    with col16:
        down_payment_pct = st.selectbox("Down Payment (%)", ["0%", "10%", "20%", "30%", "40%", "50%+"],
                                         help="% of total cost you are paying upfront (relevant for Home/Car loans)")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 5: Assets & Collateral ───────────────────────────────────
    st.markdown('<div class="card"><h3>🏠 Assets & Collateral</h3>', unsafe_allow_html=True)
    col17, col18 = st.columns(2)
    with col17:
        property_owned = st.selectbox("Property / House Owned?", ["Yes — Self-owned", "No — Rented", "No — Living with family"])
    with col18:
        collateral = st.selectbox("Collateral Available?", ["Yes — Property", "Yes — FD / Investments", "Yes — Gold", "No Collateral"])
    col19, col20 = st.columns(2)
    with col19:
        existing_investments = st.number_input("Savings / Investments (₹)", min_value=0, max_value=50000000, value=0, step=10000,
                                                help="FDs, mutual funds, stocks, PPF etc.")
    with col20:
        property_value = st.number_input("Property / Asset Value (₹) — if any", min_value=0, max_value=100000000, value=0, step=100000)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Calculated Metrics ────────────────────────────────────────────────
    effective_loan     = loan_amount + existing_loans
    loan_income_ratio  = effective_loan / income
    new_emi            = (loan_amount / 60)
    total_monthly_emi  = existing_emis + new_emi
    emi_income_ratio   = total_monthly_emi / monthly_income   # FOIR
    income_per_person  = income / (dependents + 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="value">{cibil}</div><div class="label">CIBIL Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="value">{loan_income_ratio:.1f}x</div><div class="label">Loan/Income</div></div>', unsafe_allow_html=True)
    with c3:
        foir_pct = emi_income_ratio * 100
        st.markdown(f'<div class="metric-box"><div class="value">{foir_pct:.1f}%</div><div class="label">FOIR</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="value">₹{income_per_person/12:,.0f}</div><div class="label">Income/Person/Mo</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Loan Approval"):

        # categorical encoding
        edu_encoded = 1 if education == "Graduate" else 0
        self_emp_encoded = 1 if employment_type in [
            "Self-Employed",
            "Business Owner",
            "Freelancer / Consultant"
        ] else 0

        # loan tenure numeric conversion
        tenure_map = {
            "1 Year": 1,
            "2 Years": 2,
            "3 Years": 3,
            "5 Years": 5,
            "10 Years": 10,
            "15 Years": 15,
            "20 Years": 20,
            "30 Years": 30
        }
        loan_term = tenure_map[loan_tenure]

        # NEW engineered feature
        total_assets = existing_investments + property_value

        # latest notebook engineered features
        loan_income_ratio = loan_amount / income
        emi_income_ratio = ((loan_amount / (loan_term * 12)) + existing_emis) / monthly_income
        income_per_person = income / (dependents + 1)

        # FINAL MODEL INPUT → exact notebook feature order
        user_input = np.array([[
            dependents,
            edu_encoded,
            self_emp_encoded,
            income,
            loan_amount,
            loan_term,
            cibil,
            loan_income_ratio,
            emi_income_ratio,
            income_per_person,
            total_assets
        ]])

        prediction = model.predict(user_input)[0]
        proba = model.predict_proba(user_input)[0]

        st.session_state.result = {
            "prediction": int(prediction),
            "confidence": float(max(proba) * 100),
            # personal
            "age": age, "education": education, "dependents": dependents,
            "marital_status": marital_status, "city_tier": city_tier,
            # employment
            "employment_type": employment_type, "company_type": company_type,
            "job_years": job_years, "industry": industry,
            # financial
            "income": income, "monthly_income": monthly_income,
            "loan_amount": loan_amount, "existing_loans": existing_loans,
            "existing_emis": existing_emis, "cibil": cibil,
            # loan
            "loan_type": loan_type, "loan_tenure": loan_tenure,
            "down_payment_pct": down_payment_pct,
            # assets
            "property_owned": property_owned, "collateral": collateral,
            "existing_investments": existing_investments,
            "property_value": property_value,
            # computed
            "effective_loan": effective_loan,
            "loan_income_ratio": loan_income_ratio,
            "emi_income_ratio": emi_income_ratio,
            "income_per_person": income_per_person,
            "total_assets": total_assets,
            "edu_encoded": edu_encoded,
        }

        st.session_state.page = "dashboard"
        st.rerun()


def dashboard_page():
    r = st.session_state.result
    col_back, col_new = st.columns(2)
    with col_back:
        if st.button("← Back to Home"):
            st.session_state.page = "landing"; st.session_state.result = None; st.rerun()
    with col_new:
        if st.button("🔄 Try Again"):
            st.session_state.page = "app"; st.session_state.result = None; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Verdict ────────────────────────────────────────────────────────
    if r["prediction"] == 1:
        st.markdown(f'<div class="dash-approved"><h1>✅ Loan Approved!</h1><p style="font-size:1.1rem;">AI Confidence: <strong>{r["confidence"]:.1f}%</strong></p><p>Based on your profile, your loan is likely to be approved.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="dash-rejected"><h1>❌ Loan Rejected</h1><p style="font-size:1.1rem;">AI Confidence: <strong>{r["confidence"]:.1f}%</strong></p><p>Based on your profile, your loan may not be approved.</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="value">{r["cibil"]}</div><div class="label">CIBIL Score</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="value">{r["loan_income_ratio"]:.1f}x</div><div class="label">Loan/Income</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="value">{r["emi_income_ratio"]*100:.1f}%</div><div class="label">FOIR</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="value">₹{r["income_per_person"]/12:,.0f}</div><div class="label">Income/Person/Mo</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # BANK-STYLE RISK SCORE  (100 points — aligned with real bank criteria)
    # ══════════════════════════════════════════════════════════════════════
    score = 0
    score_details = []

    # 1. CIBIL Score — 40 pts  (model: 80.4% importance → biggest weight)
    if r["cibil"] >= 750:
        score += 40; score_details.append(("CIBIL", 40, 40, "✅"))
    elif r["cibil"] >= 700:
        score += 28; score_details.append(("CIBIL", 28, 40, "🟡"))
    elif r["cibil"] >= 650:
        score += 15; score_details.append(("CIBIL", 15, 40, "🟡"))
    else:
        score_details.append(("CIBIL", 0, 40, "❌"))

    # 2. FOIR (EMI/Income Ratio) — 15 pts  (real banks cap at 40–50%)
    foir = r["emi_income_ratio"]
    if foir <= 0.30:
        score += 15; score_details.append(("FOIR", 15, 15, "✅"))
    elif foir <= 0.40:
        score += 10; score_details.append(("FOIR", 10, 15, "🟡"))
    elif foir <= 0.50:
        score += 5;  score_details.append(("FOIR", 5, 15, "🟡"))
    else:
        score_details.append(("FOIR", 0, 15, "❌"))

    # 3. Loan / Income Ratio — 10 pts
    lir = r["loan_income_ratio"]
    if lir < 3:
        score += 10; score_details.append(("Loan Ratio", 10, 10, "✅"))
    elif lir < 5:
        score += 6;  score_details.append(("Loan Ratio", 6, 10, "🟡"))
    elif lir < 8:
        score += 3;  score_details.append(("Loan Ratio", 3, 10, "🟡"))
    else:
        score_details.append(("Loan Ratio", 0, 10, "❌"))

    # 4. Employment Type — 10 pts  (banks strongly prefer stable employment)
    emp = r["employment_type"]
    ctype = r["company_type"]
    if emp == "Salaried" and ctype in ["Government / PSU", "MNC"]:
        score += 10; score_details.append(("Employment", 10, 10, "✅"))
    elif emp == "Salaried":
        score += 7;  score_details.append(("Employment", 7, 10, "🟡"))
    elif emp in ["Self-Employed", "Business Owner"]:
        score += 5;  score_details.append(("Employment", 5, 10, "🟡"))
    else:
        score += 3;  score_details.append(("Employment", 3, 10, "🟡"))

    # 5. Job / Business Stability — 8 pts
    jy = r["job_years"]
    if jy >= 5:
        score += 8; score_details.append(("Job Stability", 8, 8, "✅"))
    elif jy >= 3:
        score += 6; score_details.append(("Job Stability", 6, 8, "🟡"))
    elif jy >= 1:
        score += 3; score_details.append(("Job Stability", 3, 8, "🟡"))
    else:
        score_details.append(("Job Stability", 0, 8, "❌"))

    # 6. Income per Person — 7 pts
    ipp = r["income_per_person"]
    if ipp > 400000:
        score += 7; score_details.append(("Inc/Person", 7, 7, "✅"))
    elif ipp > 200000:
        score += 5; score_details.append(("Inc/Person", 5, 7, "🟡"))
    elif ipp > 100000:
        score += 3; score_details.append(("Inc/Person", 3, 7, "🟡"))
    else:
        score_details.append(("Inc/Person", 0, 7, "❌"))

    # 7. Collateral / Assets — 5 pts
    col_val = r["collateral"]
    prop_val = r["property_value"]
    inv_val = r["existing_investments"]
    if col_val in ["Yes — Property", "Yes — FD / Investments"] or prop_val > 0 or inv_val > 500000:
        score += 5; score_details.append(("Collateral", 5, 5, "✅"))
    elif col_val == "Yes — Gold" or inv_val > 0:
        score += 3; score_details.append(("Collateral", 3, 5, "🟡"))
    else:
        score_details.append(("Collateral", 0, 5, "❌"))

    # 8. Down Payment — 3 pts  (higher = lower bank risk)
    dp = r["down_payment_pct"]
    if dp in ["30%", "40%", "50%+"]:
        score += 3; score_details.append(("Down Pay", 3, 3, "✅"))
    elif dp == "20%":
        score += 2; score_details.append(("Down Pay", 2, 3, "🟡"))
    elif dp == "10%":
        score += 1; score_details.append(("Down Pay", 1, 3, "🟡"))
    else:
        score_details.append(("Down Pay", 0, 3, "❌"))

    # 9. Location — 2 pts
    ct = r["city_tier"]
    if "Metro" in ct:
        score += 2; score_details.append(("Location", 2, 2, "✅"))
    elif "Tier-2" in ct:
        score += 1; score_details.append(("Location", 1, 2, "🟡"))
    else:
        score_details.append(("Location", 0, 2, "❌"))

    # Total = 100
    total_max = 100
    pct = score / total_max * 100
    bar_color = "#2ecc71" if pct >= 65 else ("#f0c040" if pct >= 45 else "#e74c3c")
    profile_label = "Strong Profile 💪" if pct >= 65 else ("Moderate Profile 🤔" if pct >= 45 else "Weak Profile ⚠️")

    st.markdown(f"""
    <div class="score-section">
        <h3>📊 Bank-Style Risk Score: {score}/{total_max} — {profile_label}</h3>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct:.0f}%; background:{bar_color};"></div></div>
    </div>""", unsafe_allow_html=True)

    # Score breakdown grid
    cols = st.columns(len(score_details))
    for i, (label, got, total, icon) in enumerate(score_details):
        with cols[i]:
            st.markdown(f'<div class="metric-box"><div class="value" style="font-size:0.9rem;">{icon} {got}/{total}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    # ── Personalised Tips ─────────────────────────────────────────────────
    tips = []

    if r["cibil"] < 700:
        tips.append("💳 Pay all existing EMIs on time for 6–12 months to push CIBIL above 750 — this single factor has the biggest impact.")
    if r["cibil"] < 650:
        tips.append("🚫 CIBIL below 650 — most banks will auto-reject. Focus on credit repair before applying.")
    if foir > 0.50:
        tips.append("📉 FOIR above 50% is a red flag for banks. Clear existing loans/EMIs before taking a new one.")
    elif foir > 0.40:
        tips.append("⚠️ FOIR is between 40–50%. Banks prefer below 40%. Try reducing existing EMI burden.")
    if lir > 8:
        tips.append("💰 Loan amount is very high vs income. Consider a lower loan amount or longer tenure to reduce EMI.")
    if jy < 1:
        tips.append("🏢 Less than 1 year in current job — banks prefer 2+ years. Wait if possible before applying.")
    if r["collateral"] == "No Collateral" and r["loan_amount"] > 500000:
        tips.append("🏠 No collateral for a large loan. A secured loan (against property/FD) will get better rates & approval odds.")
    if "Tier-3" in r["city_tier"] or "Rural" in r["city_tier"]:
        tips.append("📍 Rural / Tier-3 applicants may find better options with cooperative banks or NBFCs vs private banks.")
    if r["employment_type"] == "Freelancer / Consultant":
        tips.append("📄 Freelancers face extra scrutiny. Keep 2 years of ITR, bank statements, and invoices ready.")
    if r["down_payment_pct"] == "0%":
        tips.append("💵 Zero down payment signals higher risk. Even 10–20% upfront significantly improves approval chances.")
    if r["existing_investments"] == 0 and r["property_value"] == 0:
        tips.append("📈 No assets on record. Building FDs or investments (even small ones) improves your financial profile.")
    if r["loan_type"] in ["Home Loan", "Loan Against Property"] and r["property_value"] == 0:
        tips.append("🏠 For a Home/LAP loan, the property value is crucial. Ensure it is entered correctly for accurate assessment.")

    if tips:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="score-section"><h3>💡 Personalised Tips</h3>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; color:#4a5060; font-size:0.85rem; margin-top:2rem;">Built with ❤️ using Random Forest & Streamlit</div>', unsafe_allow_html=True)


if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "app":
    app_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
