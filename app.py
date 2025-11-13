import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ---------------------- Load Model ----------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Loan Default Prediction", page_icon="💰", layout="wide")

# ---------------------- CSS (BIG TEXT + CLEAN UI) ----------------------
st.markdown("""
<style>

.custom-box {
    background: #FFFFFF;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border: 1px solid #E0E0E0;
}

/* TITLE STYLE */
.title {
    text-align:center;
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg,#007BFF,#00C6FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 🔥 BIG LABEL TEXT */
label, .stSelectbox label, .stNumberInput label {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #000000 !important;
}

/* 🔥 INPUT TEXT (number + text input) */
input[type="text"], input[type="number"] {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #111111 !important;
    padding: 10px !important;
}

/* 🔥 SELECT BOX MAIN TEXT */
div[data-baseweb="select"] span {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #111111 !important;
}

/* 🔥 SELECT DROPDOWN OPTIONS */
div[data-baseweb="select"] div {
    font-size: 20px !important;
    color: #111111 !important;
}

/* 🔥 BUTTON BIG TEXT */
[data-testid="stButton"] button {
    font-size: 22px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.markdown("<h1 class='title'>💰 Loan Default Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---------------------- LAYOUT ----------------------
col1, col2 = st.columns([0.45, 0.55])

# ---------------------- LEFT INPUT BOX ----------------------
with col1:
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.header("Applicant Details")

    person_income = st.number_input("💼 Annual Income", min_value=0, step=1000)
    person_emp_exp = st.number_input("👷 Employment Experience (years)", min_value=0, step=1)
    person_home_ownership = st.selectbox("🏠 Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("💵 Loan Amount", min_value=0, step=1000)
    loan_int_rate = st.number_input("📊 Loan Interest Rate (%)", min_value=0.0, step=0.1)
    loan_percent_income = st.number_input("📈 Loan Percent of Income", min_value=0.0, step=0.01)
    previous_loan_defaults_on_file = st.selectbox("📁 Previous Loan Default", ["YES", "NO"])

    predict_btn = st.button("🔮 Predict Default Risk", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- RIGHT OUTPUT BOX ----------------------
with col2:
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.header("Risk Analysis Result")

    if predict_btn:

        # Prepare input
        df = pd.DataFrame([{
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'person_home_ownership': person_home_ownership,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }])

        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        scaled = scaler.transform(df)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100

        if pred == 1:
            st.error("⚠ High Risk of Default")
        else:
            st.success("✅ Low Risk — Likely to Repay")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': '%', 'font': {'size': 55}},
            title={'text': "Probability of Default", 'font': {'size': 26}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E63946" if pred == 1 else "#2A9D8F"},
                'steps': [
                    {'range': [0, 25], 'color': '#2A9D8F'},
                    {'range': [25, 50], 'color': '#E9C46A'},
                    {'range': [50, 75], 'color': '#F4A261'},
                    {'range': [75, 100], 'color': '#E63946'}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Fill the form to see prediction.")

    st.markdown("</div>", unsafe_allow_html=True)
