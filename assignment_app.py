import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Login details -
USERNAME = st.secrets["USERNAME"]
PASSWORD = st.secrets["PASSWORD"]

# Login function -
def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Incorrect username or password")

# App starts here -
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# Main app after login -
st.title("Treatment Response Predictor")

# Logout button -
if st.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# Loading the created model
model = joblib.load("assignment_model_dt.pickle")

# Showing the df.head()
with st.expander('Showing the dataframe being used for prediction'):
    df = pd.read_csv('assignment_data.csv')
    st.dataframe(df.head(20))

# Placing inputs within the side Bar -
with st.sidebar:
    with st.form("inputs"):

# Inputs -
        gender = st.selectbox("Gender", ["Male", "Female"])
        smokinghabit = st.selectbox("Smoking Habit", [
            "Active Smoker",
            "Former Smoker (> 1 year)",
            "No Smoker (<100 cigarettes/life time)"])

        numfamlung = st.selectbox("Number of family lung cases", [0, 1, 2, 3])
        inistage = st.selectbox("Initial Stage", ["I", "II", "III", "IV", "IA", "IIB", "IIIB", "IIIC", "IVA"])
        age_group = st.selectbox("Age Group", ["18 - 39", "40 - 63", "64 - 90"])
        relprog1 = st.selectbox("Relapse/Progression", ["No progression", "Progression", "Relapse"])
        phartreat1 = st.selectbox("Pharmacological Treatment", [
            "Intravenous and oral chemotherapy",
            "Sequential chemotherapy-radiotherapy",
            "Adjuvant chemotherapy",
            "Intravenous chemotherapy",
            "Targeted oral therapy",
            "Concomitant chemotherapy-radiotherapy",
            "Immunotherapy",
            "Neoadjuvant chemotherapy"])

        schema1 = st.selectbox("Schema", [
            "Ipilimumab - Nivolumab",
            "Gefitinib",
            "Cisplatin - Ganetespib",
            "Cisplatin - Gemcitabine",
            "Carboplatin - Paclitaxel - Nivolumab",
            "Erlotinib",
            "Everolimus",
            "Carboplatin - Docetaxel",
            "Carboplatin - Pemetrexed",
            "Pembrolizumab",
            "Afatinib",
            "Carboplatin - Paclitaxel",
            "Cisplatin - Vinorelbine",
            "Carboplatin - Etoposide VP16"])

        phartreatstart_year = st.number_input("Treatment Start Year", 1990, 2030, 2020)
        phartreatstart_month = st.number_input("Treatment Start Month", 1, 12, 1)
        numcycles1 = st.number_input("Number of Cycles", 0, 100, 1)

        submitted = st.form_submit_button("Predict Treatment Response")

if submitted:

    patient = pd.DataFrame([{
        "Gender": gender,
        "AgeGroup": age_group,
        "smokinghabit": smokinghabit,
        "numfamlung": numfamlung,
        "inistage": inistage,
        "relprog1": relprog1,
        "phartreat1": phartreat1,
        "phartreatstart_year": phartreatstart_year,
        "phartreatstart_month": phartreatstart_month,
        "numcycles1": numcycles1,
        "schema1": schema1}])

# Prediction response -
    y_pred = model.predict(patient)
    proba = model.predict_proba(patient)

    st.subheader("Prediction")
    st.write(f"Predicted outcome: **{y_pred[0]}**")

    response_index = list(model.classes_).index("Response")
    response_probability = proba[0][response_index]
    st.markdown(f"Predicted probability of response: **{response_probability:.2f}**")

# Returning a readout of the given inputs
    st.subheader("Patient Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Gender:**", gender)
        st.write("**Age Group:**", age_group)
        st.write("**Smoking Habit:**", smokinghabit)
        st.write("**Initial Stage:**", inistage)

    with col2:
        st.write("**Family Lung Cases:**", numfamlung)
        st.write("**Treatment:**", phartreat1)
        st.write("**Schema:**", schema1)
        st.write("**Cycles:**", numcycles1)

# Plotting the diagram -
    st.subheader("Decision Tree Diagram")
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(model.named_steps["classifier"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax)

    st.pyplot(fig)