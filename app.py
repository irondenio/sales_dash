import streamlit as st
import pandas as pd
from data_processing import load_sales_data
from src.analysis import descriptive, diagnostic, exploratory, inferential, predictive, prescriptive
from src.modeling import train_model, evaluate_model

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

st.title("📊 Sales Analytics Dashboard")

# Load data (cached)
@st.cache_data
def get_data():
    return load_sales_data(data_dir="./Ventes")

df = get_data()

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Descriptive", "Diagnostic", "Exploratory", "Inferential", "Predictive", "Prescriptive"])

if section == "Descriptive":
    st.subheader("Descriptive Statistics")
    result = descriptive(df)
    st.dataframe(result["stats"])
    st.plotly_chart(result["fig"], use_container_width=True)

elif section == "Diagnostic":
    st.subheader("Diagnostic Correlations")
    # Assume "Montant" is the target
    result = diagnostic(df, target="Montant")
    st.plotly_chart(result["fig"], use_container_width=True)

elif section == "Exploratory":
    st.subheader("Exploratory Analysis")
    result = exploratory(df)
    for fig in result["figs"]:
        st.plotly_chart(fig, use_container_width=True)

elif section == "Inferential":
    st.subheader("Inferential Analysis (Group by Target)")
    result = inferential(df, target="Montant")
    st.dataframe(result["group_stats"]) 
    st.plotly_chart(result["fig"], use_container_width=True)

elif section == "Predictive":
    st.subheader("Predictive Modeling")
    model, (X_test, y_test) = train_model(df, target="Montant")
    metrics = evaluate_model(model, X_test, y_test, task="regression")
    st.write("**Model performance**")
    st.json(metrics)
    # Show predictions on test set
    st.write("**Sample predictions**")
    preds = model.predict(X_test.head())
    st.write(preds)

elif section == "Prescriptive":
    st.subheader("Prescriptive Recommendations")
    model, _ = train_model(df, target="Montant")
    result = prescriptive(df, target="Montant", model=model)
    st.write("**Top features**")
    for feat, imp in result["top_features"]:
        st.write(f"- {feat}: {imp:.2f}")
    st.write("**Recommendations**")
    for rec in result["recommendations"]:
        st.write(f"- {rec}")

st.sidebar.markdown("---")
st.sidebar.info("Data source: `Ventes` folder in the repository.")
