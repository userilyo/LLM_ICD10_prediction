import streamlit as st
import pandas as pd
from data_processing import load_and_prepare_mimic_data
from llm_module import load_llm_model, generate_icd_codes
from lstm_module import load_lstm_model, verify_icd_codes
from icd_search import hierarchical_search
from ensemble import ensemble_prediction
from explainability import explain_prediction
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting app...")

# Load models
@st.cache_resource
def load_models():
    logger.info("Loading models...")
    llm_model = load_llm_model()
    lstm_model = load_lstm_model()
    logger.info("Models loaded successfully")
    return llm_model, lstm_model

# Load and prepare MIMIC-III data
@st.cache_data
def load_data():
    logger.info("Loading MIMIC-III data...")
    data = load_and_prepare_mimic_data('data/mimic_iii_sample.csv')
    logger.info("MIMIC-III data loaded successfully")
    return data

logger.info("Setting page config...")
st.set_page_config(page_title="ICD-10 Code Prediction App", layout="wide")

logger.info("Setting title...")
st.title("ICD-10 Code Prediction App")

logger.info("Loading models and data...")
# Load models and data
llm_model, lstm_model = load_models()
mimic_data, all_icd_codes, texts, true_codes = load_data()

logger.info("Setting up sidebar...")
# Sidebar for data selection
st.sidebar.title("Data Selection")
selected_index = st.sidebar.selectbox("Select a medical case:", range(len(texts)), format_func=lambda i: f"Case {i+1}")

logger.info("Displaying main content...")
# Main content
st.header("Medical Case")
st.write(texts[selected_index])

st.header("True ICD-10 Codes")
st.write(", ".join(true_codes[selected_index]))

logger.info("Setting up prediction button...")
if st.button("Predict ICD-10 Codes"):
    logger.info("Prediction button clicked...")
    with st.spinner("Generating predictions..."):
        # Generate codes using LLM with sophisticated prompt engineering
        generated_codes = generate_icd_codes(texts[selected_index], llm_model)
        logger.info(f"Generated codes: {generated_codes}")
        
        # Verify codes using LSTM
        verified_codes = verify_icd_codes(texts[selected_index], generated_codes, lstm_model)
        logger.info(f"Verified codes: {verified_codes}")
        
        # Perform hierarchical search
        hierarchical_codes = hierarchical_search(verified_codes)
        logger.info(f"Hierarchical codes: {hierarchical_codes}")
        
        # Ensemble prediction
        final_codes = ensemble_prediction(verified_codes, hierarchical_codes)
        logger.info(f"Final codes: {final_codes}")
        
        # Generate explanation
        explanation = explain_prediction(texts[selected_index], final_codes)
        logger.info("Explanation generated")

    logger.info("Displaying results...")
    st.header("Predicted ICD-10 Codes")
    st.write(", ".join(final_codes))

    st.header("Explanation")
    st.write(explanation)

    # Display LLM-generated codes and LSTM-verified codes
    st.header("Prediction Process")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LLM-Generated Codes")
        st.write(", ".join(generated_codes))
    with col2:
        st.subheader("LSTM-Verified Codes")
        st.write(", ".join(verified_codes))

    # Calculate and display metrics
    true_set = set(true_codes[selected_index])
    pred_set = set(final_codes)
    precision = len(true_set.intersection(pred_set)) / len(pred_set) if pred_set else 0
    recall = len(true_set.intersection(pred_set)) / len(true_set) if true_set else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    st.header("Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.2f}")
    col2.metric("Recall", f"{recall:.2f}")
    col3.metric("F1 Score", f"{f1_score:.2f}")

logger.info("Setting up sidebar info...")
st.sidebar.title("About")
st.sidebar.info("This app predicts ICD-10 codes based on medical text using a combination of LLM with sophisticated prompt engineering, LSTM verification, and ensemble techniques.")

logger.info("App setup complete.")
