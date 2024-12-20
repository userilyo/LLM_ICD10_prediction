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
    try:
        data = load_and_prepare_mimic_data('data/mimic_iii_sample.csv')
        logger.info("MIMIC-III data loaded successfully")
        return data
    except FileNotFoundError:
        st.error("Error: MIMIC-III sample data file not found. Please ensure the data file exists in the data directory.")
        return pd.DataFrame(), [], [], []  # Return empty data structures
    except Exception as e:
        st.error(f"Error loading MIMIC-III data: {str(e)}")
        logger.error(f"Error loading MIMIC-III data: {str(e)}")
        return pd.DataFrame(), [], [], []  # Return empty data structures

logger.info("Setting page config...")
st.set_page_config(page_title="LLM ICD-10 Code Prediction", layout="wide")

logger.info("Setting title...")
st.title("LLM ICD-10 Code Prediction")

logger.info("Loading models and data...")
# Load models and data
llm_model, lstm_model = load_models()
mimic_data, all_icd_codes, texts, true_codes = load_data()

logger.info("Setting up sidebar...")
# Sidebar for data selection
st.sidebar.title("Data Selection")
selected_index = st.sidebar.selectbox("Select a discharge note:", range(len(texts)), format_func=lambda i: f"Case {i+1}")

logger.info("Displaying main content...")
# Main content
st.header("Discharge Note")
st.write(texts[selected_index])

st.header("True ICD-10 Codes")
st.write(", ".join(true_codes[selected_index]))

logger.info("Setting up prediction button...")
if st.button("Predict ICD-10 Codes"):
    logger.info("Prediction button clicked...")
    with st.spinner("Generating predictions..."):
        # Generate codes using LLM with confidence scores
        generated_codes_with_conf = generate_icd_codes(texts[selected_index], llm_model)
        generated_codes = [code for code, _ in generated_codes_with_conf]
        generated_confidences = [conf for _, conf in generated_codes_with_conf]
        logger.info(f"Generated codes with confidence: {generated_codes_with_conf}")
        
        # Verify codes using LSTM
        verified_codes_with_conf = verify_icd_codes(texts[selected_index], generated_codes, lstm_model)
        verified_codes = [code for code, _ in verified_codes_with_conf]
        verified_confidences = [conf for _, conf in verified_codes_with_conf]
        logger.info(f"Verified codes with confidence: {verified_codes_with_conf}")
        
        # Perform hierarchical search
        hierarchical_codes = hierarchical_search(verified_codes)
        logger.info(f"Hierarchical codes: {hierarchical_codes}")
        
        # Ensemble prediction with confidence scores
        final_codes_with_conf = ensemble_prediction(verified_codes_with_conf, hierarchical_codes)
        final_codes = [code for code, _ in final_codes_with_conf]
        final_confidences = [conf for _, conf in final_codes_with_conf]
        logger.info(f"Final codes with confidence: {final_codes_with_conf}")
        
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
st.sidebar.info("This PoC predicts ICD-10 codes based on medical text present in discharge notes using a combination of LLM with sophisticated prompt engineering, LSTM verification, and ensemble techniques.")

logger.info("App setup complete.")
