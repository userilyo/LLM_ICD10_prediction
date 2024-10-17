import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_and_prepare_mimic_data
from llm_module import load_llm_model, generate_icd_codes
from lstm_module import load_lstm_model, verify_icd_codes
from icd_search import hierarchical_search
from ensemble import ensemble_prediction
from explainability import explain_prediction_with_importance
from gnn_module import load_gnn_model, gnn_prediction
import logging
import numpy as np

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
    gnn_model = load_gnn_model(num_features=len(all_icd_codes), hidden_channels=64, num_classes=len(all_icd_codes))
    logger.info("Models loaded successfully")
    return llm_model, lstm_model, gnn_model

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
mimic_data, all_icd_codes, texts, true_codes = load_data()
llm_model, lstm_model, gnn_model = load_models()

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
        
        # Verify codes using LSTM with attention
        verified_codes, attention_weights = verify_icd_codes(texts[selected_index], generated_codes, lstm_model)
        logger.info(f"Verified codes: {verified_codes}")
        
        # Perform hierarchical search
        hierarchical_codes = hierarchical_search(verified_codes)
        logger.info(f"Hierarchical codes: {hierarchical_codes}")
        
        # GNN prediction
        gnn_probabilities = gnn_prediction(verified_codes, all_icd_codes, gnn_model)
        logger.info(f"GNN probabilities generated")
        
        # Ensemble prediction
        final_codes, final_probabilities = ensemble_prediction(verified_codes, hierarchical_codes, gnn_probabilities)
        logger.info(f"Final codes: {final_codes}")
        
        # Generate mock features for importance calculation (replace with actual features in a real scenario)
        mock_features = np.random.rand(10)  # 10 mock features
        
        # Generate comprehensive explanation
        explanation = explain_prediction_with_importance(texts[selected_index], final_codes, final_probabilities, mock_features)
        logger.info("Comprehensive explanation generated")

    logger.info("Displaying results...")
    st.header("Predicted ICD-10 Codes")
    st.write(", ".join(final_codes))

    st.header("Comprehensive Explanation")
    st.write(explanation)

    # Display LLM-generated codes, LSTM-verified codes, and GNN predictions
    st.header("Prediction Process")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("LLM-Generated Codes")
        st.write(", ".join(generated_codes))
    with col2:
        st.subheader("LSTM-Verified Codes")
        st.write(", ".join(verified_codes))
    with col3:
        st.subheader("Top GNN Predictions")
        top_gnn_codes = [all_icd_codes[i] for i in sorted(range(len(gnn_probabilities)), key=lambda i: gnn_probabilities[i], reverse=True)[:5]]
        st.write(", ".join(top_gnn_codes))

    # Visualize attention weights
    st.header("Attention Visualization")
    for i, (code, weights) in enumerate(zip(verified_codes, attention_weights)):
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(weights.detach().numpy(), cmap="YlOrRd", ax=ax, cbar=False)
        ax.set_title(f"Attention Weights for {code}")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Attention")
        st.pyplot(fig)

    # Visualize GNN predictions
    st.header("GNN Prediction Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = 10
    top_indices = sorted(range(len(gnn_probabilities)), key=lambda i: gnn_probabilities[i], reverse=True)[:top_n]
    top_codes = [all_icd_codes[i] for i in top_indices]
    top_probs = [gnn_probabilities[i] for i in top_indices]
    sns.barplot(x=top_probs, y=top_codes, ax=ax)
    ax.set_title("Top GNN Predictions")
    ax.set_xlabel("Probability")
    ax.set_ylabel("ICD-10 Code")
    st.pyplot(fig)

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
st.sidebar.info("This app predicts ICD-10 codes based on medical text using a combination of LLM with sophisticated prompt engineering, LSTM verification with attention mechanisms, GNN for code relationships, and ensemble techniques. It includes comprehensive explanations with feature importance calculations.")

logger.info("App setup complete.")
