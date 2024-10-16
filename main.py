import streamlit as st
import pandas as pd
from data_processing import preprocess_text
from llm_module import generate_icd_codes
from lstm_module import verify_icd_codes
from icd_search import hierarchical_search
from ensemble import ensemble_prediction
from explainability import explain_prediction

st.set_page_config(page_title="ICD Code Predictor", layout="wide")

st.title("ICD-10 Code Prediction")

# User input
user_input = st.text_area("Enter the medical text:", height=200)

if st.button("Predict ICD Codes"):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)

        # Generate ICD codes using LLM
        llm_codes = generate_icd_codes(processed_text)

        # Verify ICD codes using LSTM
        verified_codes = verify_icd_codes(processed_text, llm_codes)

        # Perform hierarchical search
        hierarchical_codes = hierarchical_search(verified_codes)

        # Ensemble prediction
        final_codes = ensemble_prediction(verified_codes, hierarchical_codes)

        # Display results
        st.subheader("Predicted ICD-10 Codes:")
        for code in final_codes:
            st.write(f"- {code}")

        # Explainability
        explanation = explain_prediction(processed_text, final_codes)
        st.subheader("Explanation:")
        st.write(explanation)
    else:
        st.warning("Please enter some medical text.")

# Add information about the application
st.sidebar.title("About")
st.sidebar.info(
    "This application predicts ICD-10 codes based on input medical text. "
    "It uses a combination of LLM, LSTM, and traditional ML techniques to generate accurate predictions."
)
