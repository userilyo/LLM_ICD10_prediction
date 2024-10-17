from transformers import pipeline
import numpy as np

# Load pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def explain_prediction(text: str, codes: list, probabilities: list, feature_importance: dict) -> str:
    """Provide comprehensive explainability for the predictions."""
    # Generate a summary of the input text
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    # Create explanation
    explanation = f"Based on the summary of the medical text:\n\n{summary}\n\n"
    explanation += "The following ICD-10 codes were predicted:\n\n"
    
    for code, prob in zip(codes, probabilities):
        # Add a detailed explanation for each code
        explanation += f"ICD-10 Code: {code}\n"
        explanation += f"Confidence: {prob:.2f}\n"
        explanation += f"Explanation: {get_code_explanation(code)}\n"
        explanation += f"Key factors contributing to this prediction:\n"
        
        # Add feature importance information
        for feature, importance in feature_importance.get(code, {}).items():
            explanation += f"- {feature}: {importance:.3f}\n"
        
        explanation += "\n"
    
    explanation += "\nOverall explanation of the prediction process:\n"
    explanation += "1. The medical text was first summarized to extract key information.\n"
    explanation += "2. The summarized text was then processed through multiple models:\n"
    explanation += "   a) A language model to generate initial ICD-10 code candidates.\n"
    explanation += "   b) An LSTM model with attention to verify and refine the codes.\n"
    explanation += "   c) A Graph Neural Network to consider relationships between codes.\n"
    explanation += "3. The results from these models were combined using an ensemble technique.\n"
    explanation += "4. Feature importance was calculated to determine key factors for each prediction.\n"
    explanation += "\nPlease note that this is an automated prediction and should be verified by a medical professional."
    
    return explanation

def get_code_explanation(code: str) -> str:
    """Get a detailed explanation for a specific ICD-10 code."""
    # In a real-world scenario, this would fetch explanations from a comprehensive ICD-10 database
    explanations = {
        "I21.3": "ST elevation (STEMI) myocardial infarction of unspecified site. This code is used when there's evidence of a heart attack with ST elevation on the ECG, but the exact location in the heart is not specified.",
        "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris. This indicates the presence of coronary artery disease without current symptoms of chest pain.",
        "I10": "Essential (primary) hypertension. This code is used for high blood pressure without a known underlying cause.",
        "J15.9": "Unspecified bacterial pneumonia. This code is used when there's a diagnosis of bacterial pneumonia, but the specific type of bacteria is not identified.",
        "J96.01": "Acute respiratory failure with hypoxia. This indicates severe breathing difficulty with low oxygen levels in the blood.",
        "E11.9": "Type 2 diabetes mellitus without complications. This code is for diabetes that typically develops in adulthood, without mention of any specific complications.",
    }
    return explanations.get(code, "Detailed explanation not available for this code.")

def calculate_feature_importance(features: list) -> dict:
    """Calculate feature importance using a simple method."""
    # In a real scenario, this would use a more sophisticated method
    # Here, we're just using the absolute values of features as importance
    return {f"Feature_{i}": abs(value) for i, value in enumerate(features)}

def explain_prediction_with_importance(text: str, codes: list, probabilities: list, features: list) -> str:
    """Explain prediction using feature importance."""
    feature_importance = {code: calculate_feature_importance(features) for code in codes}
    return explain_prediction(text, codes, probabilities, feature_importance)
