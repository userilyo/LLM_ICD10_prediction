import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load pre-trained traditional ML model
with open('models/traditional_ml_model.pkl', 'rb') as f:
    traditional_model = pickle.load(f)

def ensemble_prediction(verified_codes: list, hierarchical_codes: list) -> list:
    """Combine predictions using ensemble techniques."""
    # Combine verified and hierarchical codes
    all_codes = list(set(verified_codes + hierarchical_codes))
    
    # Create feature vector (simple binary representation)
    feature_vector = np.zeros(len(all_codes))
    for i, code in enumerate(all_codes):
        if code in verified_codes:
            feature_vector[i] = 1
    
    # Get predictions from traditional ML model
    ml_predictions = traditional_model.predict_proba([feature_vector])[0]
    
    # Combine predictions (simple averaging)
    combined_predictions = []
    for i, code in enumerate(all_codes):
        llm_score = 1 if code in verified_codes else 0
        ml_score = ml_predictions[i]
        combined_score = (llm_score + ml_score) / 2
        combined_predictions.append((code, combined_score))
    
    # Sort and filter predictions
    combined_predictions.sort(key=lambda x: x[1], reverse=True)
    final_codes = [code for code, score in combined_predictions if score > 0.5]
    
    return final_codes
