import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings

# Load pre-trained traditional ML model or create a dummy model
try:
    with open('models/traditional_ml_model.pkl', 'rb') as f:
        traditional_model = pickle.load(f)
except FileNotFoundError:
    warnings.warn("Traditional ML model file not found. Using a dummy RandomForestClassifier instead.", UserWarning)
    traditional_model = RandomForestClassifier(n_estimators=10, random_state=42)

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
    if isinstance(traditional_model, RandomForestClassifier) and not traditional_model.n_features_in_:
        # If using dummy model, fit it with the current feature vector
        traditional_model.fit([feature_vector], [1])  # Dummy target
    
    ml_predictions = traditional_model.predict_proba([feature_vector])[0]
    
    # Combine predictions (simple averaging)
    combined_predictions = []
    for i, code in enumerate(all_codes):
        llm_score = 1 if code in verified_codes else 0
        ml_score = ml_predictions[i] if len(ml_predictions) > i else 0.5  # Use 0.5 if ml_score is not available
        combined_score = (llm_score + ml_score) / 2
        combined_predictions.append((code, combined_score))
    
    # Sort and filter predictions
    combined_predictions.sort(key=lambda x: x[1], reverse=True)
    final_codes = [code for code, score in combined_predictions if score > 0.5]
    
    return final_codes
