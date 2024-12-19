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

def ensemble_prediction(verified_codes_with_conf: list, hierarchical_codes: list) -> list:
    """Combine predictions using ensemble techniques."""
    # Extract verified codes and their confidences
    verified_codes = [code for code, _ in verified_codes_with_conf]
    confidences = [conf for _, conf in verified_codes_with_conf]
    
    # Ensure we have at least one code to process
    if not verified_codes and not hierarchical_codes:
        return []
        
    # Combine verified and hierarchical codes
    all_codes = list(set(verified_codes + hierarchical_codes))
    
    # Create feature vector (simple binary representation)
    feature_vector = np.zeros(max(1, len(all_codes)))  # Ensure at least one feature
    for i, code in enumerate(all_codes):
        if code in verified_codes:
            feature_vector[i] = 1
    
    # Get predictions from traditional ML model
    try:
        if not hasattr(traditional_model, 'classes_'):
            # If the model hasn't been fitted, fit it with a dummy sample
            traditional_model.fit([feature_vector], [1])  # Dummy target
        
        ml_predictions = traditional_model.predict_proba([feature_vector])[0]
    except Exception as e:
        warnings.warn(f"Error in traditional model prediction: {str(e)}. Using default probabilities.", UserWarning)
        ml_predictions = np.full(len(all_codes), 0.5)  # Default to 0.5 probability for all codes
    
    # Combine predictions (simple averaging)
    combined_predictions = []
    for i, code in enumerate(all_codes):
        llm_score = 1 if code in verified_codes else 0
        ml_score = ml_predictions[i] if i < len(ml_predictions) else 0.5
        combined_score = (llm_score + ml_score) / 2
        combined_predictions.append((code, combined_score))
    
    # Sort and filter predictions
    combined_predictions.sort(key=lambda x: x[1], reverse=True)
    final_codes = [code for code, score in combined_predictions if score > 0.5]
    
    return final_codes
