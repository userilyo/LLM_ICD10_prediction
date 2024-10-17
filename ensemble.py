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

def ensemble_prediction(verified_codes: list, hierarchical_codes: list, gnn_probabilities: list) -> list:
    """Combine predictions using ensemble techniques."""
    # Combine verified and hierarchical codes
    all_codes = list(set(verified_codes + hierarchical_codes))
    
    # Create feature vector (simple binary representation)
    feature_vector = np.zeros(len(all_codes))
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
    
    # Combine predictions (weighted averaging)
    combined_predictions = []
    for i, code in enumerate(all_codes):
        llm_score = 1 if code in verified_codes else 0
        ml_score = ml_predictions[i] if i < len(ml_predictions) else 0.5
        gnn_score = gnn_probabilities[i] if i < len(gnn_probabilities) else 0.5
        
        # Assign weights to different models (can be adjusted based on performance)
        llm_weight, ml_weight, gnn_weight = 0.4, 0.3, 0.3
        combined_score = (llm_score * llm_weight + ml_score * ml_weight + gnn_score * gnn_weight) / (llm_weight + ml_weight + gnn_weight)
        combined_predictions.append((code, combined_score))
    
    # Sort and filter predictions
    combined_predictions.sort(key=lambda x: x[1], reverse=True)
    final_codes = [code for code, score in combined_predictions if score > 0.5]
    
    return final_codes
