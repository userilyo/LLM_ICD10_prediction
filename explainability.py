from transformers import pipeline

# Load pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def explain_prediction(text: str, codes: list) -> str:
    """Provide basic explainability for the predictions."""
    # Generate a summary of the input text
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    # Create explanation
    explanation = f"Based on the summary of the medical text:\n\n{summary}\n\n"
    explanation += "The following ICD-10 codes were predicted:\n\n"
    
    for code in codes:
        # Add a brief description for each code (in a real scenario, this would come from an ICD-10 database)
        explanation += f"- {code}: [Brief description of the code would go here]\n"
    
    explanation += "\nPlease note that this is an automated prediction and should be verified by a medical professional."
    
    return explanation
