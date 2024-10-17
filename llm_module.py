from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_llm_model():
    """Load a smaller pre-trained model for ICD code generation."""
    model_name = "facebook/bart-base"  # Using a smaller model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info("Model loaded successfully")
    return tokenizer, model

def generate_prompt(text: str) -> str:
    """Generate a sophisticated prompt for ICD-10 code prediction."""
    logger.info("Generating sophisticated prompt...")
    prompt = f"""You are an expert medical coder with extensive knowledge of ICD-10 codes. Your task is to analyze the following medical text and provide accurate ICD-10 codes with detailed explanations. Follow these guidelines:

1. Carefully read and understand the entire medical text.
2. Identify key symptoms, diagnoses, procedures, and treatments mentioned.
3. Assign the most specific ICD-10 codes possible based on the information provided.
4. Consider the context and any potential complications or underlying conditions.
5. Provide a detailed explanation for each assigned code, including your reasoning process.
6. List codes in order of relevance to the primary condition or reason for visit.
7. If applicable, include any relevant external cause codes (V00-Y99) for injuries or adverse effects.
8. Be aware of combination codes that classify two diagnoses together or a diagnosis with an associated complication.

Medical Text:
{text}

Please provide your response in the following format:
ICD-10 Code: [code]
Description: [brief description of the code]
Explanation: [detailed explanation for assigning this code, including any relevant guidelines or conventions applied]

Example:
ICD-10 Code: I21.3
Description: ST elevation (STEMI) myocardial infarction of unspecified site
Explanation: The patient was diagnosed with acute myocardial infarction, and the ECG showed ST-segment elevation. This code is the most specific for STEMI when the exact site is not specified in the given information.

ICD-10 Code: R07.4
Description: Chest pain, unspecified
Explanation: The patient was admitted with severe chest pain, which is a key symptom. This code is used to capture the chest pain symptom separately from the underlying cause.

Repeat this format for each relevant ICD-10 code you identify. Provide at least 3 relevant codes if possible, but do not include codes that are not strongly supported by the given information."""

    logger.info(f"Generated prompt: {prompt}")
    return prompt

def parse_output(output: str) -> list:
    """Parse the output to extract ICD-10 codes."""
    logger.info("Parsing LLM output...")
    logger.info(f"Raw output: {output}")
    codes = re.findall(r"ICD-10 Code: ([A-Z0-9.]+)", output)
    
    if not codes:
        logger.warning("No ICD-10 codes found in the output.")
    else:
        logger.info(f"Parsed codes: {codes}")
    return codes

def generate_icd_codes(text: str, model_tuple, max_length: int = 1024) -> list:
    """Generate ICD-10 codes using a pre-trained LLM with sophisticated prompt engineering."""
    logger.info("Generating ICD codes...")
    tokenizer, model = model_tuple
    
    logger.info(f"Input text: {text}")
    prompt = generate_prompt(text)
    
    logger.info("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    logger.info(f"Number of tokens: {inputs.input_ids.shape[1]}")
    
    logger.info("Running LLM inference...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("LLM inference complete.")
        logger.info(f"Raw LLM output: {decoded_output}")
    except Exception as e:
        logger.error(f"Error during LLM inference: {str(e)}")
        return []
    
    codes = parse_output(decoded_output)
    return codes

# Test function
def test_generate_icd_codes():
    logger.info("Testing generate_icd_codes function...")
    test_text = "Patient admitted with severe chest pain and shortness of breath. ECG showed ST-segment elevation. Diagnosed with acute myocardial infarction. Treated with thrombolysis and anticoagulation therapy."
    model_tuple = load_llm_model()
    codes = generate_icd_codes(test_text, model_tuple)
    logger.info(f"Test result - Generated codes: {codes}")
    return codes

if __name__ == "__main__":
    test_generate_icd_codes()
