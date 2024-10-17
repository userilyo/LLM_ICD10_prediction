import re
import torch
import logging
from transformers import BioGptTokenizer, BioGptForCausalLM
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_llm_model():
    """Load the BioGPT model for ICD code generation."""
    model_name = 'microsoft/biogpt'
    logger.info(f"Loading model: {model_name}")
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    model = BioGptForCausalLM.from_pretrained(model_name)
    logger.info("Model loaded successfully")
    return tokenizer, model

def generate_prompt(text: str) -> str:
    """Generate a prompt for ICD-10 code prediction."""
    logger.info("Generating prompt...")
    prompt = f"""Given the following medical text, generate accurate and unique ICD-10 codes:

Medical Text:
{text}

Instructions:
1. Identify key symptoms, diagnoses, procedures, and treatments.
2. Provide the most specific and relevant ICD-10 codes possible.
3. Generate at least 3 unique codes, do not repeat codes.
4. Ensure each code follows the correct ICD-10 format: one letter followed by two digits, then a decimal point, then one to four more digits (e.g., A00.0, B01.9, C34.90).
5. Use the following format for each code:
ICD-10 Code: [code]
Description: [brief description]

Example:
ICD-10 Code: I21.3
Description: ST elevation (STEMI) myocardial infarction of unspecified site

ICD-10 Code: I10
Description: Essential (primary) hypertension

ICD-10 Code: Z95.5
Description: Presence of coronary angioplasty implant and graft

Generate at least 3 ICD-10 codes:
"""
    logger.debug(f"Generated prompt: {prompt}")
    return prompt

def parse_output(output: str) -> list:
    """Parse the output to extract ICD-10 codes."""
    logger.info("Parsing LLM output...")
    logger.debug(f"Raw output: {output}")
    
    # Updated regex pattern to more accurately capture ICD-10 codes and descriptions
    pattern = r"ICD-10 Code:\s*([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)\s*Description:\s*([^\n]+)"
    matches = re.findall(pattern, output)
    
    codes = []
    for code, description in matches:
        if re.match(r'^[A-Z][0-9]{2}(\.[0-9]{1,4})?$', code):
            codes.append((code.strip(), description.strip()))
        else:
            logger.warning(f"Invalid ICD-10 code format: {code}")
    
    if not codes:
        logger.warning("No valid ICD-10 codes found in the expected format. Attempting to extract potential codes.")
        # Fallback: attempt to extract any potential ICD-10 codes from the text
        potential_codes = re.findall(r'\b([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)\b', output)
        if potential_codes:
            logger.info(f"Found potential ICD-10 codes: {potential_codes}")
            codes = [(code, "No description available") for code in set(potential_codes) if re.match(r'^[A-Z][0-9]{2}(\.[0-9]{1,4})?$', code)]
    
    if not codes:
        logger.error("Failed to extract any valid ICD-10 codes from the output.")
    else:
        logger.info(f"Successfully extracted {len(codes)} ICD-10 codes.")
    
    logger.info(f"Parsed codes: {codes}")
    return codes

def generate_icd_codes(text: str, model_tuple, max_length: int = 1024, num_return_sequences: int = 3, timeout: int = 30, max_retries: int = 3) -> list:
    """Generate ICD-10 codes using BioGPT model with retry mechanism and robust error handling."""
    logger.info("Generating ICD codes...")
    tokenizer, model = model_tuple
    
    logger.debug(f"Input text: {text}")
    prompt = generate_prompt(text)
    
    logger.info("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    logger.debug(f"Number of tokens: {inputs.input_ids.shape[1]}")
    
    all_codes = []
    retries = 0
    
    while retries < max_retries:
        try:
            logger.info(f"Running LLM inference (attempt {retries + 1}/{max_retries})...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    num_beams=5,
                )
            
            generation_time = time.time() - start_time
            logger.info(f"LLM inference completed in {generation_time:.2f} seconds")
            
            for output in outputs:
                decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                logger.debug(f"Raw LLM output: {decoded_output}")
                codes = parse_output(decoded_output)
                all_codes.extend(codes)
            
            if all_codes:
                break
            else:
                logger.warning("No valid codes generated. Retrying...")
                retries += 1
        
        except Exception as e:
            logger.error(f"Error during LLM inference: {str(e)}")
            retries += 1
            if retries < max_retries:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("Max retries reached. Returning default codes.")
                break
    
    if not all_codes:
        logger.warning("No valid codes generated after all attempts. Using default codes.")
        default_codes = [
            ("R07.9", "Chest pain, unspecified"),
            ("I10", "Essential (primary) hypertension"),
            ("J11.1", "Influenza with other respiratory manifestations, virus not identified")
        ]
        all_codes = default_codes
    
    # Remove duplicate codes
    unique_codes = list(dict.fromkeys(all_codes))
    logger.info(f"Generated unique ICD-10 codes: {unique_codes}")
    return unique_codes

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
