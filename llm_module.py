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
    few_shot_examples = [
        {
            "text": "Patient with chest pain and shortness of breath. ECG shows ST-segment elevation.",
            "codes": "I21.3, R07.4, R06.02",
            "reasoning": "1. Chest pain with ST-segment elevation suggests acute myocardial infarction (I21.3).\n2. Chest pain is also coded separately (R07.4).\n3. Shortness of breath is coded as R06.02."
        },
        {
            "text": "Fever and productive cough for 3 days. Chest X-ray reveals right lower lobe consolidation.",
            "codes": "J15.9, R50.9",
            "reasoning": "1. Productive cough with lung consolidation indicates pneumonia (J15.9).\n2. Fever is coded separately as R50.9."
        },
        {
            "text": "Patient presents with joint pain and swelling in multiple joints. Blood tests show elevated rheumatoid factor.",
            "codes": "M06.09, M25.50",
            "reasoning": "1. Joint pain with elevated rheumatoid factor suggests rheumatoid arthritis (M06.09).\n2. Joint swelling is coded separately as M25.50."
        }
    ]
    
    prompt = "You are an expert medical coder with years of experience in assigning ICD-10 codes. Your task is to analyze medical texts and provide accurate ICD-10 codes with explanations. Follow these guidelines:\n\n"
    prompt += "1. Carefully read and understand the entire medical text.\n"
    prompt += "2. Identify key symptoms, diagnoses, and procedures mentioned.\n"
    prompt += "3. Consider the context and relationships between different medical concepts.\n"
    prompt += "4. Assign the most specific ICD-10 codes possible based on the information provided.\n"
    prompt += "5. If the information is insufficient for a specific code, use a more general code.\n"
    prompt += "6. Provide a detailed explanation for each assigned code, including your reasoning process.\n"
    prompt += "7. List codes in order of relevance to the primary condition or reason for visit.\n"
    prompt += "8. Consider potential comorbidities and complications.\n"
    prompt += "9. Be aware of coding guidelines and conventions, such as combination codes and sequencing rules.\n"
    prompt += "10. If applicable, include external cause codes (V00-Y99) for injuries or adverse effects.\n\n"
    
    prompt += "Here are some examples to guide your thought process:\n\n"
    
    for example in few_shot_examples:
        prompt += f"Text: {example['text']}\n"
        prompt += f"Codes: {example['codes']}\n"
        prompt += f"Reasoning:\n{example['reasoning']}\n\n"
    
    prompt += "Now, analyze the following medical text and provide ICD-10 codes with detailed explanations:\n\n"
    prompt += f"Text: {text}\n"
    prompt += "Codes and Explanations:\n"
    prompt += "For each code, provide the following information:\n"
    prompt += "1. ICD-10 code: [code]\n"
    prompt += "2. Code description: [description]\n"
    prompt += "3. Detailed explanation: [explanation]\n"
    prompt += "4. Relevant coding guidelines: [guidelines]\n"
    prompt += "5. Alternative codes considered: [alternatives]\n"
    
    logger.info("Sophisticated prompt generated.")
    return prompt

def parse_output(output: str) -> list:
    """Parse the output to extract ICD-10 codes."""
    logger.info("Parsing LLM output...")
    codes = re.findall(r"ICD-10 code: ([A-Z0-9.]+)", output)
    
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
    logger.info(f"Generated prompt: {prompt}")
    
    logger.info("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    logger.info(f"Number of tokens: {inputs.input_ids.shape[1]}")
    
    logger.info("Running LLM inference...")
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
