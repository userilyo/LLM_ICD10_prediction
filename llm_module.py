from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_llm_model():
    """Load a smaller pre-trained model for ICD code generation."""
    model_name = "facebook/bart-base"  # Using a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_prompt(text: str) -> str:
    """Generate a sophisticated prompt for ICD-10 code prediction."""
    print("Generating sophisticated prompt...")
    few_shot_examples = [
        {"text": "Patient with chest pain and shortness of breath.", "codes": "I20.9, R06.02"},
        {"text": "Fever and cough for 3 days.", "codes": "R50.9, R05"},
        {"text": "Patient presents with joint pain and swelling in multiple joints.", "codes": "M13.0, M25.50"}
    ]
    
    prompt = "You are an expert medical coder tasked with assigning ICD-10 codes to medical texts. Follow these guidelines:\n"
    prompt += "1. Carefully analyze the given medical text for key symptoms, diagnoses, and procedures.\n"
    prompt += "2. Assign the most specific ICD-10 codes possible based on the information provided.\n"
    prompt += "3. If the information is insufficient for a specific code, use a more general code.\n"
    prompt += "4. Provide a brief explanation for each assigned code.\n"
    prompt += "5. List codes in order of relevance to the primary condition or reason for visit.\n\n"
    
    prompt += "Here are some examples:\n\n"
    
    for example in few_shot_examples:
        prompt += f"Text: {example['text']}\n"
        prompt += f"Codes: {example['codes']}\n"
        prompt += "Explanation: [Your explanation for the codes would go here]\n\n"
    
    prompt += f"Now, analyze the following medical text and provide ICD-10 codes with explanations:\n\n"
    prompt += f"Text: {text}\n"
    prompt += "Codes and Explanations:"
    
    print("Sophisticated prompt generated.")
    return prompt

def parse_output(output: str) -> list:
    """Parse the output to extract ICD-10 codes."""
    print("Parsing LLM output...")
    lines = output.split('\n')
    codes = []
    for line in lines:
        if line.startswith('Codes:'):
            codes = [code.strip() for code in line.split(':')[1].split(',')]
            break
    print(f"Parsed codes: {codes}")
    return codes

def generate_icd_codes(text: str, model_tuple, max_length: int = 256) -> list:
    """Generate ICD-10 codes using a pre-trained LLM with sophisticated prompt engineering."""
    print("Generating ICD codes...")
    tokenizer, model = model_tuple
    prompt = generate_prompt(text)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    print("Running LLM inference...")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("LLM inference complete.")
    
    return parse_output(decoded_output)
