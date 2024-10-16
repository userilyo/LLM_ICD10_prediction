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
    few_shot_examples = [
        {"text": "Patient with chest pain and shortness of breath.", "codes": "I20.9, R06.02"},
        {"text": "Fever and cough for 3 days.", "codes": "R50.9, R05"}
    ]
    
    prompt = "Given a medical text, predict the most likely ICD-10 codes. Explain your reasoning for each code.\n\n"
    
    for example in few_shot_examples:
        prompt += f"Text: {example['text']}\n"
        prompt += f"Codes: {example['codes']}\n"
        prompt += "Explanation: [Your explanation for the codes would go here]\n\n"
    
    prompt += f"Text: {text}\n"
    prompt += "Codes and Explanation:"
    
    return prompt

def parse_output(output: str) -> list:
    """Parse the output to extract ICD-10 codes."""
    lines = output.split('\n')
    codes = []
    for line in lines:
        if line.startswith('Codes:'):
            codes = [code.strip() for code in line.split(':')[1].split(',')]
            break
    return codes

def generate_icd_codes(text: str, model_tuple, max_length: int = 256) -> list:
    """Generate ICD-10 codes using a pre-trained LLM with sophisticated prompt engineering."""
    tokenizer, model = model_tuple
    prompt = generate_prompt(text)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
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
    
    return parse_output(decoded_output)
