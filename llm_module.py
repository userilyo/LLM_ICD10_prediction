from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_llm_model():
    """Load a smaller pre-trained model for ICD code generation."""
    model_name = "facebook/bart-base"  # Using a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_icd_codes(text: str, model_tuple, max_length: int = 128) -> list:
    """Generate ICD-10 codes using a pre-trained LLM."""
    tokenizer, model = model_tuple
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=3,  # Reduced from 5 to 3
            num_beams=3,  # Reduced from 5 to 3
            do_sample=True,
            temperature=0.7,
        )
    
    decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Extract potential ICD-10 codes from the generated text
    icd_codes = []
    for output in decoded_outputs:
        codes = [code.strip() for code in output.split() if code.strip().startswith('I') or code.strip().startswith('J')]
        icd_codes.extend(codes)
    
    return list(set(icd_codes))  # Remove duplicates
