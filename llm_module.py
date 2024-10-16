from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load pre-trained model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_icd_codes(text: str, max_length: int = 128) -> list:
    """Generate ICD-10 codes using a pre-trained LLM."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=5,
            num_beams=5,
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
