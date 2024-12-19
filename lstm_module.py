import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class LSTMVerifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMVerifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

def load_lstm_model():
    """Load BERT tokenizer and model for feature extraction, and initialize LSTM verifier"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models with error handling
        tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            local_files_only=False,
            resume_download=True
        )
        bert_model = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            local_files_only=False,
            resume_download=True
        ).to(device)

        input_size = 768  # BERT hidden size
        hidden_size = 64  # Reduced from 128
        num_layers = 1   # Reduced from 2
        num_classes = 1  # Binary classification (valid/invalid)
        
        lstm_verifier = LSTMVerifier(input_size, hidden_size, num_layers, num_classes).to(device)
        
        # Set models to evaluation mode
        bert_model.eval()
        lstm_verifier.eval()
        
        return tokenizer, bert_model, lstm_verifier
    except Exception as e:
        logger.error(f"Error loading LSTM model: {str(e)}")
        raise RuntimeError(f"Failed to load LSTM model: {str(e)}")

def verify_icd_codes(text: str, icd_codes: list, model_tuple) -> list:
    """Verify ICD-10 codes using LSTM-based verification and return confidence scores."""
    try:
        tokenizer, bert_model, lstm_verifier = model_tuple
        device = next(bert_model.parameters()).device
        
        # Tokenize and encode the input text
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Verify each ICD code with confidence scores
        verified_codes_with_conf = []
        for code in icd_codes:
            try:
                # Combine text embeddings with code embedding
                code_embedding = tokenizer.encode(code, return_tensors="pt").to(device)
                code_outputs = bert_model(code_embedding)
                code_embedding = code_outputs.last_hidden_state.mean(dim=1)
                combined_embedding = torch.cat([embeddings.mean(dim=1), code_embedding], dim=1)
                
                # Get prediction confidence
                with torch.no_grad():
                    confidence = lstm_verifier(combined_embedding).item()
                
                if confidence > 0.5:  # Keep threshold but return confidence
                    verified_codes_with_conf.append((code, confidence))
            except Exception as e:
                logger.warning(f"Error processing code {code}: {str(e)}")
                continue
        
        # Sort by confidence
        verified_codes_with_conf.sort(key=lambda x: x[1], reverse=True)
        return verified_codes_with_conf
        
    except Exception as e:
        logger.error(f"Error in LSTM verification: {str(e)}")
        return []
