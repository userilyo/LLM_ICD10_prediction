import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class LSTMVerifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMVerifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        out, _ = self.lstm(x, (h0, c0))
        context_vector, attention_weights = self.attention(out)
        out = self.fc(context_vector)
        return torch.sigmoid(out), attention_weights

def load_lstm_model():
    """Load BERT tokenizer and model for feature extraction, and initialize LSTM verifier"""
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    input_size = 768  # BERT hidden size
    hidden_size = 128  # Increased from 64
    num_layers = 2  # Increased from 1
    num_classes = 1  # Binary classification (valid/invalid)
    lstm_verifier = LSTMVerifier(input_size, hidden_size, num_layers, num_classes)

    return tokenizer, bert_model, lstm_verifier

def verify_icd_codes(text: str, icd_codes: list, model_tuple) -> list:
    """Verify ICD-10 codes using LSTM-based verification with attention."""
    tokenizer, bert_model, lstm_verifier = model_tuple
    
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state
    
    # Verify each ICD code
    verified_codes = []
    attention_weights_list = []
    for code in icd_codes:
        # Combine text embeddings with code embedding (simple concatenation)
        code_embedding = tokenizer.encode(code, return_tensors="pt")
        code_outputs = bert_model(code_embedding)
        code_embedding = code_outputs.last_hidden_state.mean(dim=1)
        combined_embedding = torch.cat([embeddings, code_embedding.expand(embeddings.size(0), -1)], dim=1)
        
        # Verify using LSTM with attention
        with torch.no_grad():
            prediction, attention_weights = lstm_verifier(combined_embedding)
        
        if prediction.item() > 0.5:  # Threshold for binary classification
            verified_codes.append(code)
            attention_weights_list.append(attention_weights)
    
    return verified_codes, attention_weights_list
