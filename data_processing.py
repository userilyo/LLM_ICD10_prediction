import pandas as pd
import re
from typing import List, Tuple
import os

def load_mimic_data(file_path: str) -> pd.DataFrame:
    """Load MIMIC-III data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MIMIC-III data file not found at {file_path}")
    return pd.read_csv(file_path, low_memory=False)

def preprocess_text(text: str) -> str:
    """Preprocess the input text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_icd_codes(df: pd.DataFrame, code_column: str) -> List[str]:
    """Extract unique ICD codes from the dataset."""
    return df[code_column].dropna().unique().tolist()

def prepare_data(df: pd.DataFrame, text_column: str, code_column: str) -> Tuple[List[str], List[List[str]]]:
    """Prepare data for model input."""
    texts = df[text_column].fillna('').apply(preprocess_text).tolist()
    codes = df[code_column].fillna('').apply(lambda x: x.split(',')).tolist()
    return texts, codes

# Load and preprocess MIMIC-III data
def load_and_prepare_mimic_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[str], List[List[str]]]:
    mimic_data = load_mimic_data(file_path)
    icd_codes = get_icd_codes(mimic_data, 'ICD10_codes')
    texts, codes = prepare_data(mimic_data, 'medical_text', 'ICD10_codes')
    return mimic_data, icd_codes, texts, codes
