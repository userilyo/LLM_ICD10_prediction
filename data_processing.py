import pandas as pd
import re
from typing import List

def load_mimic_data(file_path: str) -> pd.DataFrame:
    """Load MIMIC-III sample data."""
    return pd.read_csv(file_path)

def preprocess_text(text: str) -> str:
    """Preprocess the input text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_icd_codes(df: pd.DataFrame) -> List[str]:
    """Extract unique ICD-10 codes from the dataset."""
    return df['ICD10_codes'].explode().unique().tolist()

# Load and preprocess MIMIC-III sample data
mimic_data = load_mimic_data('data/mimic_iii_sample.csv')
icd_codes = get_icd_codes(mimic_data)
