"""
Voice Pipeline - Text Processor
Medical text preprocessing for better TTS pronunciation
"""

import re
from typing import Dict


# Medical abbreviation to pronunciation mappings
MEDICAL_PRONUNCIATIONS: Dict[str, str] = {
    # Common medical acronyms
    "HbA1c": "H-B-A-1-C",
    "mmHg": "millimeters of mercury",
    "mg/dL": "milligrams per deciliter",
    "mL": "milliliters",
    "mcg": "micrograms",
    "ECG": "E-C-G",
    "EKG": "E-K-G",
    "MRI": "M-R-I",
    "CT": "C-T scan",
    "X-ray": "X ray",
    
    # Cardiovascular
    "BP": "blood pressure",
    "HR": "heart rate",
    "HRV": "heart rate variability",
    "RMSSD": "R-M-S-S-D",
    "SDNN": "S-D-N-N",
    "pNN50": "P-N-N fifty",
    "AFib": "atrial fibrillation",
    "PVC": "premature ventricular contraction",
    "PAC": "premature atrial contraction",
    "QTc": "Q-T-C interval",
    "bpm": "beats per minute",
    
    # Eye/Retinal
    "DR": "diabetic retinopathy",
    "AMD": "age-related macular degeneration",
    "NPDR": "non-proliferative diabetic retinopathy",
    "PDR": "proliferative diabetic retinopathy",
    "IOP": "intraocular pressure",
    "C/D": "cup to disc",
    "RNFL": "retinal nerve fiber layer",
    "A/V": "A to V ratio",
    
    # Neurological
    "NRI": "Neurological Risk Index",
    "MCI": "mild cognitive impairment",
    "PD": "Parkinson's disease",
    "AD": "Alzheimer's disease",
    "HNR": "harmonics to noise ratio",
    "dB": "decibels",
    
    # Speech biomarkers
    "jitter": "jitter",
    "shimmer": "shimmer",
    "syll/s": "syllables per second",
    
    # Other
    "AI": "A-I",
    "ML": "machine learning",
    "vs": "versus",
    "w/": "with",
    "w/o": "without",
}

# Medical term explanations
MEDICAL_EXPLANATIONS: Dict[str, str] = {
    "jitter": (
        "Jitter measures the variation in your voice pitch from one vocal cord "
        "vibration to the next. Higher values may indicate vocal instability."
    ),
    "shimmer": (
        "Shimmer measures the variation in your voice loudness between vocal "
        "cord vibrations. It indicates voice stability."
    ),
    "hnr": (
        "Harmonics-to-noise ratio shows how clear your voice is. Higher values "
        "mean a clearer, more resonant voice."
    ),
    "nri": (
        "The Neurological Risk Index combines results from multiple tests into "
        "a single score indicating overall neurological health."
    ),
    "cup_disc_ratio": (
        "The cup-to-disc ratio measures the size of the optic cup compared to "
        "the optic disc in your eye. Higher ratios may indicate glaucoma risk."
    ),
    "rmssd": (
        "R-M-S-S-D is a measure of heart rate variability that shows how well "
        "your autonomic nervous system regulates your heart rhythm."
    ),
    "diabetic_retinopathy": (
        "Diabetic retinopathy is damage to the blood vessels in the retina "
        "caused by high blood sugar levels."
    ),
    "glaucoma": (
        "Glaucoma is a group of eye conditions that damage the optic nerve, "
        "often caused by abnormally high pressure in your eye."
    ),
    "arrhythmia": (
        "An arrhythmia is an irregular heartbeat. Your heart may beat too fast, "
        "too slow, or with an irregular rhythm."
    ),
    "atrial_fibrillation": (
        "Atrial fibrillation, or A-fib, is an irregular and often rapid heart "
        "rate that can increase your risk of stroke."
    ),
    "tremor": (
        "Tremor is an involuntary, rhythmic muscle movement. Voice tremor can "
        "indicate neurological conditions affecting speech."
    ),
    "bradykinesia": (
        "Bradykinesia means slowness of movement. It's a common symptom of "
        "Parkinson's disease affecting motor function."
    ),
}


def preprocess_for_speech(text: str) -> str:
    """
    Preprocess medical text for better TTS pronunciation
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Text optimized for speech synthesis
    """
    if not text:
        return ""
    
    # Strip markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # Italic *text*
    text = re.sub(r'#{1,6}\s*', '', text)          # Headers # ## ###
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text) # Links [text](url)
    text = re.sub(r'`(.+?)`', r'\1', text)         # Code `text`
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # Bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE) # Numbered lists
    
    # Replace medical abbreviations with pronunciations
    for abbrev, pronunciation in MEDICAL_PRONUNCIATIONS.items():
        # Use word boundary matching for accurate replacement
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        text = re.sub(pattern, pronunciation, text, flags=re.IGNORECASE)
    
    # Format numbers for clearer speech
    text = format_numbers_for_speech(text)
    
    # Add pauses after periods and colons for natural rhythm
    text = text.replace(". ", "... ")
    text = text.replace(": ", ": ... ")
    text = text.replace("! ", "!... ")
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def format_numbers_for_speech(text: str) -> str:
    """Format numbers for clearer speech synthesis"""
    
    # Format percentages: 85% -> 85 percent
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
    
    # Format decimal numbers: 0.85 -> 0 point 8 5
    def format_decimal(match):
        whole = match.group(1) or "0"
        decimal = match.group(2)
        # Split decimal digits for clarity
        decimal_spoken = " ".join(decimal)
        return f"{whole} point {decimal_spoken}"
    
    text = re.sub(r'(\d+)?\.(\d+)', format_decimal, text)
    
    # Format ranges: 0.01-0.04 -> 0.01 to 0.04
    text = re.sub(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', r'\1 to \2', text)
    
    # Format fractions: 1/2 -> one half
    text = text.replace("1/2", "one half")
    text = text.replace("1/4", "one quarter")
    text = text.replace("3/4", "three quarters")
    
    return text


def get_medical_explanation(term: str, context: str = None) -> str:
    """
    Get explanation for a medical term
    
    Args:
        term: Medical term to explain
        context: Optional context (pipeline name)
        
    Returns:
        Human-readable explanation
    """
    # Normalize term
    normalized = term.lower().replace(" ", "_").replace("-", "_")
    
    # Check explanations dictionary
    if normalized in MEDICAL_EXPLANATIONS:
        return MEDICAL_EXPLANATIONS[normalized]
    
    # Default explanation
    return f"{term} is a biomarker used in our medical analysis to assess your health."


def chunk_text(text: str, max_chars: int = 5000) -> list:
    """
    Split long text into chunks for TTS processing
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = text.replace(". ", ".|").split("|")
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
