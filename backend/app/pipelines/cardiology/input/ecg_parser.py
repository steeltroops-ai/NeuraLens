"""
Cardiology Pipeline - ECG File Parser
Parses ECG signal data from various file formats (CSV, JSON, TXT).
"""

import numpy as np
import json
import io
import logging
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ECGParser:
    """Parse ECG files into numpy arrays."""
    
    SUPPORTED_FORMATS = [".csv", ".json", ".txt"]
    
    def parse(
        self,
        content: bytes,
        filename: str,
        sample_rate: int = 500
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Parse ECG file into signal array.
        
        Args:
            content: File content as bytes
            filename: Original filename (for format detection)
            sample_rate: Default sample rate if not specified
        
        Returns:
            Tuple of (signal_array, sample_rate, metadata)
        """
        ext = "." + filename.lower().split(".")[-1]
        
        if ext == ".csv":
            return self._parse_csv(content, sample_rate)
        elif ext == ".json":
            return self._parse_json(content, sample_rate)
        elif ext == ".txt":
            return self._parse_txt(content, sample_rate)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def _parse_csv(
        self,
        content: bytes,
        default_sample_rate: int
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Parse CSV ECG file."""
        text = content.decode("utf-8")
        lines = text.strip().split("\n")
        
        # Detect header
        first_line = lines[0].strip()
        has_header = not self._is_numeric_line(first_line)
        
        start_idx = 1 if has_header else 0
        
        # Parse values
        values = []
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Handle various delimiters
            if "," in line:
                parts = line.split(",")
            elif "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()
            
            # Extract the voltage value (last column typically, or first if single column)
            try:
                if len(parts) == 1:
                    val = float(parts[0])
                else:
                    # Try last column first (time, voltage format)
                    val = float(parts[-1])
                values.append(val)
            except ValueError:
                continue
        
        signal = np.array(values, dtype=np.float64)
        
        metadata = {
            "format": "csv",
            "has_header": has_header,
            "num_samples": len(signal),
        }
        
        return signal, default_sample_rate, metadata
    
    def _parse_json(
        self,
        content: bytes,
        default_sample_rate: int
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Parse JSON ECG file."""
        data = json.loads(content.decode("utf-8"))
        
        # Try various JSON structures
        signal = None
        sample_rate = default_sample_rate
        
        # Structure 1: {"data": [...], "sample_rate": ...}
        if "data" in data:
            signal = np.array(data["data"], dtype=np.float64)
            sample_rate = data.get("sample_rate", data.get("sample_rate_hz", default_sample_rate))
        
        # Structure 2: {"leads": {"I": [...], ...}, "sample_rate_hz": ...}
        elif "leads" in data:
            leads = data["leads"]
            # Use lead I if available, otherwise first lead
            if "I" in leads:
                signal = np.array(leads["I"], dtype=np.float64)
            elif "lead_I" in leads:
                signal = np.array(leads["lead_I"], dtype=np.float64)
            else:
                first_lead = list(leads.values())[0]
                signal = np.array(first_lead, dtype=np.float64)
            sample_rate = data.get("sample_rate_hz", data.get("sample_rate", default_sample_rate))
        
        # Structure 3: {"ecg_data": [...], ...}
        elif "ecg_data" in data:
            signal = np.array(data["ecg_data"], dtype=np.float64)
            sample_rate = data.get("sample_rate", default_sample_rate)
        
        # Structure 4: Direct array
        elif isinstance(data, list):
            signal = np.array(data, dtype=np.float64)
        
        if signal is None:
            raise ValueError("Could not extract ECG signal from JSON")
        
        metadata = {
            "format": "json",
            "num_samples": len(signal),
            "duration_sec": data.get("duration_seconds"),
            "unit": data.get("unit", "mV"),
        }
        
        return signal, int(sample_rate), metadata
    
    def _parse_txt(
        self,
        content: bytes,
        default_sample_rate: int
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Parse TXT ECG file (space/tab separated)."""
        text = content.decode("utf-8")
        lines = text.strip().split("\n")
        
        values = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            try:
                # Single column: just voltage
                if len(parts) == 1:
                    val = float(parts[0])
                else:
                    # Multiple columns: assume last is voltage
                    val = float(parts[-1])
                values.append(val)
            except ValueError:
                continue
        
        signal = np.array(values, dtype=np.float64)
        
        metadata = {
            "format": "txt",
            "num_samples": len(signal),
        }
        
        return signal, default_sample_rate, metadata
    
    def _is_numeric_line(self, line: str) -> bool:
        """Check if line contains only numeric values."""
        parts = line.replace(",", " ").replace("\t", " ").split()
        try:
            for part in parts:
                float(part)
            return True
        except ValueError:
            return False


# Convenience functions
def parse_ecg_file(
    content: bytes,
    filename: str,
    sample_rate: int = 500
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Parse ECG file into numpy array."""
    parser = ECGParser()
    return parser.parse(content, filename, sample_rate)


def parse_ecg_csv(content: bytes, sample_rate: int = 500) -> np.ndarray:
    """Parse ECG CSV file."""
    parser = ECGParser()
    signal, _, _ = parser._parse_csv(content, sample_rate)
    return signal


def parse_ecg_json(content: bytes, sample_rate: int = 500) -> Tuple[np.ndarray, int]:
    """Parse ECG JSON file."""
    parser = ECGParser()
    signal, sr, _ = parser._parse_json(content, sample_rate)
    return signal, sr
