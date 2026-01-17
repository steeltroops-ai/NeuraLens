"""
Cardiology ECG Analyzer - Using HeartPy and NeuroKit2
Pre-built libraries for fast, accurate ECG analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import io

# Try importing heartpy and neurokit2
try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False


class RiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ECGResult:
    rhythm: str
    heart_rate: int
    confidence: float
    risk_level: str
    findings: List[Dict[str, str]]
    parameters: Dict[str, Any]
    recommendation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ECGAnalyzer:
    """
    ECG Analysis using HeartPy and NeuroKit2
    
    HeartPy: Simple, fast heart rate analysis
    NeuroKit2: Comprehensive ECG processing
    """
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
        
        if not HEARTPY_AVAILABLE and not NEUROKIT_AVAILABLE:
            print("WARNING: Neither HeartPy nor NeuroKit2 installed. Using fallback.")
    
    def analyze(self, ecg_signal: np.ndarray) -> ECGResult:
        """
        Analyze ECG signal and return comprehensive results
        """
        # Try NeuroKit2 first (more comprehensive)
        if NEUROKIT_AVAILABLE:
            return self._analyze_neurokit(ecg_signal)
        elif HEARTPY_AVAILABLE:
            return self._analyze_heartpy(ecg_signal)
        else:
            return self._analyze_fallback(ecg_signal)
    
    def _analyze_neurokit(self, ecg_signal: np.ndarray) -> ECGResult:
        """Analyze using NeuroKit2"""
        try:
            # Process ECG
            signals, info = nk.ecg_process(ecg_signal, sampling_rate=self.sample_rate)
            
            # Extract metrics
            heart_rate = int(np.mean(signals["ECG_Rate"].dropna()))
            r_peaks = info["ECG_R_Peaks"]
            
            # Calculate HRV metrics
            hrv = nk.hrv_time(r_peaks, sampling_rate=self.sample_rate)
            
            # Get intervals
            parameters = {
                "heart_rate_bpm": heart_rate,
                "rmssd_ms": float(hrv["HRV_RMSSD"].values[0]) if "HRV_RMSSD" in hrv else None,
                "sdnn_ms": float(hrv["HRV_SDNN"].values[0]) if "HRV_SDNN" in hrv else None,
                "mean_rr_ms": float(hrv["HRV_MeanNN"].values[0]) if "HRV_MeanNN" in hrv else None,
                "r_peaks_count": len(r_peaks),
                "quality_score": float(np.mean(signals["ECG_Quality"])) if "ECG_Quality" in signals else 0.9
            }
            
            # Classify rhythm
            rhythm, confidence = self._classify_rhythm(heart_rate, parameters)
            
            # Generate findings
            findings = self._generate_findings(heart_rate, rhythm, parameters)
            
            # Assess risk
            risk_level = self._assess_risk(rhythm, heart_rate, parameters)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(risk_level, rhythm)
            
            return ECGResult(
                rhythm=rhythm,
                heart_rate=heart_rate,
                confidence=confidence,
                risk_level=risk_level,
                findings=findings,
                parameters=parameters,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"NeuroKit2 error: {e}, falling back")
            return self._analyze_fallback(ecg_signal)
    
    def _analyze_heartpy(self, ecg_signal: np.ndarray) -> ECGResult:
        """Analyze using HeartPy"""
        try:
            # Process with HeartPy
            working_data, measures = hp.process(
                ecg_signal, 
                sample_rate=self.sample_rate,
                report_time=False
            )
            
            heart_rate = int(measures['bpm'])
            
            parameters = {
                "heart_rate_bpm": heart_rate,
                "rmssd_ms": measures.get('rmssd'),
                "sdnn_ms": measures.get('sdnn'),
                "ibi_ms": measures.get('ibi'),
                "pnn50": measures.get('pnn50'),
                "breathingrate": measures.get('breathingrate')
            }
            
            rhythm, confidence = self._classify_rhythm(heart_rate, parameters)
            findings = self._generate_findings(heart_rate, rhythm, parameters)
            risk_level = self._assess_risk(rhythm, heart_rate, parameters)
            recommendation = self._generate_recommendation(risk_level, rhythm)
            
            return ECGResult(
                rhythm=rhythm,
                heart_rate=heart_rate,
                confidence=confidence,
                risk_level=risk_level,
                findings=findings,
                parameters=parameters,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"HeartPy error: {e}, falling back")
            return self._analyze_fallback(ecg_signal)
    
    def _analyze_fallback(self, ecg_signal: np.ndarray) -> ECGResult:
        """Fallback analysis using basic signal processing"""
        from scipy import signal as scipy_signal
        
        # Simple R-peak detection
        peaks, _ = scipy_signal.find_peaks(ecg_signal, distance=self.sample_rate * 0.5)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.sample_rate
            heart_rate = int(60 / np.mean(rr_intervals))
        else:
            heart_rate = 72  # Default
        
        parameters = {
            "heart_rate_bpm": heart_rate,
            "r_peaks_count": len(peaks),
            "method": "fallback"
        }
        
        rhythm, confidence = self._classify_rhythm(heart_rate, parameters)
        findings = self._generate_findings(heart_rate, rhythm, parameters)
        risk_level = self._assess_risk(rhythm, heart_rate, parameters)
        recommendation = self._generate_recommendation(risk_level, rhythm)
        
        return ECGResult(
            rhythm=rhythm,
            heart_rate=heart_rate,
            confidence=confidence * 0.8,  # Lower confidence for fallback
            risk_level=risk_level,
            findings=findings,
            parameters=parameters,
            recommendation=recommendation
        )
    
    def _classify_rhythm(self, heart_rate: int, params: Dict) -> tuple:
        """Classify cardiac rhythm based on heart rate and HRV"""
        
        # Check HRV for irregularity (potential AFib)
        rmssd = params.get('rmssd_ms')
        if rmssd and rmssd > 100:  # High HRV variability
            return "Irregular Rhythm (Possible AFib)", 0.75
        
        # Heart rate based classification
        if heart_rate < 50:
            return "Sinus Bradycardia", 0.90
        elif heart_rate < 60:
            return "Low Normal Sinus Rhythm", 0.92
        elif heart_rate <= 100:
            return "Normal Sinus Rhythm", 0.95
        elif heart_rate <= 120:
            return "Sinus Tachycardia", 0.90
        else:
            return "Significant Tachycardia", 0.85
    
    def _generate_findings(self, heart_rate: int, rhythm: str, params: Dict) -> List[Dict]:
        """Generate clinical findings"""
        findings = []
        
        # Rhythm finding
        if "Normal" in rhythm:
            findings.append({
                "type": rhythm,
                "severity": "normal",
                "description": "Regular rhythm with normal rate between 60-100 bpm"
            })
        elif "Bradycardia" in rhythm:
            findings.append({
                "type": rhythm,
                "severity": "mild",
                "description": f"Heart rate {heart_rate} bpm is below normal range"
            })
        elif "Tachycardia" in rhythm:
            findings.append({
                "type": rhythm,
                "severity": "moderate" if heart_rate > 120 else "mild",
                "description": f"Heart rate {heart_rate} bpm is above normal range"
            })
        elif "AFib" in rhythm or "Irregular" in rhythm:
            findings.append({
                "type": rhythm,
                "severity": "moderate",
                "description": "Irregular R-R intervals detected suggesting rhythm abnormality"
            })
        
        # HRV findings
        rmssd = params.get('rmssd_ms')
        if rmssd:
            if rmssd < 20:
                findings.append({
                    "type": "Low Heart Rate Variability",
                    "severity": "mild",
                    "description": f"RMSSD {rmssd:.1f}ms suggests reduced autonomic function"
                })
            elif rmssd > 50:
                findings.append({
                    "type": "Good Heart Rate Variability",
                    "severity": "normal",
                    "description": f"RMSSD {rmssd:.1f}ms indicates healthy autonomic function"
                })
        
        return findings
    
    def _assess_risk(self, rhythm: str, heart_rate: int, params: Dict) -> str:
        """Assess overall cardiac risk"""
        risk_score = 0
        
        # Heart rate risk
        if heart_rate < 40 or heart_rate > 150:
            risk_score += 4
        elif heart_rate < 50 or heart_rate > 120:
            risk_score += 2
        elif heart_rate < 60 or heart_rate > 100:
            risk_score += 1
        
        # Rhythm risk
        if "AFib" in rhythm or "Irregular" in rhythm:
            risk_score += 3
        
        # Map to level
        if risk_score >= 5:
            return RiskLevel.CRITICAL.value
        elif risk_score >= 3:
            return RiskLevel.HIGH.value
        elif risk_score >= 2:
            return RiskLevel.MODERATE.value
        elif risk_score >= 1:
            return RiskLevel.LOW.value
        else:
            return RiskLevel.NORMAL.value
    
    def _generate_recommendation(self, risk_level: str, rhythm: str) -> str:
        """Generate clinical recommendation"""
        recommendations = {
            RiskLevel.NORMAL.value: "No immediate action required. Continue routine health monitoring.",
            RiskLevel.LOW.value: "Minor findings noted. Consider follow-up ECG if symptoms develop.",
            RiskLevel.MODERATE.value: "Consult with a cardiologist for further evaluation recommended.",
            RiskLevel.HIGH.value: "Urgent cardiology referral advised. Additional testing recommended.",
            RiskLevel.CRITICAL.value: "IMMEDIATE MEDICAL ATTENTION REQUIRED. Seek emergency care."
        }
        return recommendations.get(risk_level, recommendations[RiskLevel.NORMAL.value])


def parse_ecg_file(content: bytes, filename: str, sample_rate: int = 500) -> np.ndarray:
    """Parse ECG file into numpy array"""
    
    if filename.endswith('.csv'):
        data = np.loadtxt(io.BytesIO(content), delimiter=',', skiprows=1)
        return data[:, 0] if data.ndim > 1 else data
    
    elif filename.endswith('.txt'):
        data = np.loadtxt(io.BytesIO(content))
        return data
    
    else:
        # Generate synthetic for demo
        return generate_demo_ecg(sample_rate=sample_rate)


def generate_demo_ecg(sample_rate: int = 500, duration: float = 10, heart_rate: int = 72) -> np.ndarray:
    """Generate synthetic ECG for demo"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create realistic ECG pattern
    ecg = np.zeros_like(t)
    beat_interval = 60 / heart_rate
    
    for beat_time in np.arange(0, duration, beat_interval):
        beat_idx = int(beat_time * sample_rate)
        if beat_idx < len(ecg) - 60:
            # P wave
            ecg[beat_idx:beat_idx+10] = 0.1 * np.sin(np.linspace(0, np.pi, 10))
            # QRS complex
            ecg[beat_idx+20] = -0.1
            ecg[beat_idx+22] = 1.0
            ecg[beat_idx+24] = -0.2
            # T wave
            ecg[beat_idx+40:beat_idx+55] = 0.2 * np.sin(np.linspace(0, np.pi, 15))
    
    # Add noise
    ecg += 0.02 * np.random.randn(len(ecg))
    
    return ecg
