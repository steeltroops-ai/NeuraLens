"""
Cardiology Pipeline - Demo ECG Generation
Generates synthetic ECG signals for testing
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try importing neurokit2 for realistic ECG
try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    logger.warning("neurokit2 not installed. Using basic ECG generation.")


def generate_demo_ecg(
    duration: float = 10.0,
    sample_rate: int = 500,
    heart_rate: int = 72,
    add_arrhythmia: bool = False,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic ECG for demo purposes
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling frequency in Hz
        heart_rate: Target heart rate in bpm
        add_arrhythmia: Add simulated arrhythmia pattern
        random_state: Random seed for reproducibility
    
    Returns:
        Synthetic ECG signal array
    """
    np.random.seed(random_state)
    
    if NEUROKIT_AVAILABLE:
        return _generate_neurokit_ecg(
            duration, sample_rate, heart_rate, add_arrhythmia, random_state
        )
    else:
        return _generate_basic_ecg(
            duration, sample_rate, heart_rate, add_arrhythmia
        )


def _generate_neurokit_ecg(
    duration: float,
    sample_rate: int,
    heart_rate: int,
    add_arrhythmia: bool,
    random_state: int
) -> np.ndarray:
    """Generate realistic ECG using NeuroKit2"""
    
    try:
        # Generate base ECG
        ecg = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sample_rate,
            heart_rate=heart_rate,
            method="ecgsyn",
            random_state=random_state
        )
        
        # Add realistic noise
        noise = np.random.normal(0, 0.02, len(ecg))
        ecg_noisy = ecg + noise
        
        # Add arrhythmia if requested
        if add_arrhythmia:
            ecg_noisy = _add_ectopic_beats(ecg_noisy, sample_rate, heart_rate)
        
        return ecg_noisy
    except Exception as e:
        logger.warning(f"NeuroKit2 ecg_simulate failed: {e}. Using basic generation.")
        return _generate_basic_ecg(duration, sample_rate, heart_rate, add_arrhythmia)


def _generate_basic_ecg(
    duration: float,
    sample_rate: int,
    heart_rate: int,
    add_arrhythmia: bool
) -> np.ndarray:
    """Generate basic ECG without NeuroKit2"""
    
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    ecg = np.zeros(num_samples)
    
    beat_interval_sec = 60 / heart_rate
    beat_samples = int(beat_interval_sec * sample_rate)
    
    for beat_start in range(0, num_samples - 100, beat_samples):
        # P wave (small positive)
        p_start = beat_start
        p_end = beat_start + int(sample_rate * 0.08)
        if p_end < num_samples:
            p_duration = p_end - p_start
            ecg[p_start:p_end] = 0.15 * np.sin(np.linspace(0, np.pi, p_duration))
        
        # PR segment (flat)
        
        # QRS complex
        q_pos = beat_start + int(sample_rate * 0.12)
        if q_pos < num_samples - 20:
            ecg[q_pos] = -0.1  # Q wave
            ecg[q_pos + int(sample_rate * 0.01)] = 1.0  # R wave (peak)
            ecg[q_pos + int(sample_rate * 0.02)] = -0.2  # S wave
        
        # T wave (positive)
        t_start = beat_start + int(sample_rate * 0.25)
        t_end = beat_start + int(sample_rate * 0.40)
        if t_end < num_samples:
            t_duration = t_end - t_start
            ecg[t_start:t_end] = 0.3 * np.sin(np.linspace(0, np.pi, t_duration))
    
    # Add noise
    ecg += 0.02 * np.random.randn(len(ecg))
    
    # Add arrhythmia
    if add_arrhythmia:
        ecg = _add_ectopic_beats(ecg, sample_rate, heart_rate)
    
    return ecg


def _add_ectopic_beats(
    ecg: np.ndarray,
    sample_rate: int,
    heart_rate: int
) -> np.ndarray:
    """Add simulated PVCs (premature ventricular contractions)"""
    
    beat_interval = int(sample_rate * 60 / heart_rate)
    num_ectopics = min(3, len(ecg) // (beat_interval * 4))
    
    for i in range(num_ectopics):
        # Random position (not at start or end)
        pos = np.random.randint(beat_interval * 3, len(ecg) - beat_interval * 2)
        
        # Create wide QRS (PVC pattern)
        qrs_width = int(sample_rate * 0.12)  # Wide QRS ~120ms
        
        if pos + qrs_width < len(ecg):
            # Clear existing beat
            ecg[pos:pos + qrs_width] = 0
            
            # Add wide, tall QRS
            center = pos + qrs_width // 2
            ecg[center - 5:center] = -0.3  # Deep Q
            ecg[center:center + 3] = 1.5   # Tall R
            ecg[center + 3:center + 10] = -0.5  # Deep S
    
    return ecg


def generate_afib_ecg(
    duration: float = 10.0,
    sample_rate: int = 500,
    heart_rate: int = 100
) -> np.ndarray:
    """
    Generate ECG with atrial fibrillation pattern
    
    Characteristics:
    - Irregular RR intervals
    - No clear P waves
    - Fibrillatory baseline
    """
    num_samples = int(sample_rate * duration)
    ecg = np.zeros(num_samples)
    
    # Variable heart rate for AFib
    current_pos = 0
    
    while current_pos < num_samples - 100:
        # Irregular RR interval (varies 20-50%)
        base_interval = 60 / heart_rate
        variation = np.random.uniform(0.7, 1.3)
        beat_interval = int(sample_rate * base_interval * variation)
        
        # No P wave - just fibrillatory waves
        fib_segment = 0.05 * np.random.randn(min(30, num_samples - current_pos))
        if current_pos + len(fib_segment) < num_samples:
            ecg[current_pos:current_pos + len(fib_segment)] = fib_segment
        
        # QRS complex (normal width)
        qrs_pos = current_pos + 30
        if qrs_pos < num_samples - 20:
            ecg[qrs_pos] = -0.1
            ecg[qrs_pos + 2] = 1.0
            ecg[qrs_pos + 4] = -0.2
        
        # T wave
        t_start = current_pos + 60
        t_end = min(current_pos + 90, num_samples)
        if t_end > t_start:
            ecg[t_start:t_end] = 0.25 * np.sin(
                np.linspace(0, np.pi, t_end - t_start)
            )
        
        current_pos += beat_interval
    
    # Add fibrillatory baseline throughout
    fib_noise = 0.03 * np.random.randn(num_samples)
    ecg += fib_noise
    
    return ecg


def generate_bradycardia_ecg(
    duration: float = 10.0,
    sample_rate: int = 500,
    heart_rate: int = 45
) -> np.ndarray:
    """Generate ECG with bradycardia (slow heart rate)"""
    return generate_demo_ecg(
        duration=duration,
        sample_rate=sample_rate,
        heart_rate=heart_rate,
        add_arrhythmia=False
    )


def generate_tachycardia_ecg(
    duration: float = 10.0,
    sample_rate: int = 500,
    heart_rate: int = 130
) -> np.ndarray:
    """Generate ECG with tachycardia (fast heart rate)"""
    return generate_demo_ecg(
        duration=duration,
        sample_rate=sample_rate,
        heart_rate=heart_rate,
        add_arrhythmia=False
    )
