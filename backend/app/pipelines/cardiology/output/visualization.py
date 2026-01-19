"""
Cardiology Pipeline - ECG Visualization
Generate visualization data for ECG display.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ECGAnnotation:
    """Annotation for ECG visualization."""
    type: str  # r_peak, interval, abnormal_beat
    sample_index: int
    time_sec: float
    label: Optional[str] = None
    style: Optional[Dict[str, Any]] = None


@dataclass
class ECGPlotData:
    """Data for ECG plot visualization."""
    waveform: List[float]
    sample_rate: int
    duration_sec: float
    annotations: List[ECGAnnotation] = field(default_factory=list)
    time_axis: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "waveform": self.waveform,
            "sample_rate": self.sample_rate,
            "duration_sec": self.duration_sec,
            "annotations": [
                {
                    "type": a.type,
                    "sample_index": a.sample_index,
                    "time_sec": a.time_sec,
                    "label": a.label,
                }
                for a in self.annotations
            ],
        }


class ECGVisualizer:
    """Generate ECG visualization data."""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
    
    def create_plot_data(
        self,
        signal: np.ndarray,
        r_peaks: Optional[List[int]] = None,
        arrhythmias: Optional[List[Dict[str, Any]]] = None,
        downsample_factor: int = 1,
        max_samples: int = 50000
    ) -> ECGPlotData:
        """
        Create plot data for ECG visualization.
        
        Args:
            signal: ECG signal array
            r_peaks: R-peak indices
            arrhythmias: Detected arrhythmias
            downsample_factor: Downsample signal for efficiency
            max_samples: Maximum samples to return
        
        Returns:
            ECGPlotData for frontend visualization
        """
        # Downsample if needed
        if len(signal) > max_samples:
            factor = len(signal) // max_samples + 1
            signal = signal[::factor]
            if r_peaks:
                r_peaks = [p // factor for p in r_peaks]
            effective_sr = self.sample_rate // factor
        else:
            effective_sr = self.sample_rate
        
        # Apply additional downsampling if requested
        if downsample_factor > 1:
            signal = signal[::downsample_factor]
            if r_peaks:
                r_peaks = [p // downsample_factor for p in r_peaks]
            effective_sr = effective_sr // downsample_factor
        
        duration = len(signal) / effective_sr
        
        # Create annotations
        annotations = []
        
        # R-peak markers
        if r_peaks:
            for i, peak in enumerate(r_peaks):
                if 0 <= peak < len(signal):
                    annotations.append(ECGAnnotation(
                        type="r_peak",
                        sample_index=peak,
                        time_sec=peak / effective_sr,
                        label="R",
                        style={"color": "#ef4444", "size": 6},
                    ))
        
        # Generate time axis
        time_axis = [i / effective_sr for i in range(len(signal))]
        
        return ECGPlotData(
            waveform=signal.tolist(),
            sample_rate=effective_sr,
            duration_sec=duration,
            annotations=annotations,
            time_axis=time_axis,
        )
    
    def create_segment_data(
        self,
        signal: np.ndarray,
        start_sec: float,
        duration_sec: float,
        r_peaks: Optional[List[int]] = None
    ) -> ECGPlotData:
        """
        Create plot data for a specific segment.
        
        Args:
            signal: Full ECG signal
            start_sec: Start time in seconds
            duration_sec: Duration to extract
            r_peaks: R-peak indices
        
        Returns:
            ECGPlotData for segment
        """
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int((start_sec + duration_sec) * self.sample_rate)
        
        segment = signal[start_sample:end_sample]
        
        # Adjust R-peaks to segment
        segment_peaks = []
        if r_peaks:
            for peak in r_peaks:
                if start_sample <= peak < end_sample:
                    segment_peaks.append(peak - start_sample)
        
        return self.create_plot_data(
            segment,
            r_peaks=segment_peaks,
            downsample_factor=1,
        )


def create_ecg_plot_data(
    signal: np.ndarray,
    sample_rate: int = 500,
    r_peaks: Optional[List[int]] = None,
    max_samples: int = 10000
) -> Dict[str, Any]:
    """
    Convenience function to create ECG plot data.
    
    Args:
        signal: ECG signal
        sample_rate: Sample rate
        r_peaks: R-peak indices
        max_samples: Maximum samples
    
    Returns:
        Dictionary with plot data
    """
    visualizer = ECGVisualizer(sample_rate)
    plot_data = visualizer.create_plot_data(
        signal,
        r_peaks=r_peaks,
        max_samples=max_samples,
    )
    return plot_data.to_dict()
