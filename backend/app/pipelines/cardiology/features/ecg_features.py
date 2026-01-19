"""
Cardiology Pipeline - ECG Feature Extractor
R-peak detection, HRV metrics, and interval measurements.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import logging

from ..config import HRV_NORMAL_RANGES, INTERVAL_NORMAL_RANGES, INPUT_CONSTRAINTS

logger = logging.getLogger(__name__)

# Try importing optional dependencies
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


@dataclass
class Beat:
    """Single heartbeat segment."""
    signal: np.ndarray
    r_peak_index: int
    start_sample: int
    end_sample: int
    rr_preceding: Optional[int] = None
    rr_succeeding: Optional[int] = None
    beat_type: str = "normal"


@dataclass
class HRVMetrics:
    """Heart rate variability metrics."""
    rmssd_ms: Optional[float] = None
    sdnn_ms: Optional[float] = None
    pnn50_percent: Optional[float] = None
    mean_rr_ms: Optional[float] = None
    sdsd_ms: Optional[float] = None
    cv_rr_percent: Optional[float] = None
    mean_hr_bpm: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ECGIntervals:
    """ECG interval measurements."""
    pr_interval_ms: Optional[float] = None
    qrs_duration_ms: Optional[float] = None
    qt_interval_ms: Optional[float] = None
    qtc_ms: Optional[float] = None
    rr_interval_ms: Optional[float] = None
    all_normal: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ECGFeatures:
    """Complete ECG feature extraction results."""
    r_peaks: List[int]
    rr_intervals: List[int]
    heart_rate_bpm: float
    hrv: HRVMetrics
    intervals: ECGIntervals
    beats: List[Beat] = field(default_factory=list)
    extraction_method: str = "unknown"


class RPeakDetector:
    """Detect R-peaks in ECG signal."""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
    
    def detect(self, signal: np.ndarray) -> Tuple[List[int], str]:
        """
        Detect R-peak locations.
        
        Args:
            signal: Preprocessed ECG signal
        
        Returns:
            Tuple of (r_peak_indices, method_used)
        """
        # Try HeartPy first
        if HEARTPY_AVAILABLE:
            try:
                peaks = self._detect_heartpy(signal)
                if len(peaks) >= 3:
                    return peaks, "heartpy"
            except Exception as e:
                logger.debug(f"HeartPy detection failed: {e}")
        
        # Try NeuroKit2
        if NEUROKIT_AVAILABLE:
            try:
                peaks = self._detect_neurokit(signal)
                if len(peaks) >= 3:
                    return peaks, "neurokit2"
            except Exception as e:
                logger.debug(f"NeuroKit2 detection failed: {e}")
        
        # Fallback: simple peak detection
        peaks = self._detect_simple(signal)
        return peaks, "simple_threshold"
    
    def _detect_heartpy(self, signal: np.ndarray) -> List[int]:
        """Use HeartPy for R-peak detection."""
        wd, _ = hp.process(signal, sample_rate=self.sample_rate)
        peaks = wd['peaklist']
        return [int(p) for p in peaks if not np.isnan(p)]
    
    def _detect_neurokit(self, signal: np.ndarray) -> List[int]:
        """Use NeuroKit2 for R-peak detection."""
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=self.sample_rate)
        peaks = rpeaks['ECG_R_Peaks']
        return [int(p) for p in peaks if not np.isnan(p)]
    
    def _detect_simple(self, signal: np.ndarray) -> List[int]:
        """Simple threshold-based peak detection."""
        from scipy.signal import find_peaks
        
        height = np.percentile(signal, 90)
        min_distance = int(0.4 * self.sample_rate)
        
        peaks, _ = find_peaks(
            signal,
            height=height,
            distance=min_distance
        )
        
        return peaks.tolist()


class HRVCalculator:
    """Calculate HRV time-domain metrics."""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
    
    def calculate(self, rr_intervals: List[int]) -> HRVMetrics:
        """
        Calculate HRV metrics from RR intervals.
        
        Args:
            rr_intervals: RR intervals in samples
        
        Returns:
            HRVMetrics with all time-domain values
        """
        if len(rr_intervals) < INPUT_CONSTRAINTS["ecg"]["min_r_peaks"]:
            return HRVMetrics()
        
        # Convert to milliseconds
        rr_ms = np.array(rr_intervals) * 1000 / self.sample_rate
        
        # Filter out outliers (physiologically implausible)
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        
        if len(rr_ms) < 5:
            return HRVMetrics()
        
        # Successive differences
        diff_rr = np.diff(rr_ms)
        
        # RMSSD: Root mean square of successive differences
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        
        # SDNN: Standard deviation of NN intervals
        sdnn = np.std(rr_ms)
        
        # pNN50: Percentage of successive differences > 50ms
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        # Mean RR interval
        mean_rr = np.mean(rr_ms)
        
        # SDSD: Standard deviation of successive differences
        sdsd = np.std(diff_rr)
        
        # Coefficient of variation
        cv_rr = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        
        # Mean heart rate
        mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
        
        return HRVMetrics(
            rmssd_ms=round(rmssd, 2),
            sdnn_ms=round(sdnn, 2),
            pnn50_percent=round(pnn50, 2),
            mean_rr_ms=round(mean_rr, 2),
            sdsd_ms=round(sdsd, 2),
            cv_rr_percent=round(cv_rr, 2),
            mean_hr_bpm=round(mean_hr, 1),
        )


class IntervalCalculator:
    """Calculate ECG interval durations (PR, QRS, QT)."""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
    
    def calculate(
        self,
        signal: np.ndarray,
        r_peaks: List[int],
        heart_rate: float
    ) -> ECGIntervals:
        """
        Calculate ECG intervals.
        
        Note: Accurate interval measurement requires multi-lead ECG.
        These estimates are approximations from single-lead data.
        """
        if len(r_peaks) < 2:
            return ECGIntervals()
        
        # Mean RR interval
        rr_intervals = np.diff(r_peaks)
        mean_rr_ms = np.mean(rr_intervals) * 1000 / self.sample_rate
        
        # QRS duration estimate (from R-peak width)
        qrs = self._estimate_qrs(signal, r_peaks)
        
        # QT interval estimate
        qt = self._estimate_qt(mean_rr_ms)
        
        # QTc (Bazett's formula)
        qtc = None
        if qt and mean_rr_ms > 0:
            rr_sec = mean_rr_ms / 1000
            qtc = qt / np.sqrt(rr_sec)
        
        # Check if intervals are normal
        all_normal = True
        if qrs and qrs > INTERVAL_NORMAL_RANGES["qrs_duration_ms"]["abnormal_high"]:
            all_normal = False
        if qtc and qtc > INTERVAL_NORMAL_RANGES["qtc_ms"]["abnormal_high"]:
            all_normal = False
        
        return ECGIntervals(
            qrs_duration_ms=round(qrs, 1) if qrs else None,
            qt_interval_ms=round(qt, 1) if qt else None,
            qtc_ms=round(qtc, 1) if qtc else None,
            rr_interval_ms=round(mean_rr_ms, 1),
            all_normal=all_normal,
        )
    
    def _estimate_qrs(
        self,
        signal: np.ndarray,
        r_peaks: List[int]
    ) -> Optional[float]:
        """Estimate QRS duration from R-peak width."""
        if len(r_peaks) < 3:
            return None
        
        widths = []
        for peak in r_peaks[1:-1]:
            # Find approximate QRS boundaries
            left = max(0, peak - int(0.15 * self.sample_rate))
            right = min(len(signal), peak + int(0.15 * self.sample_rate))
            
            segment = signal[left:right]
            threshold = signal[peak] * 0.3
            
            above_threshold = segment > threshold
            if np.any(above_threshold):
                indices = np.where(above_threshold)[0]
                width_samples = indices[-1] - indices[0]
                width_ms = width_samples * 1000 / self.sample_rate
                if 40 < width_ms < 200:
                    widths.append(width_ms)
        
        return np.median(widths) if widths else None
    
    def _estimate_qt(self, mean_rr_ms: float) -> Optional[float]:
        """Estimate QT interval from heart rate using regression."""
        # QT approximation: QT = 0.38 * sqrt(RR)
        if mean_rr_ms <= 0:
            return None
        
        qt = 380 * np.sqrt(mean_rr_ms / 1000)
        return min(max(qt, 280), 520)  # Clamp to reasonable range


class BeatSegmenter:
    """Segment ECG into individual beats."""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
        self.pre_r_ms = 200
        self.post_r_ms = 600
    
    def segment(
        self,
        signal: np.ndarray,
        r_peaks: List[int]
    ) -> List[Beat]:
        """
        Segment signal into individual beats.
        
        Args:
            signal: ECG signal
            r_peaks: R-peak indices
        
        Returns:
            List of Beat objects
        """
        beats = []
        
        pre_samples = int(self.pre_r_ms * self.sample_rate / 1000)
        post_samples = int(self.post_r_ms * self.sample_rate / 1000)
        
        for i, peak in enumerate(r_peaks):
            start = max(0, peak - pre_samples)
            end = min(len(signal), peak + post_samples)
            
            if end - start < pre_samples:
                continue
            
            # RR intervals
            rr_pre = r_peaks[i] - r_peaks[i-1] if i > 0 else None
            rr_post = r_peaks[i+1] - r_peaks[i] if i < len(r_peaks) - 1 else None
            
            beats.append(Beat(
                signal=signal[start:end],
                r_peak_index=peak,
                start_sample=start,
                end_sample=end,
                rr_preceding=rr_pre,
                rr_succeeding=rr_post,
                beat_type="normal",
            ))
        
        return beats


class ECGFeatureExtractor:
    """Main ECG feature extraction class."""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
        self.r_peak_detector = RPeakDetector(sample_rate)
        self.hrv_calculator = HRVCalculator(sample_rate)
        self.interval_calculator = IntervalCalculator(sample_rate)
        self.beat_segmenter = BeatSegmenter(sample_rate)
    
    def extract(self, signal: np.ndarray) -> ECGFeatures:
        """
        Extract all ECG features.
        
        Args:
            signal: Preprocessed ECG signal
        
        Returns:
            ECGFeatures with all metrics
        """
        # Detect R-peaks
        r_peaks, method = self.r_peak_detector.detect(signal)
        
        if len(r_peaks) < 2:
            logger.warning("Insufficient R-peaks detected")
            return ECGFeatures(
                r_peaks=[],
                rr_intervals=[],
                heart_rate_bpm=0,
                hrv=HRVMetrics(),
                intervals=ECGIntervals(),
                extraction_method=method,
            )
        
        # Calculate RR intervals
        rr_intervals = [r_peaks[i+1] - r_peaks[i] for i in range(len(r_peaks) - 1)]
        
        # Calculate heart rate
        mean_rr_samples = np.mean(rr_intervals)
        mean_rr_sec = mean_rr_samples / self.sample_rate
        heart_rate = 60 / mean_rr_sec if mean_rr_sec > 0 else 0
        
        # Calculate HRV
        hrv = self.hrv_calculator.calculate(rr_intervals)
        
        # Calculate intervals
        intervals = self.interval_calculator.calculate(signal, r_peaks, heart_rate)
        
        # Segment beats
        beats = self.beat_segmenter.segment(signal, r_peaks)
        
        return ECGFeatures(
            r_peaks=r_peaks,
            rr_intervals=rr_intervals,
            heart_rate_bpm=round(heart_rate, 1),
            hrv=hrv,
            intervals=intervals,
            beats=beats,
            extraction_method=method,
        )
