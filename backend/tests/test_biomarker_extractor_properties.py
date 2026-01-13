"""
Property-based tests for BiomarkerExtractor
Uses hypothesis for property-based testing.

Feature: speech-pipeline-fix
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
import hypothesis.extra.numpy as npst

from app.services.biomarker_extractor import BiomarkerExtractor, ExtractedBiomarkers


class TestBiomarkerRangeValidityProperty:
    """
    Property 5: Biomarker Range Validity
    
    For any successful speech analysis, all biomarkers SHALL be within their valid ranges:
    - jitter: [0, 1]
    - shimmer: [0, 1]
    - hnr: [0, 30]
    - speech_rate: [0.5, 10] syllables/second
    - pause_ratio: [0, 1]
    - fluency_score: [0, 1]
    - voice_tremor: [0, 1]
    - articulation_clarity: [0, 1]
    - prosody_variation: [0, 1]
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    
    SAMPLE_RATE = 16000
    
    @given(
        duration=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
        frequency=st.floats(min_value=100.0, max_value=300.0, allow_nan=False),
        amplitude=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_jitter_within_valid_range(self, duration, frequency, amplitude):
        """
        Property: For any audio input, jitter is always in [0, 1]
        
        Feature: speech-pipeline-fix, Property 5: Biomarker Range Validity
        **Validates: Requirements 4.1**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate synthetic audio with given parameters
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples)
        audio_data = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Add some noise for realism
        audio_data += np.random.randn(num_samples).astype(np.float32) * 0.01
        
        jitter, is_estimated = extractor.calculate_jitter(audio_data)
        
        assert 0.0 <= jitter <= 1.0, f"Jitter {jitter} out of range [0, 1]"
    
    @given(
        duration=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
        amplitude=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_shimmer_within_valid_range(self, duration, amplitude):
        """
        Property: For any audio input, shimmer is always in [0, 1]
        
        Feature: speech-pipeline-fix, Property 5: Biomarker Range Validity
        **Validates: Requirements 4.2**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate synthetic audio
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples)
        frequency = 150.0  # Typical F0
        audio_data = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Add amplitude modulation for shimmer variation
        modulation = 1.0 + 0.1 * np.sin(2 * np.pi * 5 * t)
        audio_data = (audio_data * modulation).astype(np.float32)
        
        shimmer, is_estimated = extractor.calculate_shimmer(audio_data)
        
        assert 0.0 <= shimmer <= 1.0, f"Shimmer {shimmer} out of range [0, 1]"
    
    @given(
        duration=st.floats(min_value=0.5, max_value=3.0, allow_nan=False),
        frequency=st.floats(min_value=100.0, max_value=300.0, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_hnr_within_valid_range(self, duration, frequency):
        """
        Property: For any audio input, HNR is always in [0, 30] dB
        
        Feature: speech-pipeline-fix, Property 5: Biomarker Range Validity
        **Validates: Requirements 4.3**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate synthetic audio
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Add harmonics
        audio_data += 0.2 * np.sin(2 * np.pi * 2 * frequency * t).astype(np.float32)
        audio_data += 0.1 * np.sin(2 * np.pi * 3 * frequency * t).astype(np.float32)
        
        # Add noise
        audio_data += np.random.randn(num_samples).astype(np.float32) * 0.05
        
        hnr, is_estimated = extractor.calculate_hnr(audio_data)
        
        assert 0.0 <= hnr <= 30.0, f"HNR {hnr} out of range [0, 30]"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
        num_syllables=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_speech_rate_within_valid_range(self, duration, num_syllables):
        """
        Property: For any audio input, speech_rate is always in [0.5, 10] syllables/second
        
        Feature: speech-pipeline-fix, Property 5: Biomarker Range Validity
        **Validates: Requirements 4.4**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate synthetic audio with syllable-like energy bursts
        num_samples = int(duration * self.SAMPLE_RATE)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        # Add energy bursts for syllables
        syllable_duration = int(0.15 * self.SAMPLE_RATE)  # 150ms per syllable
        spacing = num_samples // (num_syllables + 1)
        
        for i in range(num_syllables):
            start = (i + 1) * spacing - syllable_duration // 2
            end = min(start + syllable_duration, num_samples)
            if start >= 0 and start < num_samples:
                t = np.linspace(0, 0.15, end - start)
                audio_data[start:end] = 0.5 * np.sin(2 * np.pi * 150 * t)
        
        speech_rate, is_estimated = extractor.calculate_speech_rate(audio_data)
        
        assert 0.5 <= speech_rate <= 10.0, f"Speech rate {speech_rate} out of range [0.5, 10]"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
        speech_proportion=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_pause_ratio_within_valid_range(self, duration, speech_proportion):
        """
        Property: For any audio input, pause_ratio is always in [0, 1]
        
        Feature: speech-pipeline-fix, Property 5: Biomarker Range Validity
        **Validates: Requirements 4.5**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate synthetic audio with speech and silence
        num_samples = int(duration * self.SAMPLE_RATE)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        # Add speech in proportion of the audio
        speech_samples = int(num_samples * speech_proportion)
        t = np.linspace(0, speech_samples / self.SAMPLE_RATE, speech_samples)
        audio_data[:speech_samples] = 0.5 * np.sin(2 * np.pi * 150 * t)
        
        pause_ratio, is_estimated = extractor.calculate_pause_ratio(audio_data)
        
        assert 0.0 <= pause_ratio <= 1.0, f"Pause ratio {pause_ratio} out of range [0, 1]"

    @given(
        duration=st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
        amplitude=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_all_biomarkers_within_valid_ranges(self, duration, amplitude):
        """
        Property: For any audio input, all biomarkers are within their valid ranges
        
        Feature: speech-pipeline-fix, Property 5: Biomarker Range Validity
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
        """
        import asyncio
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate realistic synthetic speech-like audio
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples)
        
        # Base frequency with slight variation (simulating natural speech)
        f0 = 150.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)
        
        # Generate audio with harmonics
        audio_data = amplitude * np.sin(2 * np.pi * f0 * t / self.SAMPLE_RATE * np.cumsum(np.ones(num_samples)))
        audio_data += 0.3 * amplitude * np.sin(4 * np.pi * f0 * t / self.SAMPLE_RATE * np.cumsum(np.ones(num_samples)))
        audio_data = audio_data.astype(np.float32)
        
        # Add some noise
        audio_data += np.random.randn(num_samples).astype(np.float32) * 0.02
        
        # Extract all biomarkers
        biomarkers = asyncio.get_event_loop().run_until_complete(
            extractor.extract_all(audio_data)
        )
        
        # Verify all ranges
        assert 0.0 <= biomarkers.jitter <= 1.0, f"Jitter {biomarkers.jitter} out of range"
        assert 0.0 <= biomarkers.shimmer <= 1.0, f"Shimmer {biomarkers.shimmer} out of range"
        assert 0.0 <= biomarkers.hnr <= 30.0, f"HNR {biomarkers.hnr} out of range"
        assert 0.5 <= biomarkers.speech_rate <= 10.0, f"Speech rate {biomarkers.speech_rate} out of range"
        assert 0.0 <= biomarkers.pause_ratio <= 1.0, f"Pause ratio {biomarkers.pause_ratio} out of range"
        assert 0.0 <= biomarkers.fluency_score <= 1.0, f"Fluency score {biomarkers.fluency_score} out of range"
        assert 0.0 <= biomarkers.voice_tremor <= 1.0, f"Voice tremor {biomarkers.voice_tremor} out of range"
        assert 0.0 <= biomarkers.articulation_clarity <= 1.0, f"Articulation clarity {biomarkers.articulation_clarity} out of range"
        assert 0.0 <= biomarkers.prosody_variation <= 1.0, f"Prosody variation {biomarkers.prosody_variation} out of range"



class TestBiomarkerFallbackConsistencyProperty:
    """
    Property 6: Biomarker Fallback Consistency
    
    For any biomarker extraction where a specific metric fails, the returned 
    value SHALL equal the clinically-validated default AND the is_estimated 
    flag SHALL be true for that metric.
    
    **Validates: Requirements 4.6**
    """
    
    SAMPLE_RATE = 16000
    
    def test_empty_audio_uses_fallback_for_jitter(self):
        """
        Property: Empty audio returns default jitter with is_estimated=True
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Empty audio should trigger fallback
        empty_audio = np.array([], dtype=np.float32)
        
        jitter, is_estimated = extractor.calculate_jitter(empty_audio)
        
        assert is_estimated is True, "Empty audio should set is_estimated=True"
        assert jitter == BiomarkerExtractor.DEFAULT_VALUES['jitter'], \
            f"Fallback jitter {jitter} != default {BiomarkerExtractor.DEFAULT_VALUES['jitter']}"
    
    def test_empty_audio_uses_fallback_for_shimmer(self):
        """
        Property: Empty audio returns default shimmer with is_estimated=True
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        empty_audio = np.array([], dtype=np.float32)
        
        shimmer, is_estimated = extractor.calculate_shimmer(empty_audio)
        
        assert is_estimated is True, "Empty audio should set is_estimated=True"
        assert shimmer == BiomarkerExtractor.DEFAULT_VALUES['shimmer'], \
            f"Fallback shimmer {shimmer} != default {BiomarkerExtractor.DEFAULT_VALUES['shimmer']}"
    
    def test_empty_audio_uses_fallback_for_hnr(self):
        """
        Property: Empty audio returns default HNR with is_estimated=True
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        empty_audio = np.array([], dtype=np.float32)
        
        hnr, is_estimated = extractor.calculate_hnr(empty_audio)
        
        assert is_estimated is True, "Empty audio should set is_estimated=True"
        assert hnr == BiomarkerExtractor.DEFAULT_VALUES['hnr'], \
            f"Fallback HNR {hnr} != default {BiomarkerExtractor.DEFAULT_VALUES['hnr']}"
    
    def test_empty_audio_uses_fallback_for_speech_rate(self):
        """
        Property: Empty audio returns default speech_rate with is_estimated=True
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        empty_audio = np.array([], dtype=np.float32)
        
        speech_rate, is_estimated = extractor.calculate_speech_rate(empty_audio)
        
        assert is_estimated is True, "Empty audio should set is_estimated=True"
        assert speech_rate == BiomarkerExtractor.DEFAULT_VALUES['speech_rate'], \
            f"Fallback speech_rate {speech_rate} != default {BiomarkerExtractor.DEFAULT_VALUES['speech_rate']}"
    
    def test_empty_audio_uses_fallback_for_pause_ratio(self):
        """
        Property: Empty audio returns default pause_ratio with is_estimated=True
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        empty_audio = np.array([], dtype=np.float32)
        
        pause_ratio, is_estimated = extractor.calculate_pause_ratio(empty_audio)
        
        assert is_estimated is True, "Empty audio should set is_estimated=True"
        assert pause_ratio == BiomarkerExtractor.DEFAULT_VALUES['pause_ratio'], \
            f"Fallback pause_ratio {pause_ratio} != default {BiomarkerExtractor.DEFAULT_VALUES['pause_ratio']}"
    
    def test_too_short_audio_uses_fallback_for_jitter(self):
        """
        Property: Audio too short for F0 analysis returns default jitter
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Very short audio (10ms) - not enough for F0 analysis
        short_audio = np.random.randn(int(0.01 * self.SAMPLE_RATE)).astype(np.float32)
        
        jitter, is_estimated = extractor.calculate_jitter(short_audio)
        
        # Should either succeed or use fallback
        assert 0.0 <= jitter <= 1.0, "Jitter should be in valid range"
        if is_estimated:
            assert jitter == BiomarkerExtractor.DEFAULT_VALUES['jitter'], \
                "If estimated, should equal default value"
    
    @given(metric_name=st.sampled_from([
        'jitter', 'shimmer', 'hnr', 'speech_rate', 'pause_ratio',
        'fluency_score', 'voice_tremor', 'articulation_clarity', 'prosody_variation'
    ]))
    @settings(max_examples=100)
    def test_fallback_values_are_within_valid_ranges(self, metric_name):
        """
        Property: All default fallback values are within their valid ranges
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        default_value = BiomarkerExtractor.DEFAULT_VALUES[metric_name]
        min_val, max_val = BiomarkerExtractor.VALID_RANGES[metric_name]
        
        assert min_val <= default_value <= max_val, \
            f"Default {metric_name}={default_value} not in range [{min_val}, {max_val}]"
    
    def test_extract_all_with_empty_audio_sets_all_estimated_flags(self):
        """
        Property: extract_all with empty audio sets is_estimated=True for all metrics
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        import asyncio
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        empty_audio = np.array([], dtype=np.float32)
        
        biomarkers = asyncio.get_event_loop().run_until_complete(
            extractor.extract_all(empty_audio)
        )
        
        # All metrics should be estimated for empty audio
        for metric_name, is_estimated in biomarkers.estimated_flags.items():
            assert is_estimated is True, \
                f"Empty audio should set is_estimated=True for {metric_name}"
            
            # Verify the value equals the default
            actual_value = getattr(biomarkers, metric_name)
            expected_value = BiomarkerExtractor.DEFAULT_VALUES[metric_name]
            assert actual_value == expected_value, \
                f"Fallback {metric_name}={actual_value} != default {expected_value}"
    
    @given(
        duration=st.floats(min_value=1.0, max_value=3.0, allow_nan=False),
        amplitude=st.floats(min_value=0.3, max_value=0.7, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_valid_audio_does_not_use_fallback(self, duration, amplitude):
        """
        Property: Valid audio with clear speech should not use fallback values
        
        Feature: speech-pipeline-fix, Property 6: Biomarker Fallback Consistency
        **Validates: Requirements 4.6**
        """
        import asyncio
        extractor = BiomarkerExtractor(sample_rate=self.SAMPLE_RATE)
        
        # Generate clear synthetic speech
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples)
        
        # Generate audio with clear F0 and harmonics
        f0 = 150.0
        audio_data = amplitude * np.sin(2 * np.pi * f0 * t).astype(np.float32)
        audio_data += 0.3 * amplitude * np.sin(2 * np.pi * 2 * f0 * t).astype(np.float32)
        audio_data += 0.15 * amplitude * np.sin(2 * np.pi * 3 * f0 * t).astype(np.float32)
        
        biomarkers = asyncio.get_event_loop().run_until_complete(
            extractor.extract_all(audio_data)
        )
        
        # At least some metrics should be calculated (not estimated)
        non_estimated_count = sum(
            1 for is_est in biomarkers.estimated_flags.values() if not is_est
        )
        
        # With valid audio, we expect at least some metrics to be calculated
        assert non_estimated_count > 0, \
            "Valid audio should have at least some non-estimated metrics"
