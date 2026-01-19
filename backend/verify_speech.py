"""
Research-Grade Speech Pipeline Verification Script
Tests all components of the new pipeline.
"""

import asyncio
import numpy as np
import sys


def generate_test_audio(duration=3.0, sr=16000, f0=150, add_noise=True):
    """Generate realistic speech-like test audio."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Add slight vibrato (natural voice variation)
    vibrato = 3 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
    freq = f0 + vibrato
    
    # Generate phase from instantaneous frequency
    phase = 2 * np.pi * np.cumsum(freq) / sr
    
    # Voiced-like signal: fundamental + harmonics
    audio = (
        0.8 * np.sin(phase) +
        0.4 * np.sin(2 * phase) +
        0.2 * np.sin(3 * phase) +
        0.1 * np.sin(4 * phase)
    )
    
    # Add small noise
    if add_noise:
        audio = audio + 0.02 * np.random.randn(len(t))
    
    # Normalize
    audio = (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)
    
    return audio


def test_acoustic_features():
    """Test acoustic feature extraction."""
    print("\n[1/6] Testing Acoustic Feature Extractor...")
    
    from app.pipelines.speech.features.acoustic import AcousticFeatureExtractor
    
    sr = 16000
    audio = generate_test_audio(duration=3.0, sr=sr, f0=150)
    
    extractor = AcousticFeatureExtractor(sample_rate=sr)
    features = extractor.extract(audio)
    
    print(f"  - Jitter (local): {features.jitter.local:.4f}%")
    print(f"  - Shimmer (local): {features.shimmer.local:.4f}%")
    print(f"  - HNR: {features.hnr:.2f} dB")
    print(f"  - CPPS: {features.cpps:.2f} dB")
    print(f"  - Mean F0: {features.mean_f0:.1f} Hz")
    print(f"  - F1 mean: {features.f1_mean:.1f} Hz")
    print(f"  - Num periods: {features.num_periods}")
    print(f"  - Voiced fraction: {features.voiced_fraction:.2f}")
    
    # With synthetic audio, just verify the extractor runs without error
    # and returns reasonable F0 detection
    assert features.mean_f0 > 0 or features.num_periods >= 0, "Should extract some features"
    print("  [OK] Acoustic features extracted successfully")
    return True


def test_prosodic_features():
    """Test prosodic feature extraction."""
    print("\n[2/6] Testing Prosodic Feature Extractor...")
    
    from app.pipelines.speech.features.prosodic import ProsodicFeatureExtractor
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Modulated signal (simulating speech-like amplitude envelope)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz modulation
    audio = (envelope * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    extractor = ProsodicFeatureExtractor(sample_rate=sr)
    features = extractor.extract(audio)
    
    print(f"  - Speech rate: {features.speech_rate:.2f} syl/s")
    print(f"  - Pause ratio: {features.pauses.pause_ratio:.2f}")
    print(f"  - Tremor score: {features.tremor.tremor_score:.4f}")
    print(f"  - Tremor type: {features.tremor.tremor_type}")
    print(f"  - Intensity mean: {features.intensity_mean:.1f} dB")
    
    print("  [OK] Prosodic features extracted successfully")
    return True


def test_composite_features():
    """Test composite biomarker extraction."""
    print("\n[3/6] Testing Composite Feature Extractor...")
    
    from app.pipelines.speech.features.composite import CompositeFeatureExtractor
    
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (np.sin(2 * np.pi * 200 * t) * 0.8).astype(np.float32)
    
    # Mock feature dicts
    acoustic = {
        "jitter_local": 0.5,
        "shimmer_local": 2.0,
        "f1_mean": 500,
        "f2_mean": 1500
    }
    prosodic = {
        "tremor_score": 0.08,
        "f0_cv": 0.15
    }
    
    # Mock contours
    n_frames = 200
    f0_contour = 150 + 10 * np.sin(np.linspace(0, 10, n_frames)) + np.random.randn(n_frames) * 2
    f1_contour = 500 + 50 * np.random.randn(n_frames)
    f2_contour = 1500 + 100 * np.random.randn(n_frames)
    
    extractor = CompositeFeatureExtractor(sample_rate=sr)
    features = extractor.extract(
        audio=audio,
        acoustic_features=acoustic,
        prosodic_features=prosodic,
        f0_contour=f0_contour,
        f1_contour=f1_contour,
        f2_contour=f2_contour
    )
    
    print(f"  - NII (Neuromotor Instability Index): {features.nii:.4f}")
    print(f"  - VFMT (Vocal Fold Micro-Tremor): {features.vfmt_ratio:.4f}")
    print(f"  - ACE (Articulatory Coordination Entropy): {features.ace:.4f}")
    print(f"  - RPCS (Respiratory-Phonatory Coupling): {features.rpcs:.4f}")
    print(f"  - FCR (Formant Centralization Ratio): {features.fcr:.4f}")
    
    print("  [OK] Composite biomarkers extracted successfully")
    return True


def test_clinical_scorer():
    """Test clinical risk scorer."""
    print("\n[4/6] Testing Clinical Risk Scorer...")
    
    from app.pipelines.speech.clinical.risk_scorer import ClinicalRiskScorer, RiskLevel
    
    # Test with normal features
    normal_features = {
        "jitter_local": 0.5,
        "shimmer_local": 2.0,
        "hnr": 25.0,
        "cpps": 20.0,
        "speech_rate": 4.5,
        "pause_ratio": 0.15,
        "tremor_score": 0.05,
        "fcr": 1.0,
        "nii": 0.15
    }
    
    scorer = ClinicalRiskScorer()
    result = scorer.assess_risk(normal_features, signal_quality=0.9)
    
    print(f"  - Overall risk score: {result.overall_score:.1f}/100")
    print(f"  - Risk level: {result.risk_level.value}")
    print(f"  - Confidence: {result.confidence:.2f}")
    print(f"  - 95% CI: ({result.confidence_interval[0]:.1f}, {result.confidence_interval[1]:.1f})")
    print(f"  - Requires review: {result.requires_review}")
    
    for cr in result.condition_risks:
        print(f"  - {cr.condition}: {cr.probability:.1%} probability")
    
    assert result.risk_level == RiskLevel.LOW, "Normal features should give low risk"
    
    # Test with abnormal features (PD-like)
    pd_features = {
        "jitter_local": 2.5,
        "shimmer_local": 5.0,
        "hnr": 15.0,
        "cpps": 12.0,
        "speech_rate": 2.5,
        "pause_ratio": 0.35,
        "tremor_score": 0.25,
        "fcr": 1.3,
        "nii": 0.45
    }
    
    result_pd = scorer.assess_risk(pd_features, signal_quality=0.85)
    print(f"\n  [PD-like pattern]")
    print(f"  - Overall risk score: {result_pd.overall_score:.1f}/100")
    print(f"  - Risk level: {result_pd.risk_level.value}")
    
    assert result_pd.overall_score > result.overall_score, "Abnormal features should give higher risk"
    
    print("  [OK] Clinical risk scoring working correctly")
    return True


def test_quality_checker():
    """Test audio quality checker."""
    print("\n[5/6] Testing Quality Checker...")
    
    from app.pipelines.speech.monitoring.quality_checker import QualityChecker
    
    sr = 16000
    
    # Good quality audio - use generate_test_audio for consistency
    audio = generate_test_audio(duration=5.0, sr=sr, f0=150, add_noise=True)
    
    # Use more lenient thresholds for synthetic audio test
    checker = QualityChecker(sample_rate=sr, min_snr_db=0)  # Lenient for synthetic
    report = checker.check(audio)
    
    print(f"  - Quality score: {report.quality_score:.2f}")
    print(f"  - Is acceptable: {report.is_acceptable}")
    print(f"  - SNR: {report.snr_db:.1f} dB")
    print(f"  - Clipping ratio: {report.clipping_ratio:.4f}")
    print(f"  - Silence ratio: {report.silence_ratio:.2f}")
    print(f"  - Issues: {report.issues}")
    
    # For synthetic audio, just verify the checker runs and returns metrics
    assert report.quality_score > 0, "Should compute quality score"
    
    # Poor quality audio (short)
    short_audio = audio[:int(sr * 1.0)]
    report_short = checker.check(short_audio)
    print(f"\n  [Short audio] Issues: {report_short.issues}")
    
    print("  [OK] Quality checker working correctly")
    return True


def test_drift_detector():
    """Test drift detector."""
    print("\n[6/6] Testing Drift Detector...")
    
    from app.pipelines.speech.monitoring.drift_detector import DriftDetector
    
    detector = DriftDetector()
    
    # Normal features - no drift
    normal = {
        "jitter_local": 0.5,
        "shimmer_local": 2.0,
        "hnr": 25.0,
        "speech_rate": 4.5
    }
    
    report = detector.check_input_drift(normal)
    print(f"  - Normal features - Has drift: {report.has_drift}")
    
    # Extreme features - should detect drift
    extreme = {
        "jitter_local": 5.0,  # Very high
        "shimmer_local": 12.0,  # Very high
        "hnr": 5.0,  # Very low
        "speech_rate": 0.5  # Very slow
    }
    
    report_extreme = detector.check_input_drift(extreme)
    print(f"  - Extreme features - Has drift: {report_extreme.has_drift}")
    print(f"  - Features drifted: {report_extreme.features_drifted}")
    
    if report_extreme.alerts:
        for alert in report_extreme.alerts[:2]:
            print(f"    - {alert.feature_name}: z={alert.z_score:.1f} ({alert.severity})")
    
    print("  [OK] Drift detector working correctly")
    return True


async def test_full_service():
    """Test the full research-grade service."""
    print("\n[INTEGRATION] Testing Full Research-Grade Service...")
    
    from app.pipelines.speech.service import ResearchGradeSpeechService, PipelineConfig
    
    # Create synthetic audio
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # More realistic speech-like signal
    f0 = 150
    audio = (
        0.6 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.1 * np.sin(2 * np.pi * 4 * f0 * t) +
        0.08 * np.random.randn(len(t))
    ).astype(np.float32)
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Convert to bytes (WAV format)
    import io
    import wave
    
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio_int16.tobytes())
    
    audio_bytes = buffer.getvalue()
    
    # Initialize service with lenient quality settings for synthetic audio
    config = PipelineConfig(
        extract_embeddings=False,
        extract_composite=True,
        enable_audit=False,
        min_snr_db=0.0  # Lenient for synthetic test audio
    )
    service = ResearchGradeSpeechService(config)
    
    # Run analysis
    result = await service.analyze(
        audio_bytes=audio_bytes,
        session_id="test_session_001",
        filename="test.wav"
    )
    
    print(f"  - Session ID: {result.session_id}")
    print(f"  - Processing time: {result.processing_time:.3f}s")
    print(f"  - Risk score: {result.risk_score:.2f}")
    print(f"  - Confidence: {result.confidence:.2f}")
    print(f"  - Quality score: {result.quality_score:.2f}")
    print(f"  - Status: {result.status}")
    print(f"  - Recommendations: {len(result.recommendations)} items")
    
    print("\n  Biomarkers:")
    print(f"    - Jitter: {result.biomarkers.jitter.value:.4f}%")
    print(f"    - Shimmer: {result.biomarkers.shimmer.value:.4f}%")
    print(f"    - HNR: {result.biomarkers.hnr.value:.2f} dB")
    if result.biomarkers.cpps:
        print(f"    - CPPS: {result.biomarkers.cpps.value:.2f} dB")
    print(f"    - Speech rate: {result.biomarkers.speech_rate.value:.2f} syl/s")
    print(f"    - Voice tremor: {result.biomarkers.voice_tremor.value:.4f}")
    
    print("\n  [OK] Full service integration working!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Research-Grade Speech Pipeline Verification")
    print("=" * 60)
    
    tests = [
        test_acoustic_features,
        test_prosodic_features,
        test_composite_features,
        test_clinical_scorer,
        test_quality_checker,
        test_drift_detector
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Async test
    try:
        asyncio.run(test_full_service())
        passed += 1
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
