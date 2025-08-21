# NeuroLens Demo Dataset - Judge Evaluation

## Overview
This dataset contains 12 synthetic patients with realistic neurological risk profiles for evaluating the NeuroLens multi-modal assessment system.

## Quick Start for Judges
1. **Low Risk Patient**: DEMO_001 - Healthy baseline
2. **High Risk Patient**: DEMO_003 - Significant neurological indicators
3. **Audio Files**: Located in `audio/` directory
4. **Analysis Results**: Located in `results/` directory

## Dataset Statistics
- Total Patients: 12
- Risk Distribution: {'low': 3, 'moderate': 3, 'high': 3, 'very_high': 3}
- Average Processing Time: <1 second per assessment

## Files Structure
- `patients/patient_profiles.json` - Patient demographics and medical history
- `audio/` - Speech audio files (WAV format)
- `results/speech_results.json` - ML analysis results
- `JUDGE_EVALUATION_SUMMARY.json` - Detailed statistics

Generated: 2025-08-22 01:49:51
