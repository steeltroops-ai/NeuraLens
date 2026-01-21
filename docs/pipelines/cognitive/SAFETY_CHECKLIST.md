# Cognitive Pipeline Safety & Ethics Checklist

## 1. Clinical Disclaimer
- [x] Frontend must clearly state this is **NOT a diagnostic tool**.
- [x] Results must be presented as "Screening Risk" not "Diagnosis".
- [x] "Consult a professional" recommendation must be prominent for high-risk results.

## 2. Data Privacy
- [ ] Ensure `patient_id` is encrypted or tokenized if stored.
- [ ] Session data (keystrokes) should not log PII (e.g. typing names).
- [ ] Compliance with HIPAA/GDPR for health data storage.

## 3. User Safety
- [ ] **Photosensitivity Warning**: Flashing stimuli (N-Back) must carry a warning for epilepsy.
- [ ] **Distraction Free**: Ensure environment check does not cause undue stress.
- [ ] **Fatigue Management**: Tests should not exceed 20 minutes total.

## 4. Algorithmic Fairness
- [ ] Normative baselines should account for Age and Education level (currently simplified).
- [ ] Risk scoring must be validated across diverse demographics.
