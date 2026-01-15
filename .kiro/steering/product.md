---
inclusion: always
---

# MediLens Product Guidelines

## Product Vision & Scope

MediLens is a unified AI-powered medical diagnostic platform providing healthcare professionals with multiple specialized diagnostic tools in one centralized interface. Each diagnostic module is clinically validated, HIPAA-compliant, and designed for real-world clinical workflows.

## Core Product Principles

### 1. Clinical Accuracy First
- All diagnostic modules must meet minimum accuracy thresholds before deployment
- Display confidence scores with all predictions
- Never hide uncertainty - show confidence intervals and limitations
- Provide clinical context and references for AI recommendations

### 2. Healthcare Professional-Centric Design
- Optimize for clinical workflows, not consumer patterns
- Minimize clicks to complete diagnostic tasks
- Support rapid assessment review and decision-making
- Enable efficient batch processing for high-volume clinics

### 3. Patient Safety & Compliance
- HIPAA compliance is non-negotiable for all features
- Implement audit trails for all diagnostic actions
- Anonymize data in logs, analytics, and error reports
- Clear disclaimers: AI assists, does not replace clinical judgment
- Emergency alerts for critical findings (e.g., acute stroke, MI)

### 4. Transparency & Explainability
- Show AI reasoning with visual explanations (heatmaps, attention maps)
- Provide feature importance for predictions
- Link to relevant medical literature and guidelines
- Display model version, training data characteristics, and known limitations

## Diagnostic Module Standards

### Required Components (Every Module)

**Input Validation:**
- File type, size, and quality checks before processing
- Clear error messages for invalid inputs
- Guidance on optimal image/data capture techniques

**Processing Pipeline:**
- Real-time progress indicators (<500ms ML inference target)
- Graceful degradation if processing fails
- Retry mechanisms for transient failures

**Results Display:**
- Primary diagnosis with confidence score
- Differential diagnoses ranked by probability
- Visual annotations (bounding boxes, segmentation masks, heatmaps)
- Severity classification (e.g., mild/moderate/severe)
- Recommended next steps and referral guidance

**Clinical Integration:**
- Export to PDF with clinical report format
- HL7 FHIR-compatible data structures
- Integration hooks for EMR/EHR systems
- Shareable links for second opinions

### Module-Specific Conventions

**Imaging Modules (Retinal, Radiology, Dermatology, Pathology):**
- Support DICOM format where applicable
- Side-by-side comparison view for longitudinal tracking
- Zoom, pan, and measurement tools
- Annotation capabilities for clinician notes

**Signal Processing Modules (Cardiology ECG, Speech Analysis):**
- Waveform visualization with interactive timeline
- Segment-level analysis (e.g., per heartbeat, per utterance)
- Playback controls for audio/video data
- Export raw feature data for research use

**Multi-Modal Modules (NRI Fusion, OmniMed):**
- Clear indication of which data sources contributed to prediction
- Weighted contribution visualization
- Ability to recompute with subset of modalities
- Cross-modal consistency checks

## User Experience Patterns

### Assessment Workflow (Standard Flow)

1. **Module Selection**: Clear cards with specialty icons and descriptions
2. **Data Upload**: Drag-and-drop or file picker with format guidance
3. **Quality Check**: Automated validation with actionable feedback
4. **Processing**: Progress indicator with estimated time remaining
5. **Results Review**: Structured report with visual aids
6. **Action**: Save, export, share, or start new assessment

### Dashboard Organization

**Primary Navigation:**
- Dashboard Home (overview, recent activity)
- Assessment Modules (by specialty)
- Results History (searchable, filterable)
- Reports (generated clinical reports)
- Settings (user preferences, integrations)

**Quick Actions:**
- "New Assessment" button always visible
- Recent assessments for quick re-access
- Saved templates for common workflows

### Results Presentation

**Risk Stratification Colors:**
- Green (#10b981): Low risk, normal findings
- Yellow (#f59e0b): Moderate risk, monitor closely
- Orange (#f97316): Elevated risk, follow-up recommended
- Red (#ef4444): High risk, urgent referral needed
- Gray (#6b7280): Inconclusive, repeat assessment

**Confidence Indicators:**
- High confidence (>90%): Solid color badge
- Medium confidence (70-90%): Outlined badge with note
- Low confidence (<70%): Dashed outline with prominent disclaimer

## Data Handling & Privacy

### Patient Data Management

**Data Minimization:**
- Collect only data necessary for diagnostic purpose
- Avoid storing unnecessary metadata
- Automatic data retention policies (configurable by organization)

**De-identification:**
- Strip EXIF data from uploaded images
- Remove patient identifiers from logs
- Anonymize data used for model training/improvement

**Access Control:**
- Role-based permissions (patient, provider, admin, researcher)
- Audit logs for all data access
- Session timeout after 15 minutes of inactivity
- Multi-factor authentication for sensitive operations

### Data Security

**Encryption:**
- TLS 1.3 for data in transit
- AES-256 for data at rest
- Encrypted backups with separate key management

**Compliance:**
- HIPAA compliance for US deployments
- GDPR compliance for EU users
- SOC 2 Type II certification target
- Regular security audits and penetration testing

## Performance Requirements

### Response Time Targets

- **Page Load**: <2s initial, <500ms navigation
- **API Response**: <200ms for standard queries
- **ML Inference**: <500ms per modality
- **Image Upload**: Support up to 50MB files
- **Batch Processing**: Handle 100+ assessments concurrently

### Availability & Reliability

- **Uptime**: 99.9% SLA (excluding planned maintenance)
- **Error Rate**: <0.1% for API requests
- **Data Loss**: Zero tolerance - all assessments must be recoverable
- **Backup**: Automated daily backups with 30-day retention

## Clinical Validation Standards

### Model Performance Metrics

**Classification Tasks:**
- Sensitivity (recall): >90% for critical conditions
- Specificity: >85% to minimize false positives
- AUC-ROC: >0.90 for binary classification
- F1 Score: >0.85 for multi-class problems

**Detection Tasks:**
- mAP (mean Average Precision): >0.80
- IoU (Intersection over Union): >0.70 for segmentation
- False negative rate: <5% for critical findings

**Validation Requirements:**
- External validation on held-out datasets
- Multi-site validation across diverse populations
- Subgroup analysis (age, sex, ethnicity)
- Comparison to clinical gold standard (expert consensus)

### Clinical Trial Integration

- Support for prospective clinical studies
- Randomization and blinding capabilities
- Structured data export for statistical analysis
- IRB-compliant consent workflows

## Accessibility & Internationalization

### Accessibility (WCAG 2.1 AA Compliance)

- Keyboard navigation for all interactive elements
- Screen reader compatibility
- High contrast mode support
- Adjustable font sizes
- Alt text for all diagnostic images
- Captions for audio/video content

### Internationalization

- Multi-language support (English, Spanish, Mandarin priority)
- Locale-specific date/time formats
- Right-to-left (RTL) language support
- Cultural sensitivity in medical terminology
- Region-specific clinical guidelines

## Error Handling & User Feedback

### Error Categories

**User Errors (4xx):**
- Clear, actionable error messages
- Guidance on how to fix the issue
- Examples of valid inputs
- Link to help documentation

**System Errors (5xx):**
- Generic message to user (avoid technical details)
- Automatic error reporting to monitoring system
- Retry mechanism with exponential backoff
- Fallback to cached data when appropriate

### User Feedback Mechanisms

- In-app feedback button on every page
- Bug report form with screenshot capture
- Feature request submission
- Clinical accuracy feedback (false positive/negative reporting)
- Net Promoter Score (NPS) surveys (quarterly)

## Feature Flags & Gradual Rollout

### Feature Development Stages

1. **Alpha**: Internal testing only, hidden from production users
2. **Beta**: Opt-in for select users, prominent beta badge
3. **Limited Release**: Gradual rollout to percentage of users
4. **General Availability**: Available to all users
5. **Deprecated**: Warning banner, migration path provided

### A/B Testing Guidelines

- Test one variable at a time
- Minimum sample size: 1000 users per variant
- Statistical significance: p < 0.05
- Monitor key metrics: task completion rate, time on task, error rate
- Clinical safety checks before declaring winner

## Documentation Requirements

### User Documentation

- Getting started guide for each module
- Video tutorials for common workflows
- FAQ section with searchable content
- Clinical use case examples
- Troubleshooting guides

### Clinical Documentation

- Model cards for each AI module (dataset, architecture, performance)
- Clinical validation study results
- Known limitations and contraindications
- Intended use statements
- Regulatory clearance status (FDA, CE Mark)

### Developer Documentation

- API reference with interactive examples
- Integration guides for EMR systems
- Webhook documentation for real-time updates
- SDK/client libraries (Python, JavaScript)
- Changelog with breaking changes highlighted

## Quality Assurance Checklist

Before deploying any new feature or module:

- [ ] Clinical accuracy meets minimum thresholds
- [ ] HIPAA compliance audit passed
- [ ] Accessibility audit passed (WCAG 2.1 AA)
- [ ] Performance benchmarks met (<200ms API, <500ms ML)
- [ ] Security review completed
- [ ] User acceptance testing with healthcare professionals
- [ ] Documentation updated (user + clinical + developer)
- [ ] Error handling tested (edge cases, network failures)
- [ ] Cross-browser testing (Chrome, Firefox, Safari, Edge)
- [ ] Mobile responsiveness verified
- [ ] Monitoring and alerting configured
- [ ] Rollback plan documented

## Metrics & Success Criteria

### Product Metrics

**Engagement:**
- Daily Active Users (DAU)
- Assessments per user per month
- Module adoption rate
- Feature utilization rate

**Quality:**
- Clinical accuracy feedback score
- False positive/negative reports
- User-reported bugs per release
- System uptime percentage

**Business:**
- User retention rate (90-day)
- Net Promoter Score (NPS)
- Time to complete assessment (efficiency)
- EMR integration adoption rate

### Clinical Impact Metrics

- Early detection rate for critical conditions
- Time to diagnosis reduction
- Referral appropriateness rate
- Patient outcomes (where trackable)
- Cost savings per assessment

## Ethical Considerations

### AI Ethics Principles

**Fairness:**
- Test for bias across demographic groups
- Ensure equitable performance across populations
- Avoid perpetuating healthcare disparities
- Regular bias audits with external review

**Transparency:**
- Disclose AI involvement in diagnostic process
- Explain model limitations clearly
- Provide opt-out mechanisms where appropriate
- Open about data usage and model training

**Accountability:**
- Clear responsibility chain for AI decisions
- Human oversight for critical decisions
- Mechanism to contest AI recommendations
- Regular ethics board review

### Responsible AI Development

- Diverse training data representing global populations
- Continuous monitoring for model drift
- Regular retraining with updated clinical guidelines
- Stakeholder input (patients, clinicians, ethicists)
- Transparent reporting of model failures

## Competitive Differentiation

### Unique Value Propositions

1. **Unified Platform**: Multiple specialties in one interface (not single-purpose tools)
2. **Clinical Validation**: Peer-reviewed studies for each module
3. **Explainable AI**: Visual explanations, not black-box predictions
4. **Seamless Integration**: Works with existing EMR/EHR systems
5. **Continuous Learning**: Models improve with usage (with consent)

### Market Positioning

- **Primary**: Hospital systems and large medical groups
- **Secondary**: Independent clinics and telehealth providers
- **Tertiary**: Medical education and research institutions

## Future Roadmap Considerations

### Near-Term (6-12 months)

- Complete initial 3 modules (Retinal, Radiology, Dermatology)
- EMR integration with top 3 vendors (Epic, Cerner, Allscripts)
- Mobile app for point-of-care assessments
- Telemedicine integration (Zoom, Doxy.me)

### Mid-Term (1-2 years)

- Expand to 8+ diagnostic modules
- Multi-modal fusion capabilities (OmniMed AI)
- Population health analytics dashboard
- API marketplace for third-party integrations

### Long-Term (2-5 years)

- Real-time monitoring and alerts (wearable integration)
- Predictive analytics (risk forecasting)
- Treatment recommendation engine
- Global expansion with localized models
