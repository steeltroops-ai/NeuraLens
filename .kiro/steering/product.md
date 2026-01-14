---
inclusion: always
---

---
inclusion: always
---

# MediLens Product Guidelines

MediLens is a centralized web platform providing AI-powered neurological diagnostic tools. The platform focuses on early detection of neurodegenerative conditions through multi-modal assessment.

## Product Identity

**Mission**: Democratize advanced neurological diagnostics through AI technology, enabling early detection and intervention for neurodegenerative conditions.

**Target Users**: 
- Primary: Healthcare professionals (neurologists, primary care physicians, specialists)
- Secondary: Medical researchers and clinical trial coordinators
- Tertiary: Patients (with healthcare provider guidance and supervision)

**Core Value Proposition**: Unified platform for multi-modal neurological risk assessment with validated AI models and clinical-grade reporting.

## Assessment Modalities

MediLens currently implements four core assessment modalities. When implementing or modifying features, follow these specifications:

### 1. Speech Analysis
**Route**: `/dashboard/assessments/speech`
**Purpose**: Parkinson's disease and neurological speech disorder screening via voice biomarkers
**Clinical Target**: 95.2% sensitivity, 93.8% specificity for PD detection
**Processing Time**: <200ms for real-time analysis
**Input Requirements**:
- Audio formats: WAV (preferred), MP3, M4A
- Sample rate: 16kHz minimum, 44.1kHz recommended
- Duration: 10-30 seconds sustained phonation
- Tasks: Sustained vowel /a/, /pa-ta-ka/ repetition, reading passage
**Biomarkers Extracted**:
- Jitter (frequency perturbation)
- Shimmer (amplitude perturbation)
- HNR (Harmonics-to-Noise Ratio)
- Pitch variability
- Speech rate and pauses
**UI Flow**: Microphone Check → Task Instructions → Recording → Real-time Processing → Results with Biomarker Breakdown
**Error Handling**: Validate audio quality, detect background noise, require minimum SNR

### 2. Retinal Analysis
**Route**: `/dashboard/assessments/retinal`
**Purpose**: Non-invasive Alzheimer's disease detection through retinal imaging and vascular analysis
**Clinical Target**: 89.3% accuracy for early AD detection
**Processing Time**: <500ms for image analysis
**Input Requirements**:
- Image formats: JPEG, PNG, DICOM
- Resolution: 1024x1024 minimum
- Image type: Fundus photography (color retinal images)
- Quality: Clear optic disc and macula visibility
**Biomarkers Extracted**:
- Retinal vessel density and tortuosity
- Optic disc parameters
- Macular thickness estimation
- Vascular branching patterns
- Amyloid-beta indicators
**UI Flow**: Upload/Capture → Image Quality Check → Analysis → Annotated Results with Heatmap
**Error Handling**: Validate image quality, detect poor focus/lighting, require retinal landmarks

### 3. Motor Assessment
**Route**: `/dashboard/assessments/motor`
**Purpose**: Movement pattern analysis, tremor detection, and bradykinesia assessment
**Clinical Target**: 93.7% correlation to clinical UPDRS motor scores
**Processing Time**: <300ms per movement sequence
**Input Requirements**:
- Data sources: Device accelerometer/gyroscope OR video upload
- Video formats: MP4, MOV, WebM
- Duration: 30-60 seconds per task
- Tasks: Finger tapping, hand rotation, gait analysis
**Biomarkers Extracted**:
- Tremor frequency and amplitude
- Movement speed and rhythm
- Range of motion
- Coordination metrics
- Gait parameters (stride length, cadence)
**UI Flow**: Task Selection → Guided Instructions → Motion Capture → Analysis → Movement Visualization
**Error Handling**: Validate sensor data quality, detect insufficient movement, require task completion

### 4. Cognitive Testing
**Route**: `/dashboard/assessments/cognitive`
**Purpose**: Memory, executive function, and cognitive decline assessment
**Clinical Target**: 91.4% accuracy for MCI detection
**Processing Time**: Real-time scoring during test battery
**Input Requirements**:
- Interactive web-based tasks
- Response time tracking (millisecond precision)
- Accuracy scoring
- Test battery duration: 15-20 minutes
**Tests Included**:
- Immediate and delayed recall
- Trail Making Test (digital version)
- Digit span (forward/backward)
- Semantic fluency
- Visual pattern recognition
**Biomarkers Extracted**:
- Response time patterns
- Accuracy scores by domain
- Learning curve analysis
- Attention metrics
- Executive function composite
**UI Flow**: Test Instructions → Timed Tasks → Performance Tracking → Cognitive Profile Report
**Error Handling**: Detect incomplete tests, validate response timing, handle interruptions

## NRI Fusion Algorithm

The Neurological Risk Index (NRI) combines all assessment modalities into a unified risk score.

**Implementation Requirements**:
- **Input**: Results from 2+ completed assessments (minimum)
- **Weighting**: Dynamic weights based on data quality and clinical evidence
  - Speech: 25% base weight
  - Retinal: 30% base weight
  - Motor: 25% base weight
  - Cognitive: 20% base weight
- **Output**: NRI score (0-100 scale) with risk category
  - 0-25: Minimal risk (green)
  - 26-40: Low risk (lime)
  - 41-55: Moderate risk (yellow)
  - 56-70: Elevated risk (orange)
  - 71-85: High risk (red)
  - 86-100: Critical risk (dark red)
- **Confidence Interval**: Display 95% CI for NRI score
- **Validation**: Cross-reference with age-matched normative data

**API Endpoint**: `POST /api/v1/nri/calculate`
**Frontend Component**: `components/dashboard/NRIScoreCard.tsx`

## Clinical Standards & Compliance

### Data Security (HIPAA Compliance)
**Required for all features**:
- Encrypt all patient data at rest (AES-256) and in transit (TLS 1.3)
- Anonymize data in logs, analytics, and error reports
- Implement role-based access control (RBAC)
- Audit trail for all data access and modifications
- Secure file upload with virus scanning
- Session timeout after 15 minutes of inactivity

### Performance Requirements
**Non-negotiable targets**:
- API response time: <200ms for real-time processing
- ML inference time: <500ms per modality
- Database queries: <100ms for standard operations
- Page load time: <2 seconds (initial), <500ms (navigation)
- Uptime: 99.9% availability for clinical workflows
- Concurrent users: Support 100+ simultaneous assessments

### Clinical Validation Standards
**Before deploying any diagnostic feature**:
- Validate against clinical gold standard (e.g., UPDRS, MoCA, clinical diagnosis)
- Document sensitivity, specificity, PPV, NPV
- Test on diverse patient populations (age, ethnicity, disease stage)
- Peer review by clinical advisors
- Maintain accuracy targets specified per modality

### Data Export & Interoperability
**Required formats**:
- PDF clinical reports (patient-friendly and provider-detailed versions)
- HL7 FHIR resources for EMR integration
- CSV/JSON for research data export
- DICOM for imaging data (retinal analysis)

## User Experience Patterns

### Assessment Workflow (Standard Pattern)
All assessment modalities follow this consistent flow:

1. **Pre-Assessment**
   - Display clear instructions and consent information
   - Show estimated time to complete
   - Explain data usage and privacy
   - Technical requirements check (microphone, camera, etc.)

2. **Data Collection**
   - Guided step-by-step instructions with visual aids
   - Real-time feedback on data quality
   - Progress indicators (step X of Y)
   - Ability to pause/resume (where applicable)
   - Clear error messages with recovery actions

3. **Processing**
   - Loading state with progress indication
   - Estimated time remaining
   - "Processing..." message with modality-specific context
   - No page navigation during processing

4. **Results Display**
   - Risk score prominently displayed with color coding
   - Biomarker breakdown with reference ranges
   - Visual representations (charts, heatmaps, graphs)
   - Clinical interpretation in plain language
   - Comparison to previous assessments (if available)
   - Recommendations for next steps

5. **Post-Assessment Actions**
   - Download PDF report
   - Share with healthcare provider
   - Schedule follow-up assessment
   - View historical trends

### Dashboard Organization
**Route Structure**:
- `/dashboard` - Overview with recent assessments and NRI trend
- `/dashboard/assessments` - Assessment hub with modality selection
- `/dashboard/assessments/[modality]` - Individual assessment pages
- `/dashboard/results` - Historical results and trends
- `/dashboard/reports` - Generated clinical reports
- `/dashboard/settings` - User preferences and profile

**Navigation Patterns**:
- Persistent sidebar with assessment quick access
- Breadcrumb navigation for context
- Command palette (Cmd+K) for power users
- Mobile-responsive with collapsible sidebar

### Error Handling Patterns
**User-Facing Errors**:
- Display clear, actionable error messages
- Provide recovery steps (e.g., "Try uploading a different image")
- Offer alternative actions (e.g., "Record again" vs "Upload file")
- Log errors for debugging without exposing technical details

**System Errors**:
- Graceful degradation (show cached data if API fails)
- Retry logic with exponential backoff
- Fallback to alternative processing methods
- Contact support option for persistent issues

## Development Guidelines

### Feature Implementation Checklist
When adding new features or modifying existing ones:

- [ ] **Clinical Validation**: Verify accuracy targets are met
- [ ] **Performance**: Measure and optimize response times
- [ ] **Security**: Implement encryption and access controls
- [ ] **Accessibility**: Test with screen readers and keyboard navigation
- [ ] **Responsive Design**: Test on mobile, tablet, desktop
- [ ] **Error Handling**: Cover edge cases and failure modes
- [ ] **Documentation**: Update API docs and user guides
- [ ] **Testing**: Unit tests, integration tests, E2E tests
- [ ] **Audit Logging**: Log all data access and modifications
- [ ] **Privacy**: Ensure HIPAA compliance and data anonymization

### Quality Gates (Pre-Deployment)
**All features must pass**:
1. **Clinical Validation**: Meet or exceed accuracy targets
2. **Security Review**: Pass HIPAA compliance audit
3. **Performance Testing**: Meet sub-200ms response time
4. **Accessibility Audit**: WCAG 2.1 AA compliance verified
5. **User Testing**: Validated with target users (clinicians)
6. **Code Review**: Approved by 2+ team members
7. **Documentation**: Complete API docs and user guides

### Prohibited Patterns
**Never implement**:
- ❌ Storing unencrypted patient data
- ❌ Displaying raw ML model outputs without clinical interpretation
- ❌ Auto-diagnosing without disclaimer and provider review
- ❌ Sharing patient data without explicit consent
- ❌ Bypassing authentication for "convenience"
- ❌ Hardcoding API keys or credentials
- ❌ Ignoring accessibility requirements
- ❌ Deploying without clinical validation

## Terminology & Messaging

### User-Facing Language
**Use clinical but accessible language**:
- ✅ "Neurological Risk Index" (not "AI score")
- ✅ "Assessment" (not "test" or "diagnosis")
- ✅ "Biomarker" (not "feature" or "metric")
- ✅ "Risk level" (not "probability" or "prediction")
- ✅ "Healthcare provider" (not "doctor" or "physician" exclusively)

### Disclaimers (Required)
**Display on all result pages**:
> "This assessment is a screening tool and does not constitute a medical diagnosis. Results should be reviewed by a qualified healthcare provider. If you have concerns about your health, please consult a medical professional."

### Risk Communication
**Use consistent risk categories**:
- Minimal Risk (0-25): "Results within normal range"
- Low Risk (26-40): "Slightly elevated, monitor over time"
- Moderate Risk (41-55): "Moderate elevation, discuss with provider"
- Elevated Risk (56-70): "Elevated risk, recommend clinical evaluation"
- High Risk (71-85): "High risk, clinical evaluation recommended"
- Critical Risk (86-100): "Very high risk, urgent clinical evaluation recommended"

## Future Roadmap Context

### Planned Expansions
When designing features, consider future integration with:
- EMR/EHR systems (Epic, Cerner, Meditech)
- Telemedicine platforms for remote assessments
- Wearable device integration (continuous monitoring)
- Population health analytics dashboard
- Clinical trial recruitment tools
- Multi-language support (Spanish, Mandarin, French)

### Scalability Considerations
- Design for 10,000+ concurrent users
- Support multi-tenant architecture for healthcare organizations
- Enable white-label deployment for partners
- Plan for real-time collaboration features (provider-patient)
- Consider offline mode for low-connectivity environments