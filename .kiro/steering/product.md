---
inclusion: always
---

# NeuraLens Product Guidelines

NeuraLens is a multi-modal AI-powered neurological assessment platform for early detection and comprehensive analysis of neurological conditions.

## Product Context

**Mission**: Democratize advanced neurological health screening through AI technology
**Target Users**: Healthcare professionals, clinicians, and medical researchers
**Core Value**: Early detection with 18+ months advantage over traditional methods

## Assessment Modalities

When implementing or modifying assessment features, follow these modality-specific guidelines:

### Speech Analysis (`/speech`)
- **Purpose**: Parkinson's disease screening via voice biomarkers
- **Accuracy Target**: 95.2% clinical validation
- **Processing**: Real-time audio analysis with <200ms response
- **File Formats**: Support WAV, MP3, M4A audio inputs
- **UI Pattern**: Record → Process → Results with visual feedback

### Retinal Analysis (`/retinal`) 
- **Purpose**: Non-invasive Alzheimer's detection through retinal imaging
- **Accuracy Target**: 89.3% clinical validation
- **Processing**: Image analysis with fundus photography
- **File Formats**: Support JPEG, PNG, DICOM medical images
- **UI Pattern**: Upload → Analyze → Clinical Report

### Motor Assessment (`/motor`)
- **Purpose**: Movement pattern analysis and tremor detection
- **Accuracy Target**: 93.7% correlation to clinical UPDRS scores
- **Processing**: Motion capture via device sensors or video
- **Data Types**: Accelerometer, gyroscope, video analysis
- **UI Pattern**: Guided Tasks → Motion Capture → Analysis

### Cognitive Testing (`/cognitive`)
- **Purpose**: Memory and executive function assessment
- **Accuracy Target**: 91.4% MCI detection accuracy
- **Processing**: Interactive cognitive tasks with timing analysis
- **Data Types**: Response times, accuracy scores, behavioral patterns
- **UI Pattern**: Test Battery → Performance Tracking → Cognitive Profile

## NRI Fusion Algorithm

When working with multi-modal data integration:
- **Combine**: All assessment modalities into unified risk score
- **Weight**: Each modality based on clinical evidence and data quality
- **Output**: Comprehensive neurological risk index (0-100 scale)
- **Validation**: Cross-reference with clinical standards

## Clinical Standards

### Data Handling
- **HIPAA Compliance**: All patient data must be encrypted and anonymized
- **Audit Trail**: Log all assessment actions and data access
- **Retention**: Follow healthcare data retention policies
- **Export**: Support clinical report formats (PDF, HL7 FHIR)

### Performance Requirements
- **Response Time**: <200ms for real-time processing
- **Availability**: 99.9% uptime for clinical workflows
- **Scalability**: Support concurrent assessments
- **Accuracy**: Maintain clinical validation standards

## User Experience Patterns

### Assessment Workflow
1. **Welcome/Consent**: Clear explanation of assessment process
2. **Modality Selection**: Allow individual or comprehensive assessment
3. **Data Collection**: Guided capture with progress indicators
4. **Processing**: Real-time feedback with loading states
5. **Results**: Clinical-grade reports with actionable insights
6. **Export/Share**: Healthcare provider integration options

### Accessibility Requirements
- **WCAG 2.1 AA**: Full compliance for healthcare accessibility
- **Screen Readers**: Semantic HTML and ARIA labels
- **Keyboard Navigation**: Complete keyboard accessibility
- **Visual Impairments**: High contrast, scalable text
- **Motor Impairments**: Large touch targets, alternative inputs

## Development Priorities

### Feature Development Order
1. **Core Assessment**: Implement individual modality assessments
2. **NRI Fusion**: Multi-modal risk calculation
3. **Clinical Reports**: Healthcare provider documentation
4. **Integration**: EMR/EHR system connectivity
5. **Analytics**: Population health insights

### Quality Gates
- **Clinical Validation**: All features must meet accuracy targets
- **Security Review**: Healthcare compliance verification
- **Performance Testing**: Sub-200ms response validation
- **Accessibility Audit**: WCAG 2.1 AA compliance check