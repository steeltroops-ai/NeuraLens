// NeuroLens-X ML Module
// Unified exports for all machine learning components

// Core ML Models
export { speechAnalyzer, type SpeechAnalysisResult, type SpeechFeatures } from './speech-analysis';
export {
  retinalAnalyzer,
  type RetinalAnalysisResult,
  type RetinalFeatures,
} from './retinal-analysis';
export {
  riskAssessmentCalculator,
  type RiskAssessmentResult,
  type RiskAssessmentData,
} from './risk-assessment';
export { nriFusionCalculator, type NRIFusionResult, type ModalityResult } from './nri-fusion';

// ML Integration
export {
  mlModelIntegrator,
  generateSessionId,
  validateAssessmentRequest,
  type AssessmentRequest,
  type AssessmentProgress,
  type CompleteAssessmentResult,
  type ProgressCallback,
} from './ml-integration';

// Type exports for external use
export type {
  DemographicData,
  MedicalHistory,
  FamilyHistory,
  LifestyleFactors,
  CognitiveAssessment,
} from './risk-assessment';

export type { FusionWeights } from './nri-fusion';

// Utility functions
export const ML_VERSION = '1.0.0';

export const getMLCapabilities = () => ({
  speechAnalysis: {
    supported: true,
    features: ['voice biomarkers', 'tremor detection', 'speech rate analysis'],
    accuracy: 0.85,
  },
  retinalAnalysis: {
    supported: true,
    features: ['vascular pattern analysis', 'optic disc assessment', 'pathology detection'],
    accuracy: 0.88,
  },
  riskAssessment: {
    supported: true,
    features: ['demographic factors', 'medical history', 'lifestyle analysis'],
    accuracy: 0.92,
  },
  nriFusion: {
    supported: true,
    features: ['multi-modal fusion', 'uncertainty quantification', 'confidence intervals'],
    accuracy: 0.9,
  },
});

export const getProcessingEstimates = () => ({
  speech: { min: 5, max: 15, average: 8 }, // seconds
  retinal: { min: 3, max: 10, average: 6 }, // seconds
  risk: { min: 1, max: 3, average: 2 }, // seconds
  fusion: { min: 1, max: 2, average: 1 }, // seconds
  total: { min: 10, max: 30, average: 17 }, // seconds
});

export const getRiskCategories = () => ({
  low: { range: [0, 25], description: 'Low neurological risk', color: '#10B981' },
  moderate: { range: [26, 50], description: 'Moderate neurological risk', color: '#F59E0B' },
  high: { range: [51, 75], description: 'High neurological risk', color: '#F97316' },
  critical: { range: [76, 100], description: 'Critical neurological risk', color: '#EF4444' },
});

export const getModalityDescriptions = () => ({
  speech: {
    name: 'Speech Analysis',
    description: 'Voice biomarker detection through advanced speech pattern analysis',
    biomarkers: ['speech rate', 'pause patterns', 'voice tremor', 'articulation'],
    clinicalRelevance: 'Early detection of neurological changes affecting speech production',
  },
  retinal: {
    name: 'Retinal Imaging',
    description: 'Vascular pattern analysis for early pathological changes',
    biomarkers: ['vessel density', 'tortuosity', 'optic disc changes', 'RNFL thickness'],
    clinicalRelevance: 'Retinal changes reflect brain health and neurodegeneration',
  },
  risk: {
    name: 'Risk Assessment',
    description: 'Comprehensive health and lifestyle risk factor analysis',
    biomarkers: ['demographics', 'medical history', 'family history', 'lifestyle'],
    clinicalRelevance: 'Established risk factors for neurological disorders',
  },
  motor: {
    name: 'Motor Assessment',
    description: 'Physical movement and coordination analysis',
    biomarkers: ['tremor', 'bradykinesia', 'rigidity', 'gait patterns'],
    clinicalRelevance: 'Motor symptoms often precede cognitive symptoms',
  },
});
