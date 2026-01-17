/**
 * Retinal Analysis Components
 * 
 * Reusable UI components for retinal analysis pipeline.
 * 
 * @module components/retinal
 */

// Core Components
export { ImageUploadCard } from './ImageUploadCard';
export { RiskGauge } from './RiskGauge';
export { BiomarkerCard } from './BiomarkerCard';
export { RetinalResultsCard } from './RetinalResultsCard';

// Re-export types for convenience
export type {
  RiskCategory,
  RetinalAnalysisResult,
  ImageValidationResult,
  VesselBiomarkers,
  OpticDiscBiomarkers,
  MacularBiomarkers,
  AmyloidBetaIndicators,
  RetinalBiomarkers,
  RiskAssessment,
} from '@/types/retinal-analysis';
