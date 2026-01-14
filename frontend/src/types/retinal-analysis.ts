
export interface VesselBiomarkers {
  density_percentage: number;
  tortuosity_index: number;
  avr_ratio: number;
  branching_coefficient: number;
  confidence: number;
}

export interface OpticDiscBiomarkers {
  cup_to_disc_ratio: number;
  disc_area_mm2: number;
  rim_area_mm2: number;
  confidence: number;
}

export interface MacularBiomarkers {
  thickness_um: number;
  volume_mm3: number;
  confidence: number;
}

export interface AmyloidBetaIndicators {
  presence_score: number;
  distribution_pattern: string;
  confidence: number;
}

export interface RetinalBiomarkers {
  vessels: VesselBiomarkers;
  optic_disc: OpticDiscBiomarkers;
  macula: MacularBiomarkers;
  amyloid_beta: AmyloidBetaIndicators;
}

export interface RiskAssessment {
  risk_score: number;
  risk_category: 'minimal' | 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
  confidence_interval: [number, number];
  contributing_factors: Record<string, number>;
}

export interface RetinalAnalysisResult {
  assessment_id: string;
  patient_id: string;
  biomarkers: RetinalBiomarkers;
  risk_assessment: RiskAssessment;
  quality_score: number;
  heatmap_url: string;
  segmentation_url: string;
  created_at: string;
  model_version: string;
  processing_time_ms: number;
}

export interface ImageValidationResult {
  is_valid: boolean;
  quality_score: number;
  issues: string[];
  recommendations: string[];
  snr_db: number;
  has_optic_disc: boolean;
  has_macula: boolean;
}
