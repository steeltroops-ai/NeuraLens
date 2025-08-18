// NeuroLens-X NRI Fusion Algorithm
// Unified Neuro-Risk Index calculation with uncertainty quantification

import type { SpeechAnalysisResult } from './speech-analysis';
import type { RetinalAnalysisResult } from './retinal-analysis';
import type { RiskAssessmentResult } from './risk-assessment';

export interface ModalityResult {
  modality: 'speech' | 'retinal' | 'risk' | 'motor';
  score: number;              // 0-100 risk score
  confidence: number;         // Confidence interval (±%)
  quality: number;            // Data quality score (0-1)
  processingTime: number;     // Processing time (ms)
  findings: string[];         // Key findings
  available: boolean;         // Whether modality data is available
}

export interface NRIFusionResult {
  nriScore: number;           // Unified Neuro-Risk Index (0-100)
  confidence: number;         // Overall confidence interval (±%)
  riskCategory: 'low' | 'moderate' | 'high' | 'critical';
  modalityContributions: {
    speech: number;           // Contribution weight (0-1)
    retinal: number;
    risk: number;
    motor: number;
  };
  uncertaintyFactors: string[];  // Sources of uncertainty
  recommendations: string[];     // Clinical recommendations
  processingTime: number;        // Total processing time (ms)
  dataCompleteness: number;      // Percentage of available modalities
  clinicalNotes: string[];       // Additional clinical insights
}

export interface FusionWeights {
  speech: number;
  retinal: number;
  risk: number;
  motor: number;
}

export class NRIFusionCalculator {
  // Base weights for each modality (can be adjusted based on evidence)
  private baseWeights: FusionWeights = {
    speech: 0.30,     // Strong early indicator
    retinal: 0.25,    // Objective biomarker
    risk: 0.25,       // Comprehensive risk factors
    motor: 0.20,      // Physical manifestations
  };

  // Quality thresholds for reliable results
  private qualityThresholds = {
    minimum: 0.3,     // Below this, exclude from fusion
    good: 0.7,        // Above this, full weight
    excellent: 0.9,   // Premium quality data
  };

  /**
   * Calculate unified NRI score from multiple modalities
   */
  async calculateNRI(
    speechResult?: SpeechAnalysisResult,
    retinalResult?: RetinalAnalysisResult,
    riskResult?: RiskAssessmentResult,
    motorResult?: any // Motor assessment not implemented yet
  ): Promise<NRIFusionResult> {
    const startTime = performance.now();
    
    try {
      // Prepare modality results
      const modalityResults = this.prepareModalityResults(
        speechResult,
        retinalResult,
        riskResult,
        motorResult
      );
      
      // Calculate adaptive weights based on data quality and availability
      const adaptiveWeights = this.calculateAdaptiveWeights(modalityResults);
      
      // Perform weighted fusion
      const fusedScore = this.performWeightedFusion(modalityResults, adaptiveWeights);
      
      // Apply uncertainty quantification
      const uncertaintyAnalysis = this.quantifyUncertainty(modalityResults, adaptiveWeights);
      
      // Determine risk category
      const riskCategory = this.determineRiskCategory(fusedScore);
      
      // Generate clinical recommendations
      const recommendations = this.generateClinicalRecommendations(
        fusedScore,
        modalityResults,
        riskCategory
      );
      
      // Generate clinical notes
      const clinicalNotes = this.generateClinicalNotes(modalityResults, fusedScore);
      
      // Calculate data completeness
      const dataCompleteness = this.calculateDataCompleteness(modalityResults);
      
      const processingTime = performance.now() - startTime;
      
      return {
        nriScore: Math.round(fusedScore),
        confidence: uncertaintyAnalysis.confidence,
        riskCategory,
        modalityContributions: adaptiveWeights,
        uncertaintyFactors: uncertaintyAnalysis.factors,
        recommendations,
        processingTime,
        dataCompleteness,
        clinicalNotes,
      };
    } catch (error) {
      console.error('NRI fusion failed:', error);
      throw new Error('Failed to calculate NRI: ' + (error as Error).message);
    }
  }

  /**
   * Prepare standardized modality results
   */
  private prepareModalityResults(
    speechResult?: SpeechAnalysisResult,
    retinalResult?: RetinalAnalysisResult,
    riskResult?: RiskAssessmentResult,
    motorResult?: any
  ): ModalityResult[] {
    const results: ModalityResult[] = [];
    
    // Speech modality
    if (speechResult) {
      results.push({
        modality: 'speech',
        score: speechResult.riskScore,
        confidence: speechResult.confidence,
        quality: speechResult.qualityScore,
        processingTime: speechResult.processingTime,
        findings: speechResult.findings,
        available: true,
      });
    } else {
      results.push({
        modality: 'speech',
        score: 0,
        confidence: 0,
        quality: 0,
        processingTime: 0,
        findings: [],
        available: false,
      });
    }
    
    // Retinal modality
    if (retinalResult) {
      results.push({
        modality: 'retinal',
        score: retinalResult.riskScore,
        confidence: retinalResult.confidence,
        quality: retinalResult.imageQuality,
        processingTime: retinalResult.processingTime,
        findings: retinalResult.findings,
        available: true,
      });
    } else {
      results.push({
        modality: 'retinal',
        score: 0,
        confidence: 0,
        quality: 0,
        processingTime: 0,
        findings: [],
        available: false,
      });
    }
    
    // Risk assessment modality
    if (riskResult) {
      results.push({
        modality: 'risk',
        score: riskResult.overallRisk,
        confidence: riskResult.confidence,
        quality: 0.9, // Risk assessment typically high quality
        processingTime: riskResult.processingTime,
        findings: riskResult.modifiableFactors.concat(riskResult.nonModifiableFactors),
        available: true,
      });
    } else {
      results.push({
        modality: 'risk',
        score: 0,
        confidence: 0,
        quality: 0,
        processingTime: 0,
        findings: [],
        available: false,
      });
    }
    
    // Motor modality (placeholder for future implementation)
    results.push({
      modality: 'motor',
      score: 0,
      confidence: 0,
      quality: 0,
      processingTime: 0,
      findings: [],
      available: false,
    });
    
    return results;
  }

  /**
   * Calculate adaptive weights based on data quality and availability
   */
  private calculateAdaptiveWeights(modalityResults: ModalityResult[]): FusionWeights {
    const weights: FusionWeights = { speech: 0, retinal: 0, risk: 0, motor: 0 };
    let totalWeight = 0;
    
    modalityResults.forEach(result => {
      if (result.available && result.quality >= this.qualityThresholds.minimum) {
        // Base weight from configuration
        let weight = this.baseWeights[result.modality];
        
        // Adjust weight based on data quality
        const qualityMultiplier = this.calculateQualityMultiplier(result.quality);
        weight *= qualityMultiplier;
        
        // Adjust weight based on confidence
        const confidenceMultiplier = Math.max(0.5, result.confidence / 100);
        weight *= confidenceMultiplier;
        
        weights[result.modality] = weight;
        totalWeight += weight;
      }
    });
    
    // Normalize weights to sum to 1
    if (totalWeight > 0) {
      Object.keys(weights).forEach(key => {
        weights[key as keyof FusionWeights] /= totalWeight;
      });
    }
    
    return weights;
  }

  /**
   * Calculate quality multiplier for weight adjustment
   */
  private calculateQualityMultiplier(quality: number): number {
    if (quality >= this.qualityThresholds.excellent) {
      return 1.2; // Boost for excellent quality
    } else if (quality >= this.qualityThresholds.good) {
      return 1.0; // Full weight for good quality
    } else if (quality >= this.qualityThresholds.minimum) {
      return 0.5 + 0.5 * (quality - this.qualityThresholds.minimum) / 
                   (this.qualityThresholds.good - this.qualityThresholds.minimum);
    } else {
      return 0; // Exclude poor quality data
    }
  }

  /**
   * Perform weighted fusion of modality scores
   */
  private performWeightedFusion(
    modalityResults: ModalityResult[],
    weights: FusionWeights
  ): number {
    let fusedScore = 0;
    let totalWeight = 0;
    
    modalityResults.forEach(result => {
      const weight = weights[result.modality];
      if (weight > 0 && result.available) {
        fusedScore += result.score * weight;
        totalWeight += weight;
      }
    });
    
    // If no modalities available, return neutral score
    if (totalWeight === 0) {
      return 50; // Neutral risk
    }
    
    // Apply ensemble correction for missing modalities
    const completeness = totalWeight / Object.values(this.baseWeights).reduce((a, b) => a + b, 0);
    const ensembleCorrection = this.calculateEnsembleCorrection(completeness);
    
    return Math.min(Math.max(fusedScore * ensembleCorrection, 0), 100);
  }

  /**
   * Calculate ensemble correction for missing modalities
   */
  private calculateEnsembleCorrection(completeness: number): number {
    // Adjust score based on data completeness
    // More conservative (higher risk) when data is incomplete
    if (completeness >= 0.8) {
      return 1.0; // Full confidence
    } else if (completeness >= 0.6) {
      return 1.1; // Slight increase in risk estimate
    } else if (completeness >= 0.4) {
      return 1.2; // Moderate increase
    } else {
      return 1.3; // Conservative estimate
    }
  }

  /**
   * Quantify uncertainty in the fusion result
   */
  private quantifyUncertainty(
    modalityResults: ModalityResult[],
    weights: FusionWeights
  ): { confidence: number; factors: string[] } {
    const uncertaintyFactors: string[] = [];
    let overallConfidence = 100;
    
    // Check data availability
    const availableModalities = modalityResults.filter(r => r.available).length;
    if (availableModalities < 3) {
      uncertaintyFactors.push('Limited modality data available');
      overallConfidence -= (4 - availableModalities) * 15;
    }
    
    // Check data quality
    const lowQualityModalities = modalityResults.filter(
      r => r.available && r.quality < this.qualityThresholds.good
    );
    if (lowQualityModalities.length > 0) {
      uncertaintyFactors.push('Some modalities have reduced data quality');
      overallConfidence -= lowQualityModalities.length * 10;
    }
    
    // Check confidence consistency
    const availableResults = modalityResults.filter(r => r.available);
    if (availableResults.length > 1) {
      const confidences = availableResults.map(r => r.confidence);
      const confidenceVariance = this.calculateVariance(confidences);
      if (confidenceVariance > 400) { // High variance in confidence
        uncertaintyFactors.push('Inconsistent confidence across modalities');
        overallConfidence -= 10;
      }
    }
    
    // Check score consistency
    if (availableResults.length > 1) {
      const scores = availableResults.map(r => r.score);
      const scoreVariance = this.calculateVariance(scores);
      if (scoreVariance > 900) { // High variance in scores
        uncertaintyFactors.push('Conflicting risk assessments between modalities');
        overallConfidence -= 15;
      }
    }
    
    // Minimum confidence threshold
    overallConfidence = Math.max(overallConfidence, 40);
    
    return {
      confidence: overallConfidence,
      factors: uncertaintyFactors,
    };
  }

  /**
   * Calculate variance of an array of numbers
   */
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
  }

  /**
   * Determine risk category based on NRI score
   */
  private determineRiskCategory(nriScore: number): 'low' | 'moderate' | 'high' | 'critical' {
    if (nriScore <= 25) return 'low';
    if (nriScore <= 50) return 'moderate';
    if (nriScore <= 75) return 'high';
    return 'critical';
  }

  /**
   * Generate clinical recommendations based on fusion results
   */
  private generateClinicalRecommendations(
    nriScore: number,
    modalityResults: ModalityResult[],
    riskCategory: 'low' | 'moderate' | 'high' | 'critical'
  ): string[] {
    const recommendations: string[] = [];
    
    // Risk-based recommendations
    switch (riskCategory) {
      case 'critical':
        recommendations.push('Immediate neurological evaluation recommended');
        recommendations.push('Consider comprehensive cognitive assessment');
        recommendations.push('Discuss findings with healthcare provider urgently');
        break;
      case 'high':
        recommendations.push('Neurological consultation within 3 months');
        recommendations.push('Annual cognitive monitoring recommended');
        recommendations.push('Implement risk reduction strategies');
        break;
      case 'moderate':
        recommendations.push('Regular monitoring with healthcare provider');
        recommendations.push('Focus on modifiable risk factors');
        recommendations.push('Consider lifestyle interventions');
        break;
      case 'low':
        recommendations.push('Continue healthy lifestyle practices');
        recommendations.push('Routine screening as per age guidelines');
        break;
    }
    
    // Modality-specific recommendations
    const speechResult = modalityResults.find(r => r.modality === 'speech');
    if (speechResult?.available && speechResult.score > 50) {
      recommendations.push('Speech therapy evaluation may be beneficial');
    }
    
    const retinalResult = modalityResults.find(r => r.modality === 'retinal');
    if (retinalResult?.available && retinalResult.score > 50) {
      recommendations.push('Ophthalmological follow-up recommended');
    }
    
    // Data quality recommendations
    const lowQualityModalities = modalityResults.filter(
      r => r.available && r.quality < this.qualityThresholds.good
    );
    if (lowQualityModalities.length > 0) {
      recommendations.push('Consider repeat assessment with higher quality data');
    }
    
    return recommendations;
  }

  /**
   * Generate clinical notes for healthcare providers
   */
  private generateClinicalNotes(modalityResults: ModalityResult[], nriScore: number): string[] {
    const notes: string[] = [];
    
    // Overall assessment note
    notes.push(`Unified NRI Score: ${Math.round(nriScore)}/100`);
    
    // Modality-specific notes
    modalityResults.forEach(result => {
      if (result.available) {
        const modalityName = result.modality.charAt(0).toUpperCase() + result.modality.slice(1);
        notes.push(
          `${modalityName} Analysis: ${result.score}/100 (Quality: ${Math.round(result.quality * 100)}%)`
        );
        
        if (result.findings.length > 0) {
          notes.push(`${modalityName} Findings: ${result.findings.slice(0, 3).join(', ')}`);
        }
      }
    });
    
    // Processing performance note
    const totalProcessingTime = modalityResults.reduce(
      (sum, result) => sum + result.processingTime, 0
    );
    notes.push(`Total Processing Time: ${Math.round(totalProcessingTime)}ms`);
    
    return notes;
  }

  /**
   * Calculate data completeness percentage
   */
  private calculateDataCompleteness(modalityResults: ModalityResult[]): number {
    const availableModalities = modalityResults.filter(r => r.available).length;
    const totalModalities = modalityResults.length;
    return Math.round((availableModalities / totalModalities) * 100);
  }

  /**
   * Get modality importance for interpretation
   */
  getModalityImportance(): Record<string, string> {
    return {
      speech: 'Voice biomarkers provide early detection of neurological changes',
      retinal: 'Retinal vascular patterns reflect brain health status',
      risk: 'Comprehensive risk factors inform overall probability',
      motor: 'Physical manifestations indicate disease progression',
    };
  }

  /**
   * Get risk category descriptions
   */
  getRiskCategoryDescriptions(): Record<string, string> {
    return {
      low: 'Low neurological risk - routine monitoring recommended',
      moderate: 'Moderate risk - enhanced monitoring and lifestyle interventions',
      high: 'High risk - medical evaluation and intervention recommended',
      critical: 'Critical risk - immediate medical attention required',
    };
  }
}

// Export singleton instance
export const nriFusionCalculator = new NRIFusionCalculator();
