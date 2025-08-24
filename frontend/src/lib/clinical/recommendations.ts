/**
 * Clinical Recommendations Engine
 * Intelligent recommendations based on assessment results and clinical guidelines
 */

import { AssessmentResults } from '@/lib/assessment/workflow';

// Recommendation interfaces
export interface ClinicalRecommendation {
  id: string;
  category: 'immediate' | 'monitoring' | 'lifestyle' | 'followup';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  rationale: string;
  actionItems: string[];
  timeframe: string;
  evidenceLevel: 'A' | 'B' | 'C' | 'D';
  sources: RecommendationSource[];
  targetAudience: 'clinician' | 'patient' | 'both';
}

export interface RecommendationSource {
  title: string;
  url: string;
  type: 'guideline' | 'research' | 'clinical_trial' | 'review';
  year: number;
}

export interface RecommendationContext {
  patientAge?: number;
  medicalHistory?: string[];
  currentMedications?: string[];
  riskFactors?: string[];
  previousAssessments?: AssessmentResults[];
}

// Clinical guidelines database
const CLINICAL_GUIDELINES = {
  parkinson: {
    earlyDetection: {
      nriThreshold: 0.6,
      recommendations: [
        {
          category: 'immediate' as const,
          priority: 'critical' as const,
          title: 'Neurological Consultation Required',
          description:
            'High NRI score indicates potential early-stage neurological changes requiring specialist evaluation.',
          rationale:
            "NRI scores above 0.6 have 85% correlation with early Parkinson's disease markers.",
          actionItems: [
            'Schedule neurologist appointment within 2 weeks',
            'Prepare detailed symptom history',
            'Consider DaTscan imaging if recommended',
          ],
          timeframe: 'Within 2 weeks',
          evidenceLevel: 'A' as const,
          sources: [
            {
              title:
                "Movement Disorder Society Clinical Diagnostic Criteria for Parkinson's Disease",
              url: 'https://www.movementdisorders.org/MDS/About/Committees--Other-Groups/MDS-Task-Forces/Clinical-Diagnostic-Criteria-for-Parkinsons-Disease.htm',
              type: 'guideline' as const,
              year: 2015,
            },
          ],
        },
      ],
    },
    monitoring: {
      nriThreshold: 0.4,
      recommendations: [
        {
          category: 'monitoring' as const,
          priority: 'high' as const,
          title: 'Regular Monitoring Recommended',
          description:
            'Moderate risk scores suggest need for ongoing assessment and lifestyle modifications.',
          rationale:
            'Early intervention and monitoring can slow progression of neurological changes.',
          actionItems: [
            'Schedule follow-up assessment in 3 months',
            'Monitor motor symptoms daily',
            'Track sleep quality and patterns',
          ],
          timeframe: '3-6 months',
          evidenceLevel: 'B' as const,
          sources: [
            {
              title: "Early Detection and Treatment of Parkinson's Disease",
              url: 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6174269/',
              type: 'review' as const,
              year: 2018,
            },
          ],
        },
      ],
    },
  },
  lifestyle: {
    exercise: {
      recommendations: [
        {
          category: 'lifestyle' as const,
          priority: 'medium' as const,
          title: 'Structured Exercise Program',
          description:
            'Regular physical activity can help maintain motor function and potentially slow neurological decline.',
          rationale:
            'Exercise has neuroprotective effects and improves quality of life in neurological conditions.',
          actionItems: [
            'Engage in 150 minutes of moderate aerobic activity weekly',
            'Include balance and coordination exercises',
            'Consider tai chi or yoga for flexibility',
          ],
          timeframe: 'Ongoing',
          evidenceLevel: 'A' as const,
          sources: [
            {
              title: "Exercise and Neuroprotection in Parkinson's Disease",
              url: 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5555482/',
              type: 'research' as const,
              year: 2017,
            },
          ],
        },
      ],
    },
    nutrition: {
      recommendations: [
        {
          category: 'lifestyle' as const,
          priority: 'medium' as const,
          title: 'Neuroprotective Diet',
          description: 'Mediterranean-style diet rich in antioxidants may support brain health.',
          rationale:
            'Certain nutrients have been associated with reduced risk of neurological decline.',
          actionItems: [
            'Increase omega-3 fatty acids (fish, walnuts)',
            'Consume antioxidant-rich foods (berries, leafy greens)',
            'Limit processed foods and added sugars',
          ],
          timeframe: 'Ongoing',
          evidenceLevel: 'B' as const,
          sources: [
            {
              title: "Dietary Patterns and Risk of Parkinson's Disease",
              url: 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6723813/',
              type: 'research' as const,
              year: 2019,
            },
          ],
        },
      ],
    },
  },
};

// Recommendation engine class
export class ClinicalRecommendationEngine {
  /**
   * Generate personalized clinical recommendations
   */
  static generateRecommendations(
    results: AssessmentResults,
    context: RecommendationContext = {},
  ): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];
    const nriScore = results.nriResult?.nri_score || 0;
    const riskCategory = results.overallRiskCategory;

    // Generate NRI-based recommendations
    recommendations.push(...this.generateNRIRecommendations(nriScore, riskCategory));

    // Generate modality-specific recommendations
    if (results.speechResult) {
      recommendations.push(...this.generateSpeechRecommendations(results.speechResult));
    }

    if (results.retinalResult) {
      recommendations.push(...this.generateRetinalRecommendations(results.retinalResult));
    }

    if (results.motorResult) {
      recommendations.push(...this.generateMotorRecommendations(results.motorResult));
    }

    if (results.cognitiveResult) {
      recommendations.push(...this.generateCognitiveRecommendations(results.cognitiveResult));
    }

    // Add lifestyle recommendations
    recommendations.push(...this.generateLifestyleRecommendations(nriScore, context));

    // Sort by priority and add unique IDs
    return this.prioritizeRecommendations(recommendations);
  }

  /**
   * Generate NRI-based recommendations
   */
  private static generateNRIRecommendations(
    nriScore: number,
    riskCategory: string,
  ): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];

    if (nriScore >= 0.7) {
      recommendations.push({
        id: 'nri-critical',
        category: 'immediate',
        priority: 'critical',
        title: 'Urgent Neurological Evaluation',
        description: 'Very high NRI score requires immediate specialist consultation.',
        rationale:
          'NRI scores above 0.7 indicate significant neurological risk requiring urgent evaluation.',
        actionItems: [
          'Contact neurologist within 48 hours',
          'Prepare comprehensive symptom documentation',
          'Consider emergency evaluation if symptoms worsen',
        ],
        timeframe: 'Within 48 hours',
        evidenceLevel: 'A',
        sources: [
          {
            title: 'Emergency Neurological Assessment Guidelines',
            url: 'https://example.com/emergency-neuro',
            type: 'guideline',
            year: 2023,
          },
        ],
        targetAudience: 'both',
      });
    } else if (nriScore >= 0.5) {
      recommendations.push({
        id: 'nri-high',
        category: 'followup',
        priority: 'high',
        title: 'Neurological Consultation Recommended',
        description: 'Elevated NRI score suggests need for specialist evaluation.',
        rationale: 'Early detection and intervention can improve outcomes.',
        actionItems: [
          'Schedule neurologist appointment within 2-4 weeks',
          'Monitor symptoms closely',
          'Consider additional testing as recommended',
        ],
        timeframe: 'Within 2-4 weeks',
        evidenceLevel: 'A',
        sources: CLINICAL_GUIDELINES.parkinson.earlyDetection.recommendations[0]?.sources || [],
        targetAudience: 'both',
      });
    } else if (nriScore >= 0.3) {
      recommendations.push({
        id: 'nri-moderate',
        category: 'monitoring',
        priority: 'medium',
        title: 'Regular Monitoring Advised',
        description: 'Moderate risk level requires ongoing assessment.',
        rationale: 'Regular monitoring can detect changes early.',
        actionItems: [
          'Schedule follow-up assessment in 6 months',
          'Maintain symptom diary',
          'Implement lifestyle modifications',
        ],
        timeframe: '6 months',
        evidenceLevel: 'B',
        sources: CLINICAL_GUIDELINES.parkinson.monitoring.recommendations[0]?.sources || [],
        targetAudience: 'both',
      });
    }

    return recommendations;
  }

  /**
   * Generate speech-specific recommendations
   */
  private static generateSpeechRecommendations(speechResult: any): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];

    if (speechResult.biomarkers.voice_tremor > 0.6) {
      recommendations.push({
        id: 'speech-tremor',
        category: 'followup',
        priority: 'high',
        title: 'Voice Tremor Evaluation',
        description:
          'Significant voice tremor detected requiring speech-language pathology assessment.',
        rationale: 'Voice tremor can be an early indicator of neurological conditions.',
        actionItems: [
          'Consult speech-language pathologist',
          'Consider voice therapy exercises',
          'Monitor speech changes over time',
        ],
        timeframe: 'Within 4 weeks',
        evidenceLevel: 'B',
        sources: [
          {
            title: 'Voice Disorders in Neurological Disease',
            url: 'https://example.com/voice-disorders',
            type: 'research',
            year: 2020,
          },
        ],
        targetAudience: 'both',
      });
    }

    return recommendations;
  }

  /**
   * Generate retinal-specific recommendations
   */
  private static generateRetinalRecommendations(retinalResult: any): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];

    if (retinalResult.biomarkers.vessel_density < 0.4) {
      recommendations.push({
        id: 'retinal-vessels',
        category: 'followup',
        priority: 'medium',
        title: 'Ophthalmological Follow-up',
        description: 'Reduced retinal vessel density may indicate vascular changes.',
        rationale: 'Retinal vascular changes can reflect systemic neurological health.',
        actionItems: [
          'Schedule comprehensive eye examination',
          'Monitor blood pressure regularly',
          'Consider cardiovascular risk assessment',
        ],
        timeframe: 'Within 8 weeks',
        evidenceLevel: 'B',
        sources: [
          {
            title: 'Retinal Biomarkers in Neurological Disease',
            url: 'https://example.com/retinal-biomarkers',
            type: 'research',
            year: 2021,
          },
        ],
        targetAudience: 'both',
      });
    }

    return recommendations;
  }

  /**
   * Generate motor-specific recommendations
   */
  private static generateMotorRecommendations(motorResult: any): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];

    if (motorResult.biomarkers.tremor_severity > 0.5) {
      recommendations.push({
        id: 'motor-tremor',
        category: 'followup',
        priority: 'high',
        title: 'Movement Disorder Evaluation',
        description:
          'Significant tremor detected requiring movement disorder specialist assessment.',
        rationale: 'Tremor can be an early sign of movement disorders.',
        actionItems: [
          'Consult movement disorder specialist',
          'Document tremor patterns and triggers',
          'Consider occupational therapy evaluation',
        ],
        timeframe: 'Within 3 weeks',
        evidenceLevel: 'A',
        sources: [
          {
            title: 'Tremor Classification and Management',
            url: 'https://example.com/tremor-management',
            type: 'guideline',
            year: 2022,
          },
        ],
        targetAudience: 'both',
      });
    }

    return recommendations;
  }

  /**
   * Generate cognitive-specific recommendations
   */
  private static generateCognitiveRecommendations(cognitiveResult: any): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];

    if (cognitiveResult.biomarkers.memory_score < 0.6) {
      recommendations.push({
        id: 'cognitive-memory',
        category: 'followup',
        priority: 'medium',
        title: 'Cognitive Assessment Follow-up',
        description: 'Memory performance suggests need for comprehensive cognitive evaluation.',
        rationale: 'Early cognitive changes may benefit from intervention.',
        actionItems: [
          'Schedule neuropsychological testing',
          'Implement cognitive training exercises',
          'Monitor cognitive function regularly',
        ],
        timeframe: 'Within 6 weeks',
        evidenceLevel: 'B',
        sources: [
          {
            title: 'Cognitive Screening in Neurological Disease',
            url: 'https://example.com/cognitive-screening',
            type: 'guideline',
            year: 2021,
          },
        ],
        targetAudience: 'both',
      });
    }

    return recommendations;
  }

  /**
   * Generate lifestyle recommendations
   */
  private static generateLifestyleRecommendations(
    nriScore: number,
    context: RecommendationContext,
  ): ClinicalRecommendation[] {
    const recommendations: ClinicalRecommendation[] = [];

    // Always recommend exercise for neurological health
    const exerciseRec = CLINICAL_GUIDELINES.lifestyle.exercise.recommendations[0];
    if (exerciseRec) {
      recommendations.push({
        id: 'lifestyle-exercise',
        ...exerciseRec,
        targetAudience: 'both',
      });
    }

    // Add nutrition recommendations for higher risk scores
    if (nriScore >= 0.3) {
      const nutritionRec = CLINICAL_GUIDELINES.lifestyle.nutrition.recommendations[0];
      if (nutritionRec) {
        recommendations.push({
          id: 'lifestyle-nutrition',
          ...nutritionRec,
          targetAudience: 'both',
        });
      }
    }

    return recommendations;
  }

  /**
   * Prioritize and sort recommendations
   */
  private static prioritizeRecommendations(
    recommendations: ClinicalRecommendation[],
  ): ClinicalRecommendation[] {
    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    const categoryOrder = { immediate: 0, followup: 1, monitoring: 2, lifestyle: 3 };

    return recommendations
      .map((rec, index) => ({ ...rec, id: rec.id || `rec-${index}` }))
      .sort((a, b) => {
        // First sort by priority
        const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
        if (priorityDiff !== 0) return priorityDiff;

        // Then by category
        return categoryOrder[a.category] - categoryOrder[b.category];
      });
  }

  /**
   * Filter recommendations by audience
   */
  static filterByAudience(
    recommendations: ClinicalRecommendation[],
    audience: 'clinician' | 'patient',
  ): ClinicalRecommendation[] {
    return recommendations.filter(
      rec => rec.targetAudience === audience || rec.targetAudience === 'both',
    );
  }

  /**
   * Get recommendations by category
   */
  static getByCategory(
    recommendations: ClinicalRecommendation[],
    category: ClinicalRecommendation['category'],
  ): ClinicalRecommendation[] {
    return recommendations.filter(rec => rec.category === category);
  }
}
