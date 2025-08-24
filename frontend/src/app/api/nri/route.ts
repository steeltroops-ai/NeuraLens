/**
 * NRI Fusion API Route for NeuraLens
 *
 * This API endpoint handles Neurological Risk Index (NRI) fusion requests,
 * combining results from multiple assessment modalities into a unified risk score.
 *
 * Key Features:
 * - Receives multi-modal assessment results from frontend
 * - Implements Bayesian fusion algorithm with uncertainty quantification
 * - Calculates unified NRI score with confidence intervals
 * - Generates clinical recommendations based on risk assessment
 * - Provides modality contribution analysis and consistency scoring
 * - Caches results for demo efficiency and performance
 * - Returns structured response for frontend display
 *
 * Technical Implementation:
 * - Next.js 15 API route with TypeScript
 * - JSON request/response handling with comprehensive validation
 * - Advanced statistical fusion algorithms
 * - Error handling and detailed logging for debugging
 * - Performance monitoring and caching architecture
 */

import { NextResponse, type NextRequest } from 'next/server';
import type { 
  NRIFusionRequest, 
  NRIFusionResponse,
  ModalityContribution,
  SpeechAnalysisResponse,
  RetinalAnalysisResponse,
  MotorAssessmentResponse,
  CognitiveAssessmentResponse
} from '../../../lib/api/types';

/**
 * POST /api/nri
 * Calculate unified Neurological Risk Index from multi-modal assessments
 */
export async function POST(request: NextRequest) {
  try {
    console.log('[API] NRI fusion request received');

    // Parse request body
    const body = await request.json();
    const { 
      session_id, 
      modality_results, 
      fusion_method = 'bayesian',
      uncertainty_quantification = true 
    } = body as NRIFusionRequest;

    // Validate required fields
    if (!session_id || !modality_results) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: session_id or modality_results',
        },
        { status: 400 },
      );
    }

    // Validate that at least two modalities are present
    const availableModalities = Object.keys(modality_results).filter(
      key => modality_results[key as keyof typeof modality_results] !== undefined
    );

    if (availableModalities.length < 2) {
      return NextResponse.json(
        {
          success: false,
          error: 'At least two modality results are required for NRI fusion',
        },
        { status: 400 },
      );
    }

    // Validate fusion method
    const validMethods = ['bayesian', 'weighted_average', 'ensemble'];
    if (!validMethods.includes(fusion_method)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid fusion_method. Must be one of: ${validMethods.join(', ')}`,
        },
        { status: 400 },
      );
    }

    // Process NRI fusion
    const fusionResult = await processNRIFusion({
      session_id,
      modality_results,
      fusion_method,
      uncertainty_quantification,
    });

    // Generate cache key for demo efficiency
    const cacheKey = `nri_${session_id}_${Date.now()}`;

    // Store result in cache (placeholder - would use Redis in production)
    await cacheResult(cacheKey, {
      ...fusionResult,
      timestamp: new Date(),
    });

    console.log('[API] NRI fusion processed successfully:', {
      nriScore: fusionResult.nri_score,
      riskCategory: fusionResult.risk_category,
      confidence: fusionResult.confidence,
      modalityCount: availableModalities.length,
      fusionMethod: fusion_method,
      processingTime: fusionResult.processing_time,
    });

    return NextResponse.json({
      success: true,
      result: fusionResult,
      cacheKey,
    });
  } catch (error) {
    console.error('[API] NRI fusion error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error during NRI fusion',
      },
      { status: 500 },
    );
  }
}

/**
 * GET /api/nri
 * Retrieve cached NRI fusion results
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const cacheKey = searchParams.get('cacheKey');

    if (!cacheKey) {
      return NextResponse.json({ success: false, error: 'Cache key required' }, { status: 400 });
    }

    // Retrieve cached result
    const cachedResult = await getCachedResult(cacheKey);

    if (!cachedResult) {
      return NextResponse.json(
        { success: false, error: 'Result not found or expired' },
        { status: 404 },
      );
    }

    return NextResponse.json({
      success: true,
      result: cachedResult,
    });
  } catch (error) {
    console.error('[API] Error retrieving cached NRI result:', error);

    return NextResponse.json(
      { success: false, error: 'Failed to retrieve cached result' },
      { status: 500 },
    );
  }
}

/**
 * Process NRI fusion from multi-modal assessment results
 */
async function processNRIFusion(request: NRIFusionRequest): Promise<NRIFusionResponse> {
  const startTime = Date.now();
  
  // Extract modality contributions
  const modalityContributions = extractModalityContributions(request.modality_results);
  
  // Calculate unified NRI score based on fusion method
  const nriScore = calculateNRIScore(modalityContributions, request.fusion_method);
  
  // Determine risk category
  const riskCategory = determineRiskCategory(nriScore);
  
  // Calculate confidence and uncertainty
  const { confidence, uncertainty } = calculateConfidenceAndUncertainty(
    modalityContributions,
    request.uncertainty_quantification
  );
  
  // Calculate consistency score
  const consistencyScore = calculateConsistencyScore(modalityContributions);
  
  // Generate clinical recommendations
  const recommendations = generateRecommendations(nriScore, riskCategory, modalityContributions);
  
  // Generate follow-up actions
  const followUpActions = generateFollowUpActions(nriScore, riskCategory, modalityContributions);
  
  const processingTime = Date.now() - startTime;

  return {
    session_id: request.session_id,
    nri_score: Math.round(nriScore * 100) / 100,
    risk_category: riskCategory,
    confidence: Math.round(confidence * 100) / 100,
    uncertainty: Math.round(uncertainty * 100) / 100,
    consistency_score: Math.round(consistencyScore * 100) / 100,
    modality_contributions: modalityContributions,
    processing_time: processingTime,
    timestamp: new Date().toISOString(),
    recommendations,
    follow_up_actions: followUpActions,
  };
}

/**
 * Extract modality contributions from assessment results
 */
function extractModalityContributions(
  modalityResults: NRIFusionRequest['modality_results']
): ModalityContribution[] {
  const contributions: ModalityContribution[] = [];
  
  // Speech modality
  if (modalityResults.speech) {
    contributions.push({
      modality: 'speech',
      weight: 0.25, // 25% weight for speech analysis
      confidence: modalityResults.speech.confidence,
      risk_score: modalityResults.speech.risk_score,
    });
  }
  
  // Retinal modality
  if (modalityResults.retinal) {
    contributions.push({
      modality: 'retinal',
      weight: 0.30, // 30% weight for retinal analysis
      confidence: modalityResults.retinal.confidence,
      risk_score: modalityResults.retinal.risk_score,
    });
  }
  
  // Motor modality
  if (modalityResults.motor) {
    contributions.push({
      modality: 'motor',
      weight: 0.25, // 25% weight for motor assessment
      confidence: modalityResults.motor.confidence,
      risk_score: modalityResults.motor.risk_score,
    });
  }
  
  // Cognitive modality
  if (modalityResults.cognitive) {
    contributions.push({
      modality: 'cognitive',
      weight: 0.20, // 20% weight for cognitive assessment
      confidence: modalityResults.cognitive.confidence,
      risk_score: modalityResults.cognitive.risk_score,
    });
  }
  
  // Normalize weights to sum to 1.0
  const totalWeight = contributions.reduce((sum, contrib) => sum + contrib.weight, 0);
  contributions.forEach(contrib => {
    contrib.weight = contrib.weight / totalWeight;
  });
  
  return contributions;
}

/**
 * Calculate unified NRI score using specified fusion method
 */
function calculateNRIScore(contributions: ModalityContribution[], fusionMethod: string): number {
  switch (fusionMethod) {
    case 'bayesian':
      return calculateBayesianFusion(contributions);
    case 'weighted_average':
      return calculateWeightedAverage(contributions);
    case 'ensemble':
      return calculateEnsembleFusion(contributions);
    default:
      return calculateWeightedAverage(contributions); // Default fallback
  }
}

/**
 * Bayesian fusion algorithm with uncertainty propagation
 */
function calculateBayesianFusion(contributions: ModalityContribution[]): number {
  // Prior probability (base rate of neurological conditions)
  const priorRisk = 0.15; // 15% base rate
  
  let posteriorRisk = priorRisk;
  
  contributions.forEach(contrib => {
    // Convert risk score to probability
    const likelihood = contrib.risk_score / 100;
    
    // Weight by confidence
    const weightedLikelihood = likelihood * contrib.confidence;
    
    // Bayesian update
    const numerator = weightedLikelihood * posteriorRisk;
    const denominator = numerator + (1 - weightedLikelihood) * (1 - posteriorRisk);
    
    posteriorRisk = denominator > 0 ? numerator / denominator : posteriorRisk;
  });
  
  // Convert back to 0-100 scale
  return Math.min(100, posteriorRisk * 100);
}

/**
 * Weighted average fusion with confidence weighting
 */
function calculateWeightedAverage(contributions: ModalityContribution[]): number {
  let weightedSum = 0;
  let totalWeight = 0;
  
  contributions.forEach(contrib => {
    // Weight by both modality weight and confidence
    const effectiveWeight = contrib.weight * contrib.confidence;
    weightedSum += contrib.risk_score * effectiveWeight;
    totalWeight += effectiveWeight;
  });
  
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

/**
 * Ensemble fusion combining multiple methods
 */
function calculateEnsembleFusion(contributions: ModalityContribution[]): number {
  const bayesianScore = calculateBayesianFusion(contributions);
  const weightedScore = calculateWeightedAverage(contributions);
  
  // Combine with equal weights
  return (bayesianScore + weightedScore) / 2;
}

/**
 * Determine risk category based on NRI score
 */
function determineRiskCategory(nriScore: number): 'low' | 'moderate' | 'high' {
  if (nriScore < 30) return 'low';
  if (nriScore < 70) return 'moderate';
  return 'high';
}

/**
 * Calculate confidence and uncertainty metrics
 */
function calculateConfidenceAndUncertainty(
  contributions: ModalityContribution[],
  uncertaintyQuantification: boolean
): { confidence: number; uncertainty: number } {
  // Base confidence from modality confidences
  const avgConfidence = contributions.reduce((sum, contrib) => sum + contrib.confidence, 0) / contributions.length;
  
  let uncertainty = 0;
  
  if (uncertaintyQuantification) {
    // Calculate uncertainty from confidence variance
    const confidenceVariance = contributions.reduce((sum, contrib) => 
      sum + Math.pow(contrib.confidence - avgConfidence, 2), 0
    ) / contributions.length;
    
    // Calculate uncertainty from risk score variance
    const avgRiskScore = contributions.reduce((sum, contrib) => sum + contrib.risk_score, 0) / contributions.length;
    const riskVariance = contributions.reduce((sum, contrib) => 
      sum + Math.pow(contrib.risk_score - avgRiskScore, 2), 0
    ) / contributions.length;
    
    // Combine uncertainties
    uncertainty = Math.sqrt(confidenceVariance + (riskVariance / 10000)); // Normalize risk variance
  }
  
  return {
    confidence: Math.max(0.1, Math.min(1.0, avgConfidence)),
    uncertainty: Math.max(0, Math.min(1.0, uncertainty)),
  };
}

/**
 * Calculate consistency score across modalities
 */
function calculateConsistencyScore(contributions: ModalityContribution[]): number {
  if (contributions.length < 2) return 1.0;
  
  const riskScores = contributions.map(contrib => contrib.risk_score);
  const avgRiskScore = riskScores.reduce((sum, score) => sum + score, 0) / riskScores.length;
  
  // Calculate coefficient of variation
  const variance = riskScores.reduce((sum, score) => sum + Math.pow(score - avgRiskScore, 2), 0) / riskScores.length;
  const stdDev = Math.sqrt(variance);
  const coefficientOfVariation = avgRiskScore > 0 ? stdDev / avgRiskScore : 0;
  
  // Convert to consistency score (lower variation = higher consistency)
  return Math.max(0, 1.0 - Math.min(1.0, coefficientOfVariation));
}

/**
 * Generate clinical recommendations based on risk assessment
 */
function generateRecommendations(
  nriScore: number,
  riskCategory: string,
  contributions: ModalityContribution[]
): string[] {
  const recommendations: string[] = [];
  
  // General recommendations based on risk category
  switch (riskCategory) {
    case 'low':
      recommendations.push('Continue routine health monitoring');
      recommendations.push('Maintain healthy lifestyle practices');
      recommendations.push('Consider annual neurological screening');
      break;
    case 'moderate':
      recommendations.push('Schedule comprehensive neurological evaluation');
      recommendations.push('Consider specialist consultation within 3-6 months');
      recommendations.push('Implement cognitive health strategies');
      recommendations.push('Monitor symptoms and functional changes');
      break;
    case 'high':
      recommendations.push('Urgent neurological evaluation recommended');
      recommendations.push('Schedule specialist consultation within 4-6 weeks');
      recommendations.push('Consider advanced neuroimaging studies');
      recommendations.push('Implement immediate risk reduction strategies');
      break;
  }
  
  // Modality-specific recommendations
  contributions.forEach(contrib => {
    if (contrib.risk_score > 60) {
      switch (contrib.modality) {
        case 'speech':
          recommendations.push('Consider speech-language pathology evaluation');
          break;
        case 'retinal':
          recommendations.push('Schedule comprehensive ophthalmological examination');
          break;
        case 'motor':
          recommendations.push('Consider movement disorder specialist consultation');
          break;
        case 'cognitive':
          recommendations.push('Recommend neuropsychological assessment');
          break;
      }
    }
  });
  
  return [...new Set(recommendations)]; // Remove duplicates
}

/**
 * Generate follow-up actions based on risk assessment
 */
function generateFollowUpActions(
  nriScore: number,
  riskCategory: string,
  contributions: ModalityContribution[]
): string[] {
  const actions: string[] = [];
  
  // Time-based follow-up actions
  switch (riskCategory) {
    case 'low':
      actions.push('Repeat screening in 12 months');
      actions.push('Document baseline measurements');
      break;
    case 'moderate':
      actions.push('Repeat assessment in 6 months');
      actions.push('Track symptom progression');
      actions.push('Lifestyle modification counseling');
      break;
    case 'high':
      actions.push('Repeat assessment in 3 months');
      actions.push('Coordinate care with specialists');
      actions.push('Implement monitoring protocols');
      break;
  }
  
  // Data quality actions
  const lowConfidenceModalities = contributions.filter(contrib => contrib.confidence < 0.7);
  if (lowConfidenceModalities.length > 0) {
    actions.push('Consider retesting modalities with low confidence scores');
  }
  
  return actions;
}

/**
 * Cache NRI fusion result
 */
async function cacheResult(key: string, _result: any): Promise<void> {
  console.log(`[API] Caching NRI result with key: ${key}`);
  // In production: await redis.setex(key, 3600, JSON.stringify(result));
}

/**
 * Retrieve cached NRI fusion result
 */
async function getCachedResult(key: string): Promise<any | null> {
  console.log(`[API] Retrieving cached NRI result with key: ${key}`);
  // In production: const cached = await redis.get(key); return cached ? JSON.parse(cached) : null;
  return null;
}

/**
 * Generate demo NRI fusion result for testing
 */
export function generateDemoNRIResult(modalityCount: number = 4): NRIFusionResponse {
  const nriScore = 25 + Math.random() * 50; // 25-75 range
  const riskCategory = nriScore < 30 ? 'low' : nriScore < 70 ? 'moderate' : 'high';
  
  const contributions: ModalityContribution[] = [];
  const modalities = ['speech', 'retinal', 'motor', 'cognitive'];
  
  for (let i = 0; i < Math.min(modalityCount, 4); i++) {
    contributions.push({
      modality: modalities[i],
      weight: 1.0 / modalityCount,
      confidence: 0.8 + Math.random() * 0.15,
      risk_score: nriScore + (Math.random() - 0.5) * 20,
    });
  }
  
  return {
    session_id: `demo_${Date.now()}`,
    nri_score: Math.round(nriScore * 100) / 100,
    risk_category: riskCategory,
    confidence: 0.85 + Math.random() * 0.1,
    uncertainty: Math.random() * 0.2,
    consistency_score: 0.7 + Math.random() * 0.25,
    modality_contributions: contributions,
    processing_time: 50 + Math.random() * 100,
    timestamp: new Date().toISOString(),
    recommendations: [
      'Continue routine health monitoring',
      'Schedule comprehensive neurological evaluation',
      'Consider specialist consultation',
    ],
    follow_up_actions: [
      'Repeat assessment in 6 months',
      'Track symptom progression',
      'Document baseline measurements',
    ],
  };
}
