/**
 * Retinal Analysis API Route for Neuralens
 *
 * This API endpoint handles retinal analysis requests from the frontend,
 * processes the results, and integrates with the NRI (Neuro-Risk Index) system.
 *
 * Key Features:
 * - Receives retinal analysis results from client-side ML processing
 * - Validates and processes retinal biomarkers and risk features
 * - Calculates NRI contribution from vascular and cup-disc metrics
 * - Caches results for demo efficiency and performance
 * - Returns structured response for frontend display
 *
 * Technical Implementation:
 * - Next.js 15 API route with TypeScript
 * - JSON request/response handling with comprehensive validation
 * - Error handling and detailed logging for debugging
 * - Integration with NRI calculation system
 * - Performance monitoring and caching architecture
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  RetinalResult,
  RetinalAnalysisResponse,
} from '../../../types/retinal-analysis';

/**
 * POST /api/retinal
 * Process retinal analysis results and integrate with NRI system
 */
export async function POST(request: NextRequest) {
  try {
    console.log('[API] Retinal analysis request received');

    // Parse request body
    const body = await request.json();
    const { result, sessionId, timestamp } = body;

    // Validate required fields
    if (
      !result ||
      !result.vascularScore ||
      !result.cupDiscRatio ||
      !result.confidence ||
      !result.riskFeatures
    ) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid retinal analysis result format',
        },
        { status: 400 }
      );
    }

    // Process retinal analysis result
    const processedResult = await processRetinalAnalysis(
      result as RetinalResult
    );

    // Calculate NRI contribution from retinal metrics
    const nriContribution = calculateNRIContribution(result as RetinalResult);

    // Generate cache key for demo efficiency
    const cacheKey = `retinal_${sessionId}_${Date.now()}`;

    // Store result in cache (placeholder - would use Redis in production)
    await cacheResult(cacheKey, {
      ...processedResult,
      nriContribution,
      timestamp: new Date(timestamp),
    });

    // Prepare response
    const response: RetinalAnalysisResponse = {
      result: processedResult,
      success: true,
      cacheKey,
      nriContribution,
    };

    console.log('[API] Retinal analysis processed successfully:', {
      vascularScore: processedResult.vascularScore,
      cupDiscRatio: processedResult.cupDiscRatio,
      confidence: processedResult.confidence,
      nriContribution,
      processingTime: processedResult.metadata.processingTime,
    });

    return NextResponse.json(response);
  } catch (error) {
    console.error('[API] Retinal analysis error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error during retinal analysis',
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/retinal
 * Retrieve cached retinal analysis results
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const cacheKey = searchParams.get('cacheKey');

    if (!cacheKey) {
      return NextResponse.json(
        { success: false, error: 'Cache key required' },
        { status: 400 }
      );
    }

    // Retrieve cached result
    const cachedResult = await getCachedResult(cacheKey);

    if (!cachedResult) {
      return NextResponse.json(
        { success: false, error: 'Result not found or expired' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      result: cachedResult,
    });
  } catch (error) {
    console.error('[API] Error retrieving cached result:', error);

    return NextResponse.json(
      { success: false, error: 'Failed to retrieve cached result' },
      { status: 500 }
    );
  }
}

/**
 * Process retinal analysis result and add server-side enhancements
 */
async function processRetinalAnalysis(
  result: RetinalResult
): Promise<RetinalResult> {
  // Add server-side processing and validation
  const processedResult: RetinalResult = {
    ...result,
    // Validate vascular score range
    vascularScore: Math.max(0, Math.min(1, result.vascularScore)),
    // Validate cup-disc ratio range
    cupDiscRatio: Math.max(0, Math.min(1, result.cupDiscRatio)),
    // Validate confidence range
    confidence: Math.max(0, Math.min(1, result.confidence)),
    // Add server processing timestamp
    metadata: {
      ...result.metadata,
      processingTime:
        Date.now() - (result.metadata?.timestamp?.getTime() || Date.now()),
      timestamp: new Date(),
      modelVersion: result.metadata?.modelVersion || 'efficientnet-b0-v1.0',
      gpuAccelerated: result.metadata?.gpuAccelerated || false,
    },
  };

  // Validate risk features
  if (processedResult.riskFeatures) {
    processedResult.riskFeatures = {
      ...processedResult.riskFeatures,
      // Ensure reasonable ranges for risk features
      vesselDensity: Math.max(
        0,
        Math.min(1, processedResult.riskFeatures.vesselDensity)
      ),
      tortuosityIndex: Math.max(
        0,
        Math.min(1, processedResult.riskFeatures.tortuosityIndex)
      ),
      arteriovenousRatio: Math.max(
        0.3,
        Math.min(1.2, processedResult.riskFeatures.arteriovenousRatio)
      ),
      imageQuality: Math.max(
        0,
        Math.min(1, processedResult.riskFeatures.imageQuality)
      ),
      // Validate counts
      hemorrhageCount: Math.max(
        0,
        Math.floor(processedResult.riskFeatures.hemorrhageCount)
      ),
      microaneurysmCount: Math.max(
        0,
        Math.floor(processedResult.riskFeatures.microaneurysmCount)
      ),
      softExudateCount: Math.max(
        0,
        Math.floor(processedResult.riskFeatures.softExudateCount)
      ),
    };
  }

  return processedResult;
}

/**
 * Calculate NRI contribution from retinal analysis
 * Converts retinal metrics to neurological risk assessment
 */
function calculateNRIContribution(result: RetinalResult): number {
  const { vascularScore, cupDiscRatio, confidence, riskFeatures } = result;

  // Base risk calculation from primary metrics
  let riskScore = 0;

  // Vascular score contribution (40% weight)
  // Higher vascular score = higher neurological risk
  riskScore += vascularScore * 40;

  // Cup-disc ratio contribution (35% weight)
  // Abnormal cup-disc ratio indicates increased stroke/glaucoma risk
  const normalCupDiscRatio = 0.3;
  const cupDiscDeviation = Math.abs(cupDiscRatio - normalCupDiscRatio);
  riskScore += Math.min(cupDiscDeviation * 100, 35);

  // Risk features contribution (25% weight)
  if (riskFeatures) {
    let featureRisk = 0;

    // Vessel density (normal range: 0.15-0.25)
    const normalVesselDensity = 0.2;
    const vesselDensityDeviation = Math.abs(
      riskFeatures.vesselDensity - normalVesselDensity
    );
    featureRisk += vesselDensityDeviation * 50;

    // Tortuosity index (higher = more risk)
    featureRisk += riskFeatures.tortuosityIndex * 30;

    // Arteriovenous ratio (normal ~0.67)
    const normalAVRatio = 0.67;
    const avRatioDeviation = Math.abs(
      riskFeatures.arteriovenousRatio - normalAVRatio
    );
    featureRisk += avRatioDeviation * 40;

    // Hemorrhages and microaneurysms
    featureRisk +=
      (riskFeatures.hemorrhageCount + riskFeatures.microaneurysmCount) * 5;

    // Hard and soft exudates
    featureRisk += riskFeatures.hardExudateArea * 20;
    featureRisk += riskFeatures.softExudateCount * 3;

    // Apply feature risk with 25% weight
    riskScore += Math.min(featureRisk, 25);
  }

  // Apply confidence weighting
  const weightedScore = riskScore * confidence;

  // Ensure score is within valid range (0-100)
  return Math.round(Math.max(0, Math.min(100, weightedScore)));
}

/**
 * Cache retinal analysis result
 * In production, this would use Redis or similar caching system
 */
async function cacheResult(key: string, result: any): Promise<void> {
  // Placeholder implementation - would use Redis in production
  // For now, we'll just log the caching operation
  console.log(`[API] Caching retinal result with key: ${key}`);

  // In a real implementation:
  // await redis.setex(key, 3600, JSON.stringify(result)); // Cache for 1 hour
}

/**
 * Retrieve cached retinal analysis result
 * In production, this would use Redis or similar caching system
 */
async function getCachedResult(key: string): Promise<any | null> {
  // Placeholder implementation - would use Redis in production
  console.log(`[API] Retrieving cached retinal result with key: ${key}`);

  // In a real implementation:
  // const cached = await redis.get(key);
  // return cached ? JSON.parse(cached) : null;

  // For now, return null (not found)
  return null;
}

/**
 * Validate retinal analysis request format
 */
function validateRetinalRequest(body: any): boolean {
  return (
    body &&
    body.result &&
    typeof body.result.vascularScore === 'number' &&
    typeof body.result.cupDiscRatio === 'number' &&
    typeof body.result.confidence === 'number' &&
    body.result.riskFeatures &&
    body.result.metadata
  );
}

/**
 * Generate demo retinal analysis result for testing
 * This can be used for demo purposes when the ML model is not available
 */
export function generateDemoRetinalResult(): RetinalResult {
  const vascularScore = 0.3 + Math.random() * 0.5; // 0.3-0.8 range
  const cupDiscRatio = 0.2 + Math.random() * 0.4; // 0.2-0.6 range

  return {
    vascularScore,
    cupDiscRatio,
    confidence: 0.85 + Math.random() * 0.1,
    riskFeatures: {
      vesselDensity: 0.15 + Math.random() * 0.1,
      tortuosityIndex: 0.1 + Math.random() * 0.3,
      averageVesselWidth: 6 + Math.random() * 4,
      arteriovenousRatio: 0.6 + Math.random() * 0.15,
      opticDiscArea: 2200 + Math.random() * 600,
      opticCupArea: 600 + Math.random() * 800,
      hemorrhageCount: Math.floor(Math.random() * 4),
      microaneurysmCount: Math.floor(Math.random() * 6),
      hardExudateArea: Math.random() * 0.08,
      softExudateCount: Math.floor(Math.random() * 3),
      imageQuality: 0.75 + Math.random() * 0.25,
      spatialFeatures: Array.from(
        { length: 1280 },
        () => Math.random() * 2 - 1
      ),
    },
    metadata: {
      processingTime: 120 + Math.random() * 60,
      imageDimensions: { width: 224, height: 224 },
      imageSize: 1024 * 1024 + Math.random() * 1024 * 1024,
      modelVersion: 'efficientnet-b0-retinal-demo-v1.0',
      preprocessingSteps: ['resize', 'normalize', 'tensor_conversion'],
      timestamp: new Date(),
      gpuAccelerated: true,
    },
  };
}
