/**
 * Speech Analysis API Route for Neuralens
 *
 * This API endpoint handles speech analysis requests from the frontend,
 * processes the results, and integrates with the NRI (Neuro-Risk Index) system.
 *
 * Key Features:
 * - Receives speech analysis results from client-side ML processing
 * - Validates and processes speech biomarkers
 * - Calculates NRI contribution from speech patterns
 * - Caches results for demo efficiency
 * - Returns structured response for frontend display
 *
 * Technical Implementation:
 * - Next.js 15 API route with TypeScript
 * - JSON request/response handling
 * - Error handling and validation
 * - Integration with NRI calculation system
 * - Performance monitoring and logging
 */

import { NextResponse, type NextRequest } from 'next/server';

import type { SpeechResult, SpeechAnalysisResponse } from '../../../types/speech-analysis';

/**
 * POST /api/speech
 * Process speech analysis results and integrate with NRI system
 */
export async function POST(request: NextRequest) {
  try {
    console.log('[API] Speech analysis request received');

    // Parse request body
    const body = await request.json();
    const { result, sessionId, timestamp } = body;

    // Validate required fields
    if (!result || !result.fluencyScore || !result.confidence || !result.biomarkers) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid speech analysis result format',
        },
        { status: 400 },
      );
    }

    // Process speech analysis result
    const processedResult = await processSpeechAnalysis(result as SpeechResult);

    // Calculate NRI contribution
    const nriContribution = calculateNRIContribution(result as SpeechResult);

    // Generate cache key for demo efficiency
    const cacheKey = `speech_${sessionId}_${Date.now()}`;

    // Store result in cache (placeholder - would use Redis in production)
    await cacheResult(cacheKey, {
      ...processedResult,
      nriContribution,
      timestamp: new Date(timestamp),
    });

    // Prepare response
    const response: SpeechAnalysisResponse = {
      result: processedResult,
      success: true,
      cacheKey,
    };

    console.log('[API] Speech analysis processed successfully:', {
      fluencyScore: processedResult.fluencyScore,
      confidence: processedResult.confidence,
      nriContribution,
      processingTime: processedResult.metadata.processingTime,
    });

    return NextResponse.json(response);
  } catch (error) {
    console.error('[API] Speech analysis error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error during speech analysis',
      },
      { status: 500 },
    );
  }
}

/**
 * GET /api/speech
 * Retrieve cached speech analysis results
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
    console.error('[API] Error retrieving cached result:', error);

    return NextResponse.json(
      { success: false, error: 'Failed to retrieve cached result' },
      { status: 500 },
    );
  }
}

/**
 * Process speech analysis result and add server-side enhancements
 */
async function processSpeechAnalysis(result: SpeechResult): Promise<SpeechResult> {
  // Add server-side processing and validation
  const processedResult: SpeechResult = {
    ...result,
    // Validate fluency score range
    fluencyScore: Math.max(0, Math.min(1, result.fluencyScore)),
    // Validate confidence range
    confidence: Math.max(0, Math.min(1, result.confidence)),
    // Add server processing timestamp
    metadata: {
      ...result.metadata,
      processingTime: Date.now() - (result.metadata?.timestamp?.getTime() || Date.now()),
      timestamp: new Date(),
      modelVersion: result.metadata?.modelVersion || 'whisper-tiny-v1.0',
      sampleRate: result.metadata?.sampleRate || 16000,
    },
  };

  // Validate biomarkers
  if (processedResult.biomarkers) {
    processedResult.biomarkers = {
      ...processedResult.biomarkers,
      // Ensure reasonable ranges for biomarkers
      speechRate: Math.max(50, Math.min(300, processedResult.biomarkers.speechRate)),
      pauseFrequency: Math.max(0, Math.min(60, processedResult.biomarkers.pauseFrequency)),
      pauseDuration: Math.max(0, Math.min(5000, processedResult.biomarkers.pauseDuration)),
      pitchVariation: Math.max(0, Math.min(1, processedResult.biomarkers.pitchVariation)),
    };
  }

  return processedResult;
}

/**
 * Calculate NRI contribution from speech analysis
 * Converts speech metrics to neurological risk assessment
 */
function calculateNRIContribution(result: SpeechResult): number {
  const { fluencyScore, confidence, biomarkers } = result;

  // Base risk calculation (inverse of fluency)
  let riskScore = (1 - fluencyScore) * 100;

  // Apply biomarker adjustments
  if (biomarkers) {
    // High pause frequency increases risk
    if (biomarkers.pauseFrequency > 15) {
      riskScore += (biomarkers.pauseFrequency - 15) * 2;
    }

    // Long pause duration increases risk
    if (biomarkers.pauseDuration > 1000) {
      riskScore += (biomarkers.pauseDuration - 1000) / 100;
    }

    // High pitch variation increases risk
    if (biomarkers.pitchVariation > 0.1) {
      riskScore += (biomarkers.pitchVariation - 0.1) * 200;
    }

    // Very slow or very fast speech increases risk
    const normalSpeechRate = 175; // WPM
    const speechRateDeviation = Math.abs(biomarkers.speechRate - normalSpeechRate);
    if (speechRateDeviation > 50) {
      riskScore += (speechRateDeviation - 50) * 0.5;
    }
  }

  // Apply confidence weighting
  const weightedScore = riskScore * confidence;

  // Ensure score is within valid range (0-100)
  return Math.round(Math.max(0, Math.min(100, weightedScore)));
}

/**
 * Cache speech analysis result
 * In production, this would use Redis or similar caching system
 */
async function cacheResult(key: string, _result: any): Promise<void> {
  // Placeholder implementation - would use Redis in production
  // For now, we'll just log the caching operation
  console.log(`[API] Caching result with key: ${key}`);

  // In a real implementation:
  // await redis.setex(key, 3600, JSON.stringify(result)); // Cache for 1 hour
}

/**
 * Retrieve cached speech analysis result
 * In production, this would use Redis or similar caching system
 */
async function getCachedResult(key: string): Promise<any | null> {
  // Placeholder implementation - would use Redis in production
  console.log(`[API] Retrieving cached result with key: ${key}`);

  // In a real implementation:
  // const cached = await redis.get(key);
  // return cached ? JSON.parse(cached) : null;

  // For now, return null (not found)
  return null;
}

/**
 * Validate speech analysis request format
 */
function _validateSpeechRequest(_body: any): boolean {
  return (
    _body &&
    _body.result &&
    typeof _body.result.fluencyScore === 'number' &&
    typeof _body.result.confidence === 'number' &&
    _body.result.biomarkers &&
    _body.result.metadata
  );
}

/**
 * Generate demo speech analysis result for testing
 * This can be used for demo purposes when the ML model is not available
 */
export function generateDemoResult(): SpeechResult {
  return {
    fluencyScore: 0.85 + (Math.random() - 0.5) * 0.2,
    confidence: 0.9 + (Math.random() - 0.5) * 0.1,
    biomarkers: {
      pauseDuration: 400 + Math.random() * 200,
      pauseFrequency: 8 + Math.random() * 4,
      tremorFrequency: Math.random() * 2,
      speechRate: 160 + Math.random() * 30,
      pitchVariation: 0.04 + Math.random() * 0.04,
      voiceQuality: {
        jitter: 0.01 + Math.random() * 0.02,
        shimmer: 0.02 + Math.random() * 0.02,
        hnr: 12 + Math.random() * 8,
      },
      mfccFeatures: Array.from({ length: 13 }, () => Math.random() * 2 - 1),
    },
    metadata: {
      processingTime: 80 + Math.random() * 40,
      audioDuration: 30,
      sampleRate: 16000,
      modelVersion: 'whisper-tiny-demo-v1.0',
      timestamp: new Date(),
    },
  };
}
