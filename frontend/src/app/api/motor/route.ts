/**
 * Motor Assessment API Route for NeuraLens
 *
 * This API endpoint handles motor assessment requests from the frontend,
 * processes movement data, and integrates with the NRI (Neuro-Risk Index) system.
 *
 * Key Features:
 * - Receives motor assessment data from client-side sensors and analysis
 * - Validates and processes movement biomarkers and tremor metrics
 * - Calculates NRI contribution from motor function patterns
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

import { NextResponse, type NextRequest } from 'next/server';
import type {
  MotorAssessmentRequest,
  MotorAssessmentResponse,
  MotorBiomarkers,
} from '../../../lib/api/types';

/**
 * POST /api/motor
 * Process motor assessment data and integrate with NRI system
 */
export async function POST(request: NextRequest) {
  try {
    console.log('[API] Motor assessment request received');

    // Parse request body
    const body = await request.json();
    const { session_id, sensor_data, assessment_type } = body as MotorAssessmentRequest;

    // Validate required fields
    if (!session_id || !sensor_data || !assessment_type) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: session_id, sensor_data, or assessment_type',
        },
        { status: 400 },
      );
    }

    // Validate assessment type
    const validTypes = ['tremor', 'finger_tapping', 'gait', 'balance'];
    if (!validTypes.includes(assessment_type)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid assessment_type. Must be one of: ${validTypes.join(', ')}`,
        },
        { status: 400 },
      );
    }

    // Process motor assessment data
    const processedResult = await processMotorAssessment({
      session_id,
      sensor_data,
      assessment_type,
    });

    // Calculate NRI contribution
    const nriContribution = calculateNRIContribution(processedResult);

    // Generate cache key for demo efficiency
    const cacheKey = `motor_${session_id}_${Date.now()}`;

    // Store result in cache (placeholder - would use Redis in production)
    await cacheResult(cacheKey, {
      ...processedResult,
      nriContribution,
      timestamp: new Date(),
    });

    console.log('[API] Motor assessment processed successfully:', {
      assessmentType: assessment_type,
      riskScore: processedResult.risk_score,
      confidence: processedResult.confidence,
      nriContribution,
      processingTime: processedResult.processing_time,
    });

    return NextResponse.json({
      success: true,
      result: processedResult,
      cacheKey,
      nriContribution,
    });
  } catch (error) {
    console.error('[API] Motor assessment error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error during motor assessment',
      },
      { status: 500 },
    );
  }
}

/**
 * GET /api/motor
 * Retrieve cached motor assessment results
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
    console.error('[API] Error retrieving cached motor result:', error);

    return NextResponse.json(
      { success: false, error: 'Failed to retrieve cached result' },
      { status: 500 },
    );
  }
}

/**
 * Process motor assessment data and add server-side enhancements
 */
async function processMotorAssessment(
  request: MotorAssessmentRequest,
): Promise<MotorAssessmentResponse> {
  const startTime = Date.now();

  // Analyze sensor data based on assessment type
  const biomarkers = analyzeSensorData(request.sensor_data, request.assessment_type);

  // Calculate overall risk score
  const riskScore = calculateMotorRiskScore(biomarkers, request.assessment_type);

  // Determine movement quality assessment
  const movementQuality = assessMovementQuality(biomarkers, request.assessment_type);

  const processingTime = Date.now() - startTime;

  return {
    session_id: request.session_id,
    risk_score: Math.round(Math.max(0, Math.min(100, riskScore))),
    confidence: calculateConfidence(biomarkers, request.sensor_data),
    processing_time: processingTime,
    biomarkers,
    assessment_type: request.assessment_type,
    movement_quality: movementQuality,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Analyze sensor data to extract motor biomarkers
 */
function analyzeSensorData(
  sensorData: MotorAssessmentRequest['sensor_data'],
  assessmentType: string,
): MotorBiomarkers {
  // Extract movement frequency from accelerometer data
  const movementFrequency = calculateMovementFrequency(sensorData.accelerometer || []);

  // Calculate amplitude variation
  const amplitudeVariation = calculateAmplitudeVariation(sensorData.accelerometer || []);

  // Assess coordination based on multi-axis data
  const coordinationIndex = calculateCoordinationIndex(sensorData);

  // Detect and quantify tremor
  const tremorSeverity = calculateTremorSeverity(sensorData.accelerometer || [], assessmentType);

  // Calculate fatigue index from movement patterns
  const fatigueIndex = calculateFatigueIndex(sensorData.accelerometer || []);

  // Assess movement asymmetry
  const asymmetryScore = calculateAsymmetryScore(sensorData);

  return {
    movement_frequency: Math.round(movementFrequency * 100) / 100,
    amplitude_variation: Math.round(amplitudeVariation * 100) / 100,
    coordination_index: Math.round(coordinationIndex * 100) / 100,
    tremor_severity: Math.round(tremorSeverity * 100) / 100,
    fatigue_index: Math.round(fatigueIndex * 100) / 100,
    asymmetry_score: Math.round(asymmetryScore * 100) / 100,
  };
}

/**
 * Calculate movement frequency from accelerometer data
 */
function calculateMovementFrequency(
  accelerometerData: Array<{ x: number; y: number; z: number }>,
): number {
  if (accelerometerData.length < 10) return 0;

  // Calculate magnitude of acceleration
  const magnitudes = accelerometerData.map(point =>
    Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
  );

  // Count peaks to estimate frequency
  let peakCount = 0;
  for (let i = 1; i < magnitudes.length - 1; i++) {
    if (magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1]) {
      peakCount++;
    }
  }

  // Normalize to Hz (assuming 100Hz sampling rate)
  return (peakCount / accelerometerData.length) * 100;
}

/**
 * Calculate amplitude variation coefficient
 */
function calculateAmplitudeVariation(
  accelerometerData: Array<{ x: number; y: number; z: number }>,
): number {
  if (accelerometerData.length < 2) return 0;

  const magnitudes = accelerometerData.map(point =>
    Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
  );

  const mean = magnitudes.reduce((sum, val) => sum + val, 0) / magnitudes.length;
  const variance =
    magnitudes.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / magnitudes.length;
  const stdDev = Math.sqrt(variance);

  return mean > 0 ? stdDev / mean : 0; // Coefficient of variation
}

/**
 * Calculate coordination index from multi-axis sensor data
 */
function calculateCoordinationIndex(sensorData: MotorAssessmentRequest['sensor_data']): number {
  const { accelerometer = [], gyroscope = [] } = sensorData;

  if (accelerometer.length < 10 || gyroscope.length < 10) return 0.5;

  // Calculate correlation between accelerometer and gyroscope data
  const accelMagnitudes = accelerometer.map(point =>
    Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
  );
  const gyroMagnitudes = gyroscope.map(point =>
    Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
  );

  // Simple correlation coefficient
  const minLength = Math.min(accelMagnitudes.length, gyroMagnitudes.length);
  let correlation = 0;

  for (let i = 0; i < minLength - 1; i++) {
    const accelChange = accelMagnitudes[i + 1] - accelMagnitudes[i];
    const gyroChange = gyroMagnitudes[i + 1] - gyroMagnitudes[i];
    correlation += Math.abs(accelChange * gyroChange);
  }

  return Math.min(1, correlation / minLength);
}

/**
 * Calculate tremor severity based on frequency analysis
 */
function calculateTremorSeverity(
  accelerometerData: Array<{ x: number; y: number; z: number }>,
  assessmentType: string,
): number {
  if (accelerometerData.length < 20) return 0;

  // Tremor frequency ranges: 4-6 Hz for Parkinson's, 8-12 Hz for essential tremor
  const tremorFrequencyRange = assessmentType === 'tremor' ? [4, 12] : [3, 8];

  const magnitudes = accelerometerData.map(point =>
    Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
  );

  // Simple frequency domain analysis (would use FFT in production)
  const samplingRate = 100; // Assumed 100Hz
  const windowSize = Math.min(magnitudes.length, samplingRate * 2); // 2-second window

  let tremorPower = 0;
  let totalPower = 0;

  for (let i = 0; i < windowSize - 1; i++) {
    const frequency = (i * samplingRate) / windowSize;
    const power = Math.abs(magnitudes[i + 1] - magnitudes[i]);

    totalPower += power;
    if (frequency >= tremorFrequencyRange[0] && frequency <= tremorFrequencyRange[1]) {
      tremorPower += power;
    }
  }

  return totalPower > 0 ? tremorPower / totalPower : 0;
}

/**
 * Calculate fatigue index from movement degradation over time
 */
function calculateFatigueIndex(
  accelerometerData: Array<{ x: number; y: number; z: number }>,
): number {
  if (accelerometerData.length < 50) return 0;

  const segmentSize = Math.floor(accelerometerData.length / 5);
  const segments = [];

  for (let i = 0; i < 5; i++) {
    const start = i * segmentSize;
    const end = Math.min(start + segmentSize, accelerometerData.length);
    const segment = accelerometerData.slice(start, end);

    const avgMagnitude =
      segment.reduce(
        (sum, point) => sum + Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
        0,
      ) / segment.length;

    segments.push(avgMagnitude);
  }

  // Calculate decline from first to last segment
  const initialPower = segments[0];
  const finalPower = segments[segments.length - 1];

  return initialPower > 0 ? Math.max(0, (initialPower - finalPower) / initialPower) : 0;
}

/**
 * Calculate movement asymmetry score
 */
function calculateAsymmetryScore(sensorData: MotorAssessmentRequest['sensor_data']): number {
  const { accelerometer = [], position = [] } = sensorData;

  if (accelerometer.length < 10) return 0;

  // Calculate asymmetry in X and Y axes (left-right, forward-backward)
  const xValues = accelerometer.map(point => point.x);
  const yValues = accelerometer.map(point => point.y);

  const xMean = xValues.reduce((sum, val) => sum + val, 0) / xValues.length;
  const yMean = yValues.reduce((sum, val) => sum + val, 0) / yValues.length;

  // Calculate deviation from center
  const xAsymmetry = Math.abs(xMean);
  const yAsymmetry = Math.abs(yMean);

  return Math.min(1, Math.sqrt(xAsymmetry * xAsymmetry + yAsymmetry * yAsymmetry));
}

/**
 * Calculate overall motor risk score
 */
function calculateMotorRiskScore(biomarkers: MotorBiomarkers, assessmentType: string): number {
  let riskScore = 0;

  // Weight factors based on assessment type
  const weights = getAssessmentWeights(assessmentType);

  // Tremor severity (higher = more risk)
  riskScore += biomarkers.tremor_severity * weights.tremor * 100;

  // Amplitude variation (higher = more risk)
  riskScore += biomarkers.amplitude_variation * weights.amplitude * 50;

  // Coordination index (lower = more risk)
  riskScore += (1 - biomarkers.coordination_index) * weights.coordination * 30;

  // Fatigue index (higher = more risk)
  riskScore += biomarkers.fatigue_index * weights.fatigue * 40;

  // Asymmetry score (higher = more risk)
  riskScore += biomarkers.asymmetry_score * weights.asymmetry * 35;

  // Movement frequency deviation from normal
  const normalFrequency = getNormalFrequency(assessmentType);
  const frequencyDeviation =
    Math.abs(biomarkers.movement_frequency - normalFrequency) / normalFrequency;
  riskScore += Math.min(frequencyDeviation, 1) * weights.frequency * 25;

  return riskScore;
}

/**
 * Get assessment-specific weight factors
 */
function getAssessmentWeights(assessmentType: string) {
  const weights = {
    tremor: {
      tremor: 0.4,
      amplitude: 0.2,
      coordination: 0.15,
      fatigue: 0.1,
      asymmetry: 0.1,
      frequency: 0.05,
    },
    finger_tapping: {
      tremor: 0.1,
      amplitude: 0.3,
      coordination: 0.25,
      fatigue: 0.2,
      asymmetry: 0.1,
      frequency: 0.05,
    },
    gait: {
      tremor: 0.15,
      amplitude: 0.2,
      coordination: 0.2,
      fatigue: 0.15,
      asymmetry: 0.25,
      frequency: 0.05,
    },
    balance: {
      tremor: 0.2,
      amplitude: 0.25,
      coordination: 0.3,
      fatigue: 0.1,
      asymmetry: 0.1,
      frequency: 0.05,
    },
  };

  return weights[assessmentType as keyof typeof weights] || weights.tremor;
}

/**
 * Get normal frequency range for assessment type
 */
function getNormalFrequency(assessmentType: string): number {
  const normalFrequencies = {
    tremor: 0.5, // Low frequency for rest
    finger_tapping: 4.0, // 4 Hz for finger tapping
    gait: 2.0, // 2 Hz for walking
    balance: 1.0, // 1 Hz for balance corrections
  };

  return normalFrequencies[assessmentType as keyof typeof normalFrequencies] || 2.0;
}

/**
 * Assess movement quality based on biomarkers
 */
function assessMovementQuality(biomarkers: MotorBiomarkers, assessmentType: string): string {
  const riskScore = calculateMotorRiskScore(biomarkers, assessmentType);

  if (riskScore < 20) return 'excellent';
  if (riskScore < 40) return 'good';
  if (riskScore < 60) return 'fair';
  if (riskScore < 80) return 'poor';
  return 'concerning';
}

/**
 * Calculate confidence based on data quality
 */
function calculateConfidence(
  biomarkers: MotorBiomarkers,
  sensorData: MotorAssessmentRequest['sensor_data'],
): number {
  let confidence = 0.9; // Base confidence

  // Reduce confidence for insufficient data
  const totalDataPoints =
    (sensorData.accelerometer?.length || 0) +
    (sensorData.gyroscope?.length || 0) +
    (sensorData.position?.length || 0);

  if (totalDataPoints < 100) confidence *= 0.7;
  if (totalDataPoints < 50) confidence *= 0.5;

  // Reduce confidence for extreme values
  if (biomarkers.tremor_severity > 0.8) confidence *= 0.9;
  if (biomarkers.coordination_index < 0.2) confidence *= 0.8;

  return Math.round(Math.max(0.1, Math.min(1.0, confidence)) * 100) / 100;
}

/**
 * Calculate NRI contribution from motor assessment
 */
function calculateNRIContribution(result: MotorAssessmentResponse): number {
  // Motor assessment contributes 25% to overall NRI
  const motorWeight = 0.25;
  const weightedScore = result.risk_score * motorWeight * result.confidence;

  return Math.round(Math.max(0, Math.min(25, weightedScore)));
}

/**
 * Cache motor assessment result
 */
async function cacheResult(key: string, _result: any): Promise<void> {
  console.log(`[API] Caching motor result with key: ${key}`);
  // In production: await redis.setex(key, 3600, JSON.stringify(result));
}

/**
 * Retrieve cached motor assessment result
 */
async function getCachedResult(key: string): Promise<any | null> {
  console.log(`[API] Retrieving cached motor result with key: ${key}`);
  // In production: const cached = await redis.get(key); return cached ? JSON.parse(cached) : null;
  return null;
}

/**
 * Generate demo motor assessment result for testing
 */
export function generateDemoMotorResult(
  assessmentType: string = 'tremor',
): MotorAssessmentResponse {
  const baseRisk = 20 + Math.random() * 40; // 20-60 range

  return {
    session_id: `demo_${Date.now()}`,
    risk_score: Math.round(baseRisk),
    confidence: 0.85 + Math.random() * 0.1,
    processing_time: 150 + Math.random() * 100,
    biomarkers: {
      movement_frequency: 2.0 + Math.random() * 2.0,
      amplitude_variation: 0.1 + Math.random() * 0.3,
      coordination_index: 0.6 + Math.random() * 0.3,
      tremor_severity: Math.random() * 0.4,
      fatigue_index: Math.random() * 0.3,
      asymmetry_score: Math.random() * 0.2,
    },
    assessment_type: assessmentType,
    movement_quality: baseRisk < 30 ? 'good' : baseRisk < 50 ? 'fair' : 'concerning',
    timestamp: new Date().toISOString(),
  };
}
