/**
 * Enhanced Speech Analysis Types
 * Matches backend EnhancedSpeechAnalysisResponse schema
 */

// Individual biomarker with clinical metadata
export interface BiomarkerResult {
    value: number;
    unit: string;
    normal_range: [number, number];
    is_estimated: boolean;
    confidence: number | null;
}

// All 9 biomarkers from backend
export interface EnhancedBiomarkers {
    jitter: BiomarkerResult;
    shimmer: BiomarkerResult;
    hnr: BiomarkerResult;
    speech_rate: BiomarkerResult;
    pause_ratio: BiomarkerResult;
    fluency_score: BiomarkerResult;
    voice_tremor: BiomarkerResult;
    articulation_clarity: BiomarkerResult;
    prosody_variation: BiomarkerResult;
}

// Baseline comparison for tracking changes
export interface BaselineComparison {
    biomarker_name: string;
    current_value: number;
    baseline_value: number;
    delta: number;
    delta_percent: number;
    direction: 'improved' | 'worsened' | 'stable';
}

// Enhanced speech analysis response from backend
export interface EnhancedSpeechAnalysisResponse {
    session_id: string;
    processing_time: number;
    timestamp: string;
    confidence: number;
    risk_score: number;
    quality_score: number;
    biomarkers: EnhancedBiomarkers;
    legacy_biomarkers?: Record<string, unknown>;
    file_info?: {
        filename?: string;
        size?: number;
        content_type?: string;
        duration?: number;
        sample_rate?: number;
        resampled?: boolean;
    };
    recommendations: string[];
    baseline_comparisons?: BaselineComparison[];
    status: 'completed' | 'partial' | 'error';
    error_message?: string;
}

// Biomarker display configuration
export interface BiomarkerDisplayConfig {
    key: keyof EnhancedBiomarkers;
    label: string;
    description: string;
    icon: string;
    higherIsBetter: boolean;
    formatValue: (value: number, unit: string) => string;
}

// Biomarker display configurations
export const BIOMARKER_CONFIGS: BiomarkerDisplayConfig[] = [
    {
        key: 'fluency_score',
        label: 'Fluency',
        description: 'Speech fluency and rhythm measure',
        icon: 'waves',
        higherIsBetter: true,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    },
    {
        key: 'articulation_clarity',
        label: 'Articulation',
        description: 'Clarity of speech articulation',
        icon: 'message-circle',
        higherIsBetter: true,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    },
    {
        key: 'speech_rate',
        label: 'Speech Rate',
        description: 'Speaking speed in syllables per second',
        icon: 'gauge',
        higherIsBetter: false, // Normal range is best
        formatValue: (v, u) => `${v.toFixed(1)} ${u}`,
    },
    {
        key: 'voice_tremor',
        label: 'Voice Tremor',
        description: 'Tremor intensity in voice',
        icon: 'activity',
        higherIsBetter: false,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    },
    {
        key: 'jitter',
        label: 'Jitter',
        description: 'Fundamental frequency variation',
        icon: 'zap',
        higherIsBetter: false,
        formatValue: (v) => `${(v * 100).toFixed(2)}%`,
    },
    {
        key: 'shimmer',
        label: 'Shimmer',
        description: 'Amplitude variation in voice',
        icon: 'bar-chart-2',
        higherIsBetter: false,
        formatValue: (v) => `${(v * 100).toFixed(2)}%`,
    },
    {
        key: 'hnr',
        label: 'HNR',
        description: 'Harmonics-to-Noise Ratio',
        icon: 'volume-2',
        higherIsBetter: true,
        formatValue: (v, u) => `${v.toFixed(1)} ${u}`,
    },
    {
        key: 'pause_ratio',
        label: 'Pause Ratio',
        description: 'Proportion of silence in speech',
        icon: 'pause',
        higherIsBetter: false, // Normal range is best
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    },
    {
        key: 'prosody_variation',
        label: 'Prosody',
        description: 'Prosodic richness and variation',
        icon: 'music',
        higherIsBetter: true,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    },
];

// Helper to check if value is within normal range
export function isWithinNormalRange(biomarker: BiomarkerResult): boolean {
    const [min, max] = biomarker.normal_range;
    return biomarker.value >= min && biomarker.value <= max;
}

// Helper to get status based on value and normal range
export function getBiomarkerStatus(
    biomarker: BiomarkerResult,
    higherIsBetter: boolean
): 'normal' | 'warning' | 'abnormal' {
    const [min, max] = biomarker.normal_range;
    const { value } = biomarker;

    if (value >= min && value <= max) return 'normal';

    // Check how far outside the range
    const deviation = value < min ? (min - value) / min : (value - max) / max;

    if (deviation > 0.5) return 'abnormal';
    return 'warning';
}
