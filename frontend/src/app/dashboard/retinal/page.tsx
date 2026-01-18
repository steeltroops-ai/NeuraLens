'use client';

/**
 * Retinal Analysis Page - Layered Pipeline Architecture v4.0
 * 
 * Fully aligned with backend response structure.
 * Features:
 * - Clear data flow visualization
 * - Pipeline stage tracking
 * - Comprehensive error display
 * - Console logging for debugging
 */

import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Eye,
    Info,
    Shield,
    Clock,
    Target,
    AlertCircle,
    Loader2,
    Upload,
    ImageIcon,
    Brain,
    Activity,
    CheckCircle2,
    AlertTriangle,
    XCircle,
    Download,
    RefreshCw,
    Layers,
    Droplet,
    FileText,
    Stethoscope,
    TrendingUp,
    ChevronRight,
    Terminal,
    HeartPulse,
} from 'lucide-react';
import { ExplanationPanel } from '@/components/explanation/ExplanationPanel';

// ============================================================================
// Types matching backend v4.0 structure
// ============================================================================

interface PipelineError {
    stage: string;
    error_type: string;
    message: string;
    timestamp: string;
}

interface PipelineState {
    session_id: string;
    current_stage: string;
    stages_completed: string[];
    stages_timing_ms: Record<string, number>;
    errors: PipelineError[];
    warnings: string[];
    started_at: string;
    completed_at?: string;
}

interface BiomarkerValue {
    value: number;
    normal_range?: number[];
    threshold?: number;
    status: 'normal' | 'abnormal' | 'borderline';
    measurement_confidence: number;
    clinical_significance?: string;
    source?: string;
}

interface VesselBiomarkers {
    tortuosity_index: BiomarkerValue;
    av_ratio: BiomarkerValue;
    vessel_density: BiomarkerValue;
    fractal_dimension: BiomarkerValue;
    branching_coefficient: BiomarkerValue;
}

interface OpticDiscBiomarkers {
    cup_disc_ratio: BiomarkerValue;
    disc_area_mm2: BiomarkerValue;
    rim_area_mm2: BiomarkerValue;
    rnfl_thickness: BiomarkerValue;
    notching_detected: boolean;
}

interface MacularBiomarkers {
    thickness: BiomarkerValue;
    volume: BiomarkerValue;
}

interface LesionBiomarkers {
    hemorrhage_count: BiomarkerValue;
    microaneurysm_count: BiomarkerValue;
    exudate_area_percent: BiomarkerValue;
    cotton_wool_spots: number;
    neovascularization_detected: boolean;
    venous_beading_detected: boolean;
    irma_detected: boolean;
}

interface CompleteBiomarkers {
    vessels: VesselBiomarkers;
    optic_disc: OpticDiscBiomarkers;
    macula: MacularBiomarkers;
    lesions: LesionBiomarkers;
}

interface DiabeticRetinopathyResult {
    grade: number;
    grade_name: string;
    probability: number;
    probabilities_all_grades: Record<string, number>;
    referral_urgency: string;
    clinical_action: string;
    macular_edema_present: boolean;
    clinically_significant_macular_edema: boolean;
}

interface RiskAssessment {
    overall_score: number;
    category: string;
    confidence: number;
    confidence_interval_95: [number, number];
    primary_finding: string;
    contributing_factors: Record<string, number>;
    systemic_risk_indicators: Record<string, string>;
}

interface ClinicalFinding {
    finding_type: string;
    anatomical_location: string;
    severity: string;
    description: string;
    clinical_relevance: string;
    icd10_code?: string;
    requires_referral: boolean;
    confidence: number;
}

interface DifferentialDiagnosis {
    diagnosis: string;
    probability: number;
    supporting_evidence: string[];
    icd10_code: string;
}

interface ImageQuality {
    overall_score: number;
    gradability: string;
    is_gradable: boolean;
    issues: string[];
    snr_db: number;
    focus_score: number;
    illumination_score: number;
    contrast_score: number;
    optic_disc_visible: boolean;
    macula_visible: boolean;
    resolution: [number, number];
    file_size_mb: number;
}

interface RetinalAnalysisResponse {
    success: boolean;
    session_id: string;
    patient_id: string;
    pipeline_state: PipelineState;
    timestamp: string;
    total_processing_time_ms: number;
    model_version: string;
    image_quality: ImageQuality;
    biomarkers: CompleteBiomarkers;
    diabetic_retinopathy: DiabeticRetinopathyResult;
    risk_assessment: RiskAssessment;
    findings: ClinicalFinding[];
    differential_diagnoses: DifferentialDiagnosis[];
    recommendations: string[];
    clinical_summary: string;
    heatmap_base64?: string;
    segmentation_base64?: string;
}

type AnalysisState = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

// Pipeline stages for display
const PIPELINE_STAGES = [
    { key: 'input_validation', label: 'Input', icon: Upload },
    { key: 'quality_assessment', label: 'Quality', icon: Target },
    { key: 'vessel_analysis', label: 'Vessels', icon: Activity },
    { key: 'optic_disc_analysis', label: 'Disc', icon: Eye },
    { key: 'lesion_detection', label: 'Lesions', icon: AlertTriangle },
    { key: 'dr_grading', label: 'DR Grade', icon: Stethoscope },
    { key: 'risk_calculation', label: 'Risk', icon: TrendingUp },
    { key: 'clinical_assessment', label: 'Clinical', icon: FileText },
    { key: 'heatmap_generation', label: 'Heatmap', icon: Layers },
    { key: 'output_formatting', label: 'Output', icon: Download },
];

// ============================================================================
// Pipeline Stage Indicator Component  
// ============================================================================

function PipelineStageIndicator({ pipelineState, isProcessing }: { pipelineState?: PipelineState; isProcessing: boolean }) {
    const completedStages = pipelineState?.stages_completed || [];
    const currentStage = pipelineState?.current_stage || '';

    return (
        <div className="bg-zinc-900 rounded-xl border border-zinc-700 p-4">
            <div className="flex items-center gap-2 mb-3">
                <Terminal className="h-4 w-4 text-cyan-400" />
                <span className="text-[11px] font-mono text-cyan-400">PIPELINE v4.0</span>
                {pipelineState?.session_id && (
                    <span className="text-[9px] font-mono text-zinc-500 ml-auto">
                        {pipelineState.session_id.slice(0, 8)}
                    </span>
                )}
            </div>

            {/* Stage indicators */}
            <div className="flex items-center gap-1 flex-wrap">
                {PIPELINE_STAGES.map((stage, idx) => {
                    const isCompleted = completedStages.includes(stage.key);
                    const isCurrent = currentStage === stage.key;
                    const Icon = stage.icon;
                    const timing = pipelineState?.stages_timing_ms?.[stage.key];

                    return (
                        <React.Fragment key={stage.key}>
                            <div className={`flex items-center gap-1 px-2 py-1 rounded text-[9px] font-mono ${isCompleted ? 'bg-green-500/20 text-green-400' :
                                    isCurrent ? 'bg-cyan-500/20 text-cyan-400 animate-pulse' :
                                        'bg-zinc-800 text-zinc-500'
                                }`}>
                                {isCompleted ? <CheckCircle2 className="h-3 w-3" /> :
                                    isCurrent ? <Loader2 className="h-3 w-3 animate-spin" /> :
                                        <Icon className="h-3 w-3" />}
                                <span className="hidden sm:inline">{stage.label}</span>
                                {timing && <span className="text-[8px] opacity-60">{timing.toFixed(0)}ms</span>}
                            </div>
                            {idx < PIPELINE_STAGES.length - 1 && (
                                <ChevronRight className={`h-3 w-3 flex-shrink-0 ${isCompleted ? 'text-green-400' : 'text-zinc-600'
                                    }`} />
                            )}
                        </React.Fragment>
                    );
                })}
            </div>

            {/* Errors */}
            {pipelineState?.errors && pipelineState.errors.length > 0 && (
                <div className="mt-3 p-2 bg-red-500/10 rounded border border-red-500/30">
                    <div className="text-[10px] font-mono text-red-400 mb-1">ERRORS:</div>
                    {pipelineState.errors.map((err, idx) => (
                        <div key={idx} className="text-[10px] font-mono text-red-300">
                            [{err.stage}] {err.error_type}: {err.message}
                        </div>
                    ))}
                </div>
            )}

            {/* Warnings */}
            {pipelineState?.warnings && pipelineState.warnings.length > 0 && (
                <div className="mt-2 p-2 bg-yellow-500/10 rounded border border-yellow-500/30">
                    <div className="text-[10px] font-mono text-yellow-400 mb-1">WARNINGS:</div>
                    {pipelineState.warnings.map((warn, idx) => (
                        <div key={idx} className="text-[10px] font-mono text-yellow-300">{warn}</div>
                    ))}
                </div>
            )}
        </div>
    );
}

// ============================================================================
// Biomarker Display Card
// ============================================================================

function BiomarkerCard({ label, biomarker, icon: Icon }: {
    label: string;
    biomarker: BiomarkerValue;
    icon: React.ElementType
}) {
    const colors = {
        normal: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', badge: 'bg-green-100' },
        borderline: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', badge: 'bg-yellow-100' },
        abnormal: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100' },
    }[biomarker.status] || { bg: 'bg-zinc-50', border: 'border-zinc-200', text: 'text-zinc-700', badge: 'bg-zinc-100' };

    const formatValue = (v: number) => {
        if (v === Math.floor(v)) return v.toString();
        return v.toFixed(3);
    };

    return (
        <div className={`${colors.bg} ${colors.border} border rounded-lg p-2`}>
            <div className="flex items-center gap-1.5 mb-1">
                <Icon className={`h-3 w-3 ${colors.text}`} />
                <span className="text-[10px] font-medium text-zinc-700 truncate">{label}</span>
            </div>
            <div className="text-[14px] font-bold text-zinc-900">{formatValue(biomarker.value)}</div>
            <div className="flex items-center justify-between">
                <span className={`text-[9px] font-medium ${colors.text} capitalize`}>{biomarker.status}</span>
                <span className="text-[8px] text-zinc-400">{(biomarker.measurement_confidence * 100).toFixed(0)}%</span>
            </div>
            {biomarker.clinical_significance && (
                <div className="text-[8px] text-zinc-500 mt-1 line-clamp-2">{biomarker.clinical_significance}</div>
            )}
        </div>
    );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function RetinalAnalysisPage() {
    const [state, setState] = useState<AnalysisState>('idle');
    const [results, setResults] = useState<RetinalAnalysisResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [showHeatmap, setShowHeatmap] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const addLog = useCallback((message: string) => {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        const logEntry = `[${timestamp}] ${message}`;
        console.log(`[RETINAL] ${message}`);
        setLogs(prev => [...prev.slice(-20), logEntry]);
    }, []);

    const analyzeImage = useCallback(async (imageFile: File) => {
        setState('processing');
        setError(null);
        addLog(`Starting analysis for: ${imageFile.name} (${(imageFile.size / 1024 / 1024).toFixed(2)}MB)`);

        try {
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('patient_id', 'ANONYMOUS');

            addLog('Sending to backend API...');
            const response = await fetch('/api/retinal/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`API error ${response.status}: ${errText}`);
            }

            const data: RetinalAnalysisResponse = await response.json();
            addLog(`Response received: success=${data.success}`);

            if (data.pipeline_state) {
                addLog(`Stages completed: ${data.pipeline_state.stages_completed.length}`);
                for (const stage of data.pipeline_state.stages_completed) {
                    const timing = data.pipeline_state.stages_timing_ms[stage];
                    addLog(`  [OK] ${stage}${timing ? ` (${timing.toFixed(1)}ms)` : ''}`);
                }
            }

            if (data.diabetic_retinopathy) {
                addLog(`DR Grade: ${data.diabetic_retinopathy.grade} - ${data.diabetic_retinopathy.grade_name}`);
            }

            if (data.risk_assessment) {
                addLog(`Risk Score: ${data.risk_assessment.overall_score.toFixed(1)} (${data.risk_assessment.category})`);
            }

            setResults(data);
            setState(data.success ? 'complete' : 'error');

            if (!data.success) {
                setError('Analysis failed - check pipeline errors');
            }
        } catch (err) {
            const msg = err instanceof Error ? err.message : 'Unknown error';
            addLog(`ERROR: ${msg}`);
            setError(msg);
            setState('error');
        }
    }, [addLog]);

    const handleFileSelect = useCallback((file: File) => {
        addLog(`File selected: ${file.name}`);

        if (!file.type.startsWith('image/')) {
            addLog('ERROR: Invalid file type');
            setError('Please select an image file');
            return;
        }
        if (file.size > 15 * 1024 * 1024) {
            addLog('ERROR: File too large');
            setError('File must be less than 15MB');
            return;
        }
        setPreviewUrl(URL.createObjectURL(file));
        analyzeImage(file);
    }, [analyzeImage, addLog]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    }, [handleFileSelect]);

    const handleReset = useCallback(() => {
        addLog('Resetting...');
        setState('idle');
        setResults(null);
        setError(null);
        setShowHeatmap(false);
        setLogs([]);
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
    }, [previewUrl, addLog]);

    const isProcessing = state === 'processing' || state === 'uploading';

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
            {/* Header */}
            <div className="bg-white rounded-xl border border-zinc-200 p-5">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600">
                        <Eye className="h-6 w-6 text-white" />
                    </div>
                    <div className="flex-1">
                        <div className="flex items-center gap-2">
                            <h1 className="text-[18px] font-bold text-zinc-900">Retinal Analysis</h1>
                            <span className="px-2 py-0.5 bg-cyan-100 text-cyan-700 text-[10px] font-mono rounded">
                                v4.0 MODULAR
                            </span>
                        </div>
                        <p className="text-[12px] text-zinc-500 mt-1">
                            12 biomarkers | ICDR grading | Multi-factorial risk | Evidence-based citations
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-4 gap-2 mt-4">
                    {[
                        { icon: Target, label: '93% DR', sub: 'Accuracy' },
                        { icon: Clock, label: '<2s', sub: 'Processing' },
                        { icon: Brain, label: '12', sub: 'Biomarkers' },
                        { icon: Shield, label: 'ETDRS', sub: 'Standards' },
                    ].map((stat, i) => (
                        <div key={i} className="flex items-center gap-2 p-2 bg-zinc-50 rounded-lg">
                            <stat.icon className="h-4 w-4 text-zinc-500" />
                            <div>
                                <div className="text-[12px] font-semibold text-zinc-900">{stat.label}</div>
                                <div className="text-[9px] text-zinc-500">{stat.sub}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Pipeline State */}
            {(isProcessing || results) && (
                <PipelineStageIndicator
                    pipelineState={results?.pipeline_state}
                    isProcessing={isProcessing}
                />
            )}

            {/* Main Content */}
            <AnimatePresence mode="wait">
                {state === 'complete' && results?.success ? (
                    <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                        {/* Summary */}
                        {results.clinical_summary && (
                            <div className="bg-blue-50 rounded-xl border border-blue-200 p-4">
                                <div className="flex items-start gap-3">
                                    <FileText className="h-5 w-5 text-blue-600" />
                                    <div>
                                        <div className="text-[12px] font-semibold text-blue-800 mb-1">Clinical Summary</div>
                                        <div className="text-[12px] text-blue-700">{results.clinical_summary}</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Results Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            {/* Left: Image & DR */}
                            <div className="space-y-4">
                                {previewUrl && (
                                    <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-[12px] font-semibold text-zinc-700">Fundus Image</span>
                                            <button
                                                onClick={() => setShowHeatmap(!showHeatmap)}
                                                className={`px-2 py-1 text-[10px] font-medium rounded ${showHeatmap ? 'bg-cyan-600 text-white' : 'bg-zinc-100 text-zinc-600'}`}
                                            >
                                                {showHeatmap ? 'Heatmap ON' : 'Show Heatmap'}
                                            </button>
                                        </div>
                                        <div className="relative aspect-square bg-zinc-900 rounded-lg overflow-hidden">
                                            <img src={showHeatmap && results.heatmap_base64 ? `data:image/png;base64,${results.heatmap_base64}` : previewUrl}
                                                alt="Fundus" className="absolute inset-0 w-full h-full object-contain" />
                                        </div>
                                    </div>
                                )}

                                {results.diabetic_retinopathy && (
                                    <div className={`rounded-xl border p-4 ${results.diabetic_retinopathy.grade === 0 ? 'bg-green-50 border-green-200' :
                                            results.diabetic_retinopathy.grade <= 2 ? 'bg-yellow-50 border-yellow-200' :
                                                'bg-red-50 border-red-200'
                                        }`}>
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-[12px] font-semibold text-zinc-700">Diabetic Retinopathy</span>
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${results.diabetic_retinopathy.grade === 0 ? 'bg-green-100 text-green-700' :
                                                    results.diabetic_retinopathy.grade <= 2 ? 'bg-yellow-100 text-yellow-700' :
                                                        'bg-red-100 text-red-700'
                                                }`}>Grade {results.diabetic_retinopathy.grade}</span>
                                        </div>
                                        <div className="text-[16px] font-bold text-zinc-900">{results.diabetic_retinopathy.grade_name}</div>
                                        <div className="text-[11px] text-zinc-600 mt-1">Prob: {(results.diabetic_retinopathy.probability * 100).toFixed(0)}%</div>
                                        <div className="text-[10px] text-zinc-500 mt-2">{results.diabetic_retinopathy.clinical_action}</div>
                                    </div>
                                )}

                                {results.risk_assessment && (
                                    <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                        <div className="text-[12px] font-semibold text-zinc-700 mb-2">Risk Assessment</div>
                                        <div className="text-center">
                                            <div className={`text-[28px] font-bold ${results.risk_assessment.category === 'minimal' || results.risk_assessment.category === 'low' ? 'text-green-600' :
                                                    results.risk_assessment.category === 'moderate' ? 'text-yellow-600' :
                                                        'text-red-600'
                                                }`}>{results.risk_assessment.overall_score.toFixed(0)}</div>
                                            <div className="text-[11px] text-zinc-500 capitalize">{results.risk_assessment.category} Risk</div>
                                            <div className="text-[9px] text-zinc-400 mt-1">
                                                95% CI: {results.risk_assessment.confidence_interval_95[0].toFixed(0)} - {results.risk_assessment.confidence_interval_95[1].toFixed(0)}
                                            </div>
                                        </div>
                                        <div className="mt-3 text-[10px] text-zinc-600">
                                            <strong>Primary:</strong> {results.risk_assessment.primary_finding}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Center: Biomarkers */}
                            <div className="space-y-4">
                                {results.biomarkers && (
                                    <>
                                        {/* Vessel Biomarkers */}
                                        <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                            <div className="text-[12px] font-semibold text-zinc-700 mb-3 flex items-center gap-2">
                                                <Activity className="h-4 w-4 text-cyan-600" />
                                                Vessel Biomarkers
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <BiomarkerCard label="Tortuosity" biomarker={results.biomarkers.vessels.tortuosity_index} icon={Activity} />
                                                <BiomarkerCard label="AV Ratio" biomarker={results.biomarkers.vessels.av_ratio} icon={Target} />
                                                <BiomarkerCard label="Density" biomarker={results.biomarkers.vessels.vessel_density} icon={Droplet} />
                                                <BiomarkerCard label="Fractal D" biomarker={results.biomarkers.vessels.fractal_dimension} icon={Layers} />
                                            </div>
                                        </div>

                                        {/* Optic Disc */}
                                        <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                            <div className="text-[12px] font-semibold text-zinc-700 mb-3 flex items-center gap-2">
                                                <Eye className="h-4 w-4 text-blue-600" />
                                                Optic Disc
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <BiomarkerCard label="CDR" biomarker={results.biomarkers.optic_disc.cup_disc_ratio} icon={Eye} />
                                                <BiomarkerCard label="RNFL" biomarker={results.biomarkers.optic_disc.rnfl_thickness} icon={Brain} />
                                            </div>
                                        </div>

                                        {/* Lesions */}
                                        <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                            <div className="text-[12px] font-semibold text-zinc-700 mb-3 flex items-center gap-2">
                                                <AlertCircle className="h-4 w-4 text-red-600" />
                                                Lesions (DR Markers)
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <BiomarkerCard label="Hemorrhages" biomarker={results.biomarkers.lesions.hemorrhage_count} icon={XCircle} />
                                                <BiomarkerCard label="Microaneurysms" biomarker={results.biomarkers.lesions.microaneurysm_count} icon={AlertTriangle} />
                                                <BiomarkerCard label="Exudates %" biomarker={results.biomarkers.lesions.exudate_area_percent} icon={Layers} />
                                                <BiomarkerCard label="Macula" biomarker={results.biomarkers.macula.thickness} icon={HeartPulse} />
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Right: Findings & Recommendations */}
                            <div className="space-y-4">
                                {results.findings.length > 0 && (
                                    <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                        <div className="text-[12px] font-semibold text-zinc-700 mb-2">Clinical Findings</div>
                                        <div className="space-y-2">
                                            {results.findings.map((f, i) => (
                                                <div key={i} className="flex items-start gap-2 p-2 bg-zinc-50 rounded">
                                                    {f.severity === 'normal' ? <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" /> :
                                                        f.severity === 'mild' ? <AlertTriangle className="h-4 w-4 text-yellow-500 flex-shrink-0" /> :
                                                            <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />}
                                                    <div className="min-w-0">
                                                        <div className="text-[11px] font-medium text-zinc-800">{f.finding_type}</div>
                                                        <div className="text-[10px] text-zinc-500">{f.description}</div>
                                                        {f.icd10_code && <span className="text-[9px] font-mono text-blue-600">{f.icd10_code}</span>}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {results.recommendations.length > 0 && (
                                    <div className="bg-blue-50 rounded-xl border border-blue-200 p-4">
                                        <div className="text-[12px] font-semibold text-blue-800 mb-2">Recommendations</div>
                                        <ul className="space-y-1">
                                            {results.recommendations.map((rec, i) => (
                                                <li key={i} className="flex items-start gap-2 text-[11px] text-blue-700">
                                                    <CheckCircle2 className="h-3 w-3 text-blue-500 flex-shrink-0 mt-0.5" />
                                                    <span>{rec}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {results.differential_diagnoses && results.differential_diagnoses.length > 0 && (
                                    <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                        <div className="text-[12px] font-semibold text-zinc-700 mb-2">Differential Diagnoses</div>
                                        <div className="space-y-2">
                                            {results.differential_diagnoses.map((dx, i) => (
                                                <div key={i} className="p-2 bg-zinc-50 rounded">
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-[11px] font-medium text-zinc-800">{dx.diagnosis}</span>
                                                        <span className="text-[10px] text-zinc-500">{(dx.probability * 100).toFixed(0)}%</span>
                                                    </div>
                                                    <div className="text-[9px] font-mono text-blue-600">{dx.icd10_code}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* AI Explanation */}
                                <ExplanationPanel
                                    data={{
                                        diabetic_retinopathy: results.diabetic_retinopathy,
                                        risk_assessment: results.risk_assessment,
                                        biomarkers: results.biomarkers,
                                        findings: results.findings,
                                        recommendations: results.recommendations,
                                    }}
                                    pipelineType="retinal"
                                    autoTrigger={false}
                                />
                            </div>
                        </div>

                        {/* Reset Button */}
                        <div className="flex justify-center">
                            <button
                                onClick={handleReset}
                                className="flex items-center gap-2 px-4 py-2 bg-zinc-100 text-zinc-700 rounded-lg hover:bg-zinc-200 transition"
                            >
                                <RefreshCw className="h-4 w-4" />
                                Analyze Another Image
                            </button>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {/* Upload Zone */}
                        <div
                            className={`bg-white rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition ${isDragging ? 'border-cyan-500 bg-cyan-50' :
                                    error ? 'border-red-300 bg-red-50' :
                                        'border-zinc-200 hover:border-cyan-400'
                                }`}
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={() => setIsDragging(false)}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                className="hidden"
                                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                            />

                            {isProcessing ? (
                                <div className="py-8">
                                    <Loader2 className="h-12 w-12 text-cyan-600 mx-auto animate-spin" />
                                    <div className="mt-4 text-[14px] font-medium text-zinc-700">Analyzing...</div>
                                    <div className="mt-1 text-[11px] text-zinc-500">Processing through layered pipeline</div>
                                </div>
                            ) : error ? (
                                <div className="py-8">
                                    <XCircle className="h-12 w-12 text-red-500 mx-auto" />
                                    <div className="mt-4 text-[14px] font-medium text-red-700">{error}</div>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); handleReset(); }}
                                        className="mt-2 text-[12px] text-red-600 underline"
                                    >
                                        Try again
                                    </button>
                                </div>
                            ) : (
                                <div className="py-8">
                                    <Upload className="h-12 w-12 text-zinc-400 mx-auto" />
                                    <div className="mt-4 text-[14px] font-medium text-zinc-700">
                                        Drop fundus image here or click to browse
                                    </div>
                                    <div className="mt-1 text-[11px] text-zinc-500">
                                        JPEG, PNG up to 15MB | Min 512x512
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Console */}
                        <div className="bg-zinc-900 rounded-xl border border-zinc-700 p-4 max-h-[400px] overflow-auto">
                            <div className="flex items-center gap-2 mb-3">
                                <Terminal className="h-4 w-4 text-green-400" />
                                <span className="text-[11px] font-mono text-green-400">Console</span>
                            </div>
                            <div className="space-y-1 font-mono text-[10px]">
                                {logs.length === 0 ? (
                                    <div className="text-zinc-500">Waiting for image...</div>
                                ) : (
                                    logs.map((log, i) => (
                                        <div key={i} className={
                                            log.includes('ERROR') ? 'text-red-400' :
                                                log.includes('[OK]') ? 'text-green-400' :
                                                    log.includes('Starting') || log.includes('Sending') ? 'text-cyan-400' :
                                                        'text-zinc-400'
                                        }>{log}</div>
                                    ))
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}
