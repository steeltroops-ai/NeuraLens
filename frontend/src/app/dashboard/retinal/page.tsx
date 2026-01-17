'use client';

/**
 * Retinal Imaging Module Page - PRD v2.0.0 Compliant
 * 
 * Complete implementation matching PRD Section 6 Frontend Integration:
 * - Image upload zone with drag-and-drop & preview
 * - Quality check feedback
 * - DR grade display with color-coded badge
 * - Biomarker cards with status indicators (8 total)
 * - Risk gauge (0-100)
 * - Urgency indicator
 * - Heatmap overlay with toggle
 * - Recommendations panel
 * - Findings list
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
    ZoomIn,
    Layers,
    Heart,
    Droplet,
} from 'lucide-react';
import { ExplanationPanel } from '@/components/explanation/ExplanationPanel';

// ============================================================================
// PRD-Compliant Types (Section 5 Response Schema)
// ============================================================================

interface BiomarkerValue {
    value: number;
    normal_range?: number[];
    threshold?: number;
    status: 'normal' | 'abnormal' | 'borderline';
}

interface DiabeticRetinopathy {
    grade: number;
    grade_name: string;
    probability: number;
    referral_urgency: string;
}

interface Finding {
    type: string;
    location: string;
    severity: 'normal' | 'mild' | 'moderate' | 'severe';
    description: string;
}

interface ImageQuality {
    score: number;
    issues: string[];
    usable: boolean;
}

interface RiskAssessment {
    overall_score: number;
    category: string;
    confidence: number;
    primary_finding: string;
}

interface RetinalAnalysisResponse {
    success: boolean;
    session_id: string;
    timestamp: string;
    processing_time_ms: number;
    risk_assessment: RiskAssessment;
    diabetic_retinopathy: DiabeticRetinopathy;
    biomarkers: Record<string, BiomarkerValue>;
    findings: Finding[];
    heatmap_base64?: string;
    image_quality: ImageQuality;
    recommendations: string[];
}

type AnalysisState = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

// ============================================================================
// Biomarker Display Configuration (PRD Section 4)
// ============================================================================

const BIOMARKER_CONFIG: Record<string, { 
    displayName: string; 
    icon: React.ElementType;
    unit: string;
    format: (v: number) => string;
}> = {
    vessel_tortuosity: { displayName: "Vessel Tortuosity", icon: Activity, unit: "index", format: (v) => v.toFixed(3) },
    av_ratio: { displayName: "AV Ratio", icon: Target, unit: "ratio", format: (v) => v.toFixed(2) },
    cup_disc_ratio: { displayName: "Cup-to-Disc Ratio", icon: Eye, unit: "ratio", format: (v) => v.toFixed(2) },
    vessel_density: { displayName: "Vessel Density", icon: Droplet, unit: "index", format: (v) => v.toFixed(2) },
    hemorrhage_count: { displayName: "Hemorrhages", icon: XCircle, unit: "count", format: (v) => v.toFixed(0) },
    microaneurysm_count: { displayName: "Microaneurysms", icon: AlertTriangle, unit: "count", format: (v) => v.toFixed(0) },
    exudate_area: { displayName: "Exudate Area", icon: Layers, unit: "%", format: (v) => v.toFixed(2) + "%" },
    rnfl_thickness: { displayName: "RNFL Thickness", icon: Brain, unit: "norm", format: (v) => v.toFixed(2) },
};

// DR Grade colors (PRD Section 6)
const DR_GRADE_COLORS: Record<number, { bg: string; text: string; border: string }> = {
    0: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200' },
    1: { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200' },
    2: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
    3: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
    4: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-300' },
};

// ============================================================================
// Component: Risk Gauge (PRD Section 6)
// ============================================================================

function RiskGauge({ score, category, confidence }: { score: number; category: string; confidence: number }) {
    const angle = (score / 100) * 180 - 90; // -90 to 90 degrees
    
    const getCategoryColor = () => {
        switch (category) {
            case 'minimal': return 'text-green-600';
            case 'low': return 'text-green-500';
            case 'moderate': return 'text-yellow-500';
            case 'elevated': return 'text-orange-500';
            case 'high': return 'text-red-500';
            case 'critical': return 'text-red-700';
            default: return 'text-zinc-500';
        }
    };

    return (
        <div className="flex flex-col items-center p-6 bg-white rounded-xl border border-zinc-200">
            <h3 className="text-[14px] font-semibold text-zinc-700 mb-4">Risk Assessment</h3>
            
            {/* Semi-circular gauge */}
            <div className="relative w-48 h-24 overflow-hidden">
                <svg className="w-48 h-48 -mt-24" viewBox="0 0 100 50">
                    {/* Background arc */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke="#e5e7eb"
                        strokeWidth="8"
                        strokeLinecap="round"
                    />
                    {/* Colored arc based on score */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke="url(#gradient)"
                        strokeWidth="8"
                        strokeLinecap="round"
                        strokeDasharray={`${(score / 100) * 125.6} 125.6`}
                    />
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#22c55e" />
                            <stop offset="50%" stopColor="#eab308" />
                            <stop offset="100%" stopColor="#ef4444" />
                        </linearGradient>
                    </defs>
                </svg>
                {/* Score display */}
                <div className="absolute inset-0 flex flex-col items-center justify-end pb-2">
                    <span className={`text-3xl font-bold ${getCategoryColor()}`}>{score.toFixed(0)}</span>
                </div>
            </div>
            
            <div className="text-center mt-2">
                <span className={`text-[14px] font-semibold capitalize ${getCategoryColor()}`}>
                    {category} Risk
                </span>
                <div className="text-[11px] text-zinc-500 mt-1">
                    Confidence: {(confidence * 100).toFixed(0)}%
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// Component: DR Grade Badge (PRD Section 6)
// ============================================================================

function DRGradeBadge({ dr }: { dr: DiabeticRetinopathy }) {
    const colors = DR_GRADE_COLORS[dr.grade] || DR_GRADE_COLORS[0];
    
    const getUrgencyLabel = () => {
        switch (dr.referral_urgency) {
            case 'routine_12_months': return 'Routine (12 mo)';
            case 'monitor_6_months': return 'Monitor (6 mo)';
            case 'refer_1_month': return 'Refer (1 mo)';
            case 'urgent_1_week': return 'URGENT (1 wk)';
            default: return dr.referral_urgency;
        }
    };

    return (
        <div className={`p-4 rounded-xl border ${colors.bg} ${colors.border}`}>
            <div className="flex items-center justify-between mb-2">
                <span className="text-[12px] font-medium text-zinc-500">Diabetic Retinopathy</span>
                <span className={`px-2 py-0.5 text-[11px] font-semibold rounded-full ${colors.bg} ${colors.text}`}>
                    Grade {dr.grade}
                </span>
            </div>
            <div className={`text-[16px] font-bold ${colors.text}`}>
                {dr.grade_name}
            </div>
            <div className="flex items-center justify-between mt-2 text-[12px]">
                <span className="text-zinc-500">Probability: {(dr.probability * 100).toFixed(0)}%</span>
                <span className={`font-medium ${colors.text}`}>{getUrgencyLabel()}</span>
            </div>
        </div>
    );
}

// ============================================================================
// Component: Biomarker Card (PRD Section 6)
// ============================================================================

function BiomarkerCard({ name, biomarker }: { name: string; biomarker: BiomarkerValue }) {
    const config = BIOMARKER_CONFIG[name];
    if (!config) return null;

    const Icon = config.icon;
    
    const getStatusColor = () => {
        switch (biomarker.status) {
            case 'normal': return { bg: 'bg-green-50', border: 'border-green-200', icon: 'text-green-500' };
            case 'borderline': return { bg: 'bg-yellow-50', border: 'border-yellow-200', icon: 'text-yellow-500' };
            case 'abnormal': return { bg: 'bg-red-50', border: 'border-red-200', icon: 'text-red-500' };
            default: return { bg: 'bg-zinc-50', border: 'border-zinc-200', icon: 'text-zinc-500' };
        }
    };

    const colors = getStatusColor();
    const normalRange = biomarker.normal_range;

    return (
        <div className={`p-3 rounded-lg border ${colors.bg} ${colors.border}`}>
            <div className="flex items-center gap-2 mb-2">
                <Icon className={`h-4 w-4 ${colors.icon}`} />
                <span className="text-[12px] font-medium text-zinc-700">{config.displayName}</span>
            </div>
            <div className="text-[18px] font-bold text-zinc-900">
                {config.format(biomarker.value)}
            </div>
            {normalRange && (
                <div className="text-[10px] text-zinc-500 mt-1">
                    Normal: {normalRange[0]} - {normalRange[1]}
                </div>
            )}
            <div className="flex items-center gap-1 mt-2">
                {biomarker.status === 'normal' && <CheckCircle2 className="h-3 w-3 text-green-500" />}
                {biomarker.status === 'borderline' && <AlertTriangle className="h-3 w-3 text-yellow-500" />}
                {biomarker.status === 'abnormal' && <XCircle className="h-3 w-3 text-red-500" />}
                <span className={`text-[10px] font-medium capitalize ${colors.icon}`}>
                    {biomarker.status}
                </span>
            </div>
        </div>
    );
}

// ============================================================================
// Component: Findings List (PRD Section 6)
// ============================================================================

function FindingsList({ findings }: { findings: Finding[] }) {
    const getSeverityIcon = (severity: string) => {
        switch (severity) {
            case 'normal': return <CheckCircle2 className="h-4 w-4 text-green-500" />;
            case 'mild': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
            case 'moderate': return <AlertCircle className="h-4 w-4 text-orange-500" />;
            case 'severe': return <XCircle className="h-4 w-4 text-red-500" />;
            default: return <Info className="h-4 w-4 text-zinc-500" />;
        }
    };

    return (
        <div className="bg-white rounded-xl border border-zinc-200 p-4">
            <h3 className="text-[14px] font-semibold text-zinc-900 mb-3">Clinical Findings</h3>
            <ul className="space-y-2">
                {findings.map((finding, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                        {getSeverityIcon(finding.severity)}
                        <div className="flex-1">
                            <div className="text-[13px] font-medium text-zinc-700">{finding.type}</div>
                            <div className="text-[11px] text-zinc-500">{finding.description}</div>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
}

// ============================================================================
// Component: Recommendations Panel (PRD Section 6)
// ============================================================================

function RecommendationsPanel({ recommendations }: { recommendations: string[] }) {
    return (
        <div className="bg-blue-50 rounded-xl border border-blue-200 p-4">
            <h3 className="text-[14px] font-semibold text-blue-800 mb-3 flex items-center gap-2">
                <Info className="h-4 w-4" />
                Recommendations
            </h3>
            <ul className="space-y-2">
                {recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-[13px] text-blue-700">
                        <span className="text-blue-400">-</span>
                        {rec}
                    </li>
                ))}
            </ul>
        </div>
    );
}

// ============================================================================
// Component: Heatmap Overlay (PRD Section 6)
// ============================================================================

function HeatmapOverlay({ 
    originalUrl, 
    heatmapBase64,
    showHeatmap,
    onToggle
}: { 
    originalUrl: string; 
    heatmapBase64?: string;
    showHeatmap: boolean;
    onToggle: () => void;
}) {
    return (
        <div className="bg-white rounded-xl border border-zinc-200 p-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-[14px] font-semibold text-zinc-900">Analysis View</h3>
                <button
                    onClick={onToggle}
                    className={`px-3 py-1.5 text-[12px] font-medium rounded-lg transition-all ${
                        showHeatmap 
                            ? 'bg-cyan-600 text-white' 
                            : 'bg-zinc-100 text-zinc-700 hover:bg-zinc-200'
                    }`}
                >
                    <span className="flex items-center gap-1.5">
                        <Layers className="h-3.5 w-3.5" />
                        {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
                    </span>
                </button>
            </div>
            
            <div className="relative aspect-square bg-zinc-900 rounded-lg overflow-hidden">
                {/* Original image */}
                <img
                    src={originalUrl}
                    alt="Fundus photograph"
                    className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-300 ${
                        showHeatmap ? 'opacity-0' : 'opacity-100'
                    }`}
                />
                {/* Heatmap overlay */}
                {heatmapBase64 && (
                    <img
                        src={`data:image/png;base64,${heatmapBase64}`}
                        alt="Grad-CAM attention heatmap"
                        className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-300 ${
                            showHeatmap ? 'opacity-100' : 'opacity-0'
                        }`}
                    />
                )}
                
                {/* Zoom hint */}
                <div className="absolute bottom-2 right-2 bg-black/50 text-white text-[10px] px-2 py-1 rounded flex items-center gap-1">
                    <ZoomIn className="h-3 w-3" />
                    Click to zoom
                </div>
            </div>
            
            <div className="mt-2 text-[11px] text-zinc-500 text-center">
                {showHeatmap 
                    ? 'Grad-CAM attention map highlighting regions of interest' 
                    : 'Original fundus photograph'
                }
            </div>
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
    const fileInputRef = useRef<HTMLInputElement>(null);

    const analyzeImage = useCallback(async (imageFile: File) => {
        setState('processing');
        setError(null);

        try {
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('patient_id', `PATIENT_${Date.now()}`);
            formData.append('session_id', crypto.randomUUID());

            const response = await fetch('http://localhost:8000/api/retinal/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Analysis failed: ${response.status}`);
            }

            const data: RetinalAnalysisResponse = await response.json();
            setResults(data);
            setState('complete');
        } catch (err) {
            console.error('Retinal analysis error:', err);
            setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
            setState('error');
        }
    }, []);

    const handleFileSelect = useCallback((file: File) => {
        if (!file.type.startsWith('image/')) {
            setError('Please select an image file (JPG, PNG, TIFF)');
            return;
        }
        if (file.size > 15 * 1024 * 1024) {
            setError('File size must be less than 15MB');
            return;
        }

        setPreviewUrl(URL.createObjectURL(file));
        setState('uploading');
        analyzeImage(file);
    }, [analyzeImage]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    }, [handleFileSelect]);

    const handleReset = useCallback(() => {
        setState('idle');
        setResults(null);
        setError(null);
        setShowHeatmap(false);
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
    }, [previewUrl]);

    const isProcessing = state === 'processing' || state === 'uploading';

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="space-y-6"
        >
            {/* Header with stats (PRD Section 6) */}
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-cyan-50">
                        <Eye className="h-6 w-6 text-cyan-600" strokeWidth={1.5} />
                    </div>
                    <div className="flex-1">
                        <h1 className="text-[20px] font-semibold text-zinc-900">
                            Retinal Analysis
                        </h1>
                        <p className="text-[13px] text-zinc-500 mt-1">
                            AI-powered fundus image analysis for diabetic retinopathy screening, 
                            glaucoma risk assessment, and neurological biomarker extraction.
                        </p>
                    </div>
                </div>

                {/* Stats cards (PRD accuracy specs) */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Target className="h-4 w-4 text-cyan-500" />
                        <div>
                            <div className="text-[13px] font-medium text-zinc-900">93%</div>
                            <div className="text-[11px] text-zinc-500">DR Accuracy</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Clock className="h-4 w-4 text-blue-500" />
                        <div>
                            <div className="text-[13px] font-medium text-zinc-900">&lt;2s</div>
                            <div className="text-[11px] text-zinc-500">Processing</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Shield className="h-4 w-4 text-green-500" />
                        <div>
                            <div className="text-[13px] font-medium text-zinc-900">HIPAA</div>
                            <div className="text-[11px] text-zinc-500">Compliant</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Brain className="h-4 w-4 text-purple-500" />
                        <div>
                            <div className="text-[13px] font-medium text-zinc-900">8</div>
                            <div className="text-[11px] text-zinc-500">Biomarkers</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <AnimatePresence mode="wait">
                {state === 'complete' && results ? (
                    <motion.div
                        key="results"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="space-y-6"
                    >
                        {/* Results Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Left Column: Image & Heatmap */}
                            <div className="space-y-4">
                                {previewUrl && (
                                    <HeatmapOverlay
                                        originalUrl={previewUrl}
                                        heatmapBase64={results.heatmap_base64}
                                        showHeatmap={showHeatmap}
                                        onToggle={() => setShowHeatmap(!showHeatmap)}
                                    />
                                )}
                                
                                {/* DR Grade Badge */}
                                <DRGradeBadge dr={results.diabetic_retinopathy} />
                                
                                {/* Risk Gauge */}
                                <RiskGauge 
                                    score={results.risk_assessment.overall_score}
                                    category={results.risk_assessment.category}
                                    confidence={results.risk_assessment.confidence}
                                />
                            </div>
                            
                            {/* Center Column: Biomarkers & Findings */}
                            <div className="space-y-4">
                                <div className="bg-white rounded-xl border border-zinc-200 p-4">
                                    <h3 className="text-[14px] font-semibold text-zinc-900 mb-3">
                                        Biomarker Analysis (8 Markers)
                                    </h3>
                                    <div className="grid grid-cols-2 gap-2">
                                        {Object.entries(results.biomarkers).map(([name, biomarker]) => (
                                            <BiomarkerCard key={name} name={name} biomarker={biomarker as BiomarkerValue} />
                                        ))}
                                    </div>
                                </div>
                                
                                <FindingsList findings={results.findings} />
                                
                                <RecommendationsPanel recommendations={results.recommendations} />
                            </div>
                            
                            {/* Right Column: AI Explanation */}
                            <div>
                                <ExplanationPanel 
                                    pipeline="retinal"
                                    results={results}
                                    patientContext={{ age: 55, sex: 'unknown' }}
                                />
                            </div>
                        </div>
                        
                        {/* Action Bar */}
                        <div className="flex items-center justify-between p-4 bg-zinc-50 rounded-xl border border-zinc-200">
                            <div className="text-[12px] text-zinc-500">
                                Session: {results.session_id.slice(0, 8)}... | 
                                Processed in {results.processing_time_ms}ms | 
                                Quality: {(results.image_quality.score * 100).toFixed(0)}%
                            </div>
                            <div className="flex items-center gap-3">
                                <button className="flex items-center gap-2 px-4 py-2 text-[13px] font-medium text-zinc-700 bg-white border border-zinc-200 rounded-lg hover:bg-zinc-50">
                                    <Download className="h-4 w-4" />
                                    Export PDF
                                </button>
                                <button 
                                    onClick={handleReset}
                                    className="flex items-center gap-2 px-4 py-2 text-[13px] font-medium text-white bg-cyan-600 rounded-lg hover:bg-cyan-700"
                                >
                                    <RefreshCw className="h-4 w-4" />
                                    New Analysis
                                </button>
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        key="uploader"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="bg-white rounded-xl border border-zinc-200 p-6"
                    >
                        {/* Image Requirements (PRD Section 5) */}
                        <div className="mb-6">
                            <h2 className="text-[14px] font-semibold text-zinc-900 mb-2">
                                Image Requirements
                            </h2>
                            <ul className="space-y-1.5 text-[13px] text-zinc-500">
                                <li className="flex items-start gap-2">
                                    <span className="text-cyan-600">1.</span>
                                    High-quality fundus photograph (minimum 512x512, recommended 1024x1024)
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-cyan-600">2.</span>
                                    Clear visibility of optic disc, vessels, and macular region
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-cyan-600">3.</span>
                                    Supported formats: JPEG, PNG (max 15MB)
                                </li>
                            </ul>
                        </div>

                        {/* Upload Zone */}
                        <div
                            className={`
                                relative border-2 border-dashed rounded-xl p-8 
                                transition-all duration-200 cursor-pointer
                                ${isDragging 
                                    ? 'border-cyan-400 bg-cyan-50' 
                                    : 'border-zinc-300 hover:border-zinc-400 hover:bg-zinc-50'
                                }
                                ${isProcessing ? 'pointer-events-none opacity-60' : ''}
                            `}
                            onDrop={handleDrop}
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/jpeg,image/png,image/tiff"
                                className="hidden"
                                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                                disabled={isProcessing}
                            />

                            <div className="flex flex-col items-center text-center">
                                {previewUrl && isProcessing ? (
                                    <div className="relative mb-4">
                                        <img
                                            src={previewUrl}
                                            alt="Selected fundus image"
                                            className="h-32 w-32 object-cover rounded-lg border border-zinc-200"
                                        />
                                        <div className="absolute inset-0 flex items-center justify-center bg-white/80 rounded-lg">
                                            <Loader2 className="h-8 w-8 text-cyan-600 animate-spin" />
                                        </div>
                                    </div>
                                ) : (
                                    <div className={`
                                        h-16 w-16 rounded-full flex items-center justify-center mb-4
                                        ${isDragging ? 'bg-cyan-100' : 'bg-zinc-100'}
                                    `}>
                                        {isDragging ? (
                                            <Upload className="h-8 w-8 text-cyan-600" />
                                        ) : (
                                            <ImageIcon className="h-8 w-8 text-zinc-400" />
                                        )}
                                    </div>
                                )}

                                <p className="text-[14px] font-medium text-zinc-700 mb-1">
                                    {isProcessing 
                                        ? 'Analyzing fundus image...' 
                                        : isDragging 
                                            ? 'Drop image here' 
                                            : 'Drop fundus photograph here or click to browse'
                                    }
                                </p>
                                <p className="text-[12px] text-zinc-500">
                                    JPEG, PNG up to 15MB
                                </p>

                                {!isProcessing && (
                                    <button
                                        className="mt-4 px-4 py-2 bg-cyan-600 text-white text-[13px] font-medium rounded-lg hover:bg-cyan-700 transition-colors"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            fileInputRef.current?.click();
                                        }}
                                    >
                                        Choose Image File
                                    </button>
                                )}
                            </div>
                        </div>

                        {/* Processing State */}
                        {isProcessing && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mt-6 p-4 bg-cyan-50 border border-cyan-200 rounded-lg"
                            >
                                <div className="flex items-center gap-3">
                                    <Loader2 className="h-5 w-5 text-cyan-600 animate-spin" />
                                    <div>
                                        <div className="text-[13px] font-medium text-cyan-800">
                                            Running AI analysis...
                                        </div>
                                        <div className="text-[12px] text-cyan-600">
                                            Extracting 8 biomarkers, grading DR, generating heatmap
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Error State */}
                        {state === 'error' && error && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg"
                            >
                                <div className="flex items-start gap-3">
                                    <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                                    <div className="flex-1">
                                        <div className="text-[13px] font-medium text-red-800">
                                            Analysis Failed
                                        </div>
                                        <div className="text-[12px] text-red-700 mt-1">{error}</div>
                                        <button
                                            onClick={handleReset}
                                            className="mt-3 text-[12px] font-medium text-red-500 hover:text-red-700"
                                        >
                                            Try Again
                                        </button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Info Panel */}
            <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-6">
                <div className="flex items-start gap-3">
                    <Info className="h-4 w-4 text-zinc-500 flex-shrink-0 mt-0.5" />
                    <div className="text-[12px] text-zinc-500">
                        <p className="font-medium text-zinc-700 mb-1">About Retinal Analysis</p>
                        <p>
                            This module analyzes fundus photographs using EfficientNet-B4 and 
                            specialized biomarker extraction to detect Diabetic Retinopathy (ICDR grades 0-4),
                            assess glaucoma risk via cup-to-disc ratio, and identify vascular abnormalities 
                            associated with hypertension and neurodegeneration. Results include Grad-CAM 
                            attention visualization and clinical recommendations.
                        </p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
