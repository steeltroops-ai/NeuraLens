'use client';

import { useState } from 'react';
import { Upload, Heart, AlertTriangle, CheckCircle2, Loader2, Activity, Zap, TrendingUp } from 'lucide-react';

interface ECGAnalysisResult {
    rhythm: string;
    heartRate: number;
    confidence: number;
    riskLevel: 'normal' | 'low' | 'moderate' | 'high' | 'critical';
    findings: Array<{
        type: string;
        severity: string;
        description: string;
    }>;
    recommendation: string;
}

export function CardiologyAssessment() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState<ECGAnalysisResult | null>(null);
    const [recordingMode, setRecordingMode] = useState<'upload' | 'live'>('upload');

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setResult(null);
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setIsAnalyzing(true);

        // Simulate AI analysis
        await new Promise(resolve => setTimeout(resolve, 2500));

        // Mock result
        setResult({
            rhythm: 'Sinus Rhythm',
            heartRate: 72,
            confidence: 94.7,
            riskLevel: 'normal',
            findings: [
                {
                    type: 'Normal Sinus Rhythm',
                    severity: 'normal',
                    description: 'Regular rhythm with normal P waves, QRS complexes, and T waves',
                },
                {
                    type: 'Heart Rate',
                    severity: 'normal',
                    description: 'Heart rate within normal range (60-100 bpm)',
                },
                {
                    type: 'PR Interval',
                    severity: 'normal',
                    description: 'PR interval 160ms - within normal limits',
                },
            ],
            recommendation: 'No immediate action required. Continue routine monitoring. Maintain healthy lifestyle.',
        });

        setIsAnalyzing(false);
    };

    const getRiskColor = (risk: string) => {
        switch (risk) {
            case 'normal': return 'text-green-600 bg-green-50 border-green-200';
            case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
            case 'moderate': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
            case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
            case 'critical': return 'text-red-600 bg-red-50 border-red-200';
            default: return 'text-zinc-600 bg-zinc-50 border-zinc-200';
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'normal': return 'text-green-600';
            case 'mild': return 'text-yellow-600';
            case 'moderate': return 'text-orange-600';
            case 'severe': return 'text-red-600';
            default: return 'text-zinc-600';
        }
    };

    return (
        <div className="space-y-6">
            {/* Header - Enhanced with cardiology theme */}
            <div className="relative overflow-hidden bg-white rounded-2xl border border-zinc-200/80 p-8">
                {/* Gradient background */}
                <div className="absolute inset-0 bg-gradient-to-br from-red-50/40 via-transparent to-pink-50/30 pointer-events-none" />

                <div className="relative">
                    <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                            <div className="p-3 rounded-xl bg-gradient-to-br from-red-500 to-pink-600 shadow-lg shadow-red-500/20">
                                <Heart className="h-7 w-7 text-white" strokeWidth={2} />
                            </div>
                            <div>
                                <div className="flex items-center gap-3 mb-2">
                                    <h1 className="text-[24px] font-semibold text-zinc-900">CardioPredict AI</h1>
                                    <span className="px-2.5 py-1 bg-red-100 text-red-700 text-[11px] font-medium rounded-full">
                                        ECG Analysis
                                    </span>
                                </div>
                                <p className="text-[14px] text-zinc-600 max-w-xl">
                                    AI-powered ECG analysis for arrhythmia, atrial fibrillation, and cardiac conditions
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Feature pills */}
                    <div className="flex flex-wrap gap-2 mt-6">
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <Activity className="h-4 w-4 text-red-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">ECG Analysis</span>
                        </div>
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <Zap className="h-4 w-4 text-pink-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">Real-time Detection</span>
                        </div>
                        <div className="flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg">
                            <TrendingUp className="h-4 w-4 text-rose-600" strokeWidth={2} />
                            <span className="text-[12px] font-medium text-zinc-700">15+ Conditions</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mode Selector */}
            <div className="flex gap-3">
                <button
                    onClick={() => setRecordingMode('upload')}
                    className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${recordingMode === 'upload'
                        ? 'bg-red-600 text-white shadow-md'
                        : 'bg-white text-zinc-600 border border-zinc-200 hover:bg-zinc-50 hover:border-zinc-300'
                        }`}
                >
                    <Upload size={16} className="inline mr-2" />
                    Upload ECG
                </button>
                <button
                    onClick={() => setRecordingMode('live')}
                    className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${recordingMode === 'live'
                        ? 'bg-red-600 text-white shadow-md'
                        : 'bg-white text-zinc-600 border border-zinc-200 hover:bg-zinc-50 hover:border-zinc-300'
                        }`}
                >
                    <Activity size={16} className="inline mr-2" />
                    Live Monitoring
                </button>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Upload/Recording Section */}
                <div className="space-y-4">
                    <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
                        <h2 className="text-lg font-medium text-zinc-900 mb-4 flex items-center gap-2">
                            <Heart size={20} className="text-red-500" />
                            {recordingMode === 'upload' ? 'Upload ECG Data' : 'Live ECG Monitoring'}
                        </h2>

                        {recordingMode === 'upload' ? (
                            <>
                                {/* File Upload Area */}
                                <div className="relative">
                                    <input
                                        type="file"
                                        accept=".csv,.txt,.dat,.xml"
                                        onChange={handleFileSelect}
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                        id="ecg-upload"
                                    />
                                    <label
                                        htmlFor="ecg-upload"
                                        className="flex flex-col items-center justify-center border-2 border-dashed border-zinc-300 rounded-lg p-8 hover:border-red-500 hover:bg-red-50 transition-all cursor-pointer bg-zinc-50"
                                    >
                                        <Activity size={48} className="text-zinc-400 mb-4" />
                                        <p className="text-sm font-medium text-zinc-900 mb-1">
                                            Drop ECG file here or click to browse
                                        </p>
                                        <p className="text-xs text-zinc-500">
                                            Supports: CSV, TXT, DAT, XML (Max 10MB)
                                        </p>
                                    </label>
                                </div>

                                {/* File Info */}
                                {selectedFile && (
                                    <div className="mt-4 bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                                        <div className="flex items-center justify-between mb-3">
                                            <div>
                                                <p className="text-sm font-medium text-zinc-900">{selectedFile.name}</p>
                                                <p className="text-xs text-zinc-500 mt-1">
                                                    {(selectedFile.size / 1024).toFixed(2)} KB
                                                </p>
                                            </div>
                                            <CheckCircle2 size={20} className="text-green-500" />
                                        </div>
                                        <button
                                            onClick={handleAnalyze}
                                            disabled={isAnalyzing}
                                            className="w-full px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 shadow-sm"
                                        >
                                            {isAnalyzing ? (
                                                <>
                                                    <Loader2 size={16} className="animate-spin" />
                                                    Analyzing ECG...
                                                </>
                                            ) : (
                                                <>
                                                    <Zap size={16} />
                                                    Analyze ECG
                                                </>
                                            )}
                                        </button>
                                    </div>
                                )}
                            </>
                        ) : (
                            /* Live Monitoring Placeholder */
                            <div className="border-2 border-dashed border-zinc-300 rounded-lg p-8 bg-zinc-50">
                                <div className="text-center">
                                    <Activity size={48} className="text-zinc-400 mx-auto mb-4" />
                                    <p className="text-sm text-zinc-500 mb-4">
                                        Connect ECG device to start live monitoring
                                    </p>
                                    <button className="px-4 py-2 bg-zinc-900 text-white rounded-lg text-sm font-medium hover:bg-zinc-800 transition-colors shadow-sm">
                                        Connect Device
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* ECG Waveform Visualization */}
                    {selectedFile && (
                        <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
                            <h3 className="text-sm font-medium text-zinc-900 mb-4">ECG Waveform</h3>
                            <div className="bg-zinc-50 rounded-lg p-4 h-48 flex items-center justify-center border border-zinc-200">
                                {/* Simulated ECG waveform */}
                                <svg className="w-full h-full" viewBox="0 0 400 100">
                                    <path
                                        d="M 0 50 L 40 50 L 45 30 L 50 70 L 55 50 L 60 45 L 70 55 L 80 50 L 120 50 L 125 30 L 130 70 L 135 50 L 140 45 L 150 55 L 160 50 L 200 50 L 205 30 L 210 70 L 215 50 L 220 45 L 230 55 L 240 50 L 280 50 L 285 30 L 290 70 L 295 50 L 300 45 L 310 55 L 320 50 L 360 50 L 365 30 L 370 70 L 375 50 L 380 45 L 390 55 L 400 50"
                                        stroke="#ef4444"
                                        strokeWidth="2"
                                        fill="none"
                                        className="animate-pulse"
                                    />
                                    <line x1="0" y1="50" x2="400" y2="50" stroke="#e4e4e7" strokeWidth="1" />
                                </svg>
                            </div>
                        </div>
                    )}

                    {/* Clinical Info */}
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                        <div className="flex items-start gap-3">
                            <AlertTriangle size={20} className="text-blue-600 mt-0.5 flex-shrink-0" />
                            <div>
                                <h3 className="text-sm font-medium text-blue-900 mb-1">Clinical Guidelines</h3>
                                <p className="text-xs text-blue-700">
                                    Ensure 12-lead ECG with proper electrode placement. AI analysis is for screening purposes.
                                    Critical findings require immediate physician review.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Results Section */}
                <div className="space-y-4">
                    <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
                        <h2 className="text-lg font-medium text-zinc-900 mb-4 flex items-center gap-2">
                            <CheckCircle2 size={20} className="text-green-500" />
                            Analysis Results
                        </h2>

                        {!result && !isAnalyzing && (
                            <div className="text-center py-12">
                                <Heart size={48} className="text-zinc-300 mx-auto mb-4" />
                                <p className="text-sm text-zinc-500">
                                    Upload ECG data to begin cardiac analysis
                                </p>
                            </div>
                        )}

                        {isAnalyzing && (
                            <div className="text-center py-12">
                                <Loader2 size={48} className="text-red-500 mx-auto mb-4 animate-spin" />
                                <p className="text-sm text-zinc-900 mb-2">Analyzing cardiac rhythm...</p>
                                <p className="text-xs text-zinc-500">
                                    Running arrhythmia detection and classification
                                </p>
                            </div>
                        )}

                        {result && (
                            <div className="space-y-4">
                                {/* Primary Diagnosis */}
                                <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                                    <div className="flex items-start justify-between mb-3">
                                        <div>
                                            <h3 className="text-base font-semibold text-zinc-900 mb-1">{result.rhythm}</h3>
                                            <p className="text-xs text-zinc-500">Primary Rhythm Classification</p>
                                        </div>
                                        <span className={`text-xs font-semibold uppercase px-2.5 py-1 rounded border ${getRiskColor(result.riskLevel)}`}>
                                            {result.riskLevel}
                                        </span>
                                    </div>

                                    {/* Confidence Bar */}
                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 h-2 bg-zinc-200 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-red-500 to-pink-500 transition-all duration-500"
                                                style={{ width: `${result.confidence}%` }}
                                            />
                                        </div>
                                        <span className="text-xs font-medium text-zinc-500">
                                            {result.confidence.toFixed(1)}%
                                        </span>
                                    </div>
                                </div>

                                {/* Vital Signs */}
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-3">
                                        <div className="flex items-center gap-2 mb-1">
                                            <Heart size={14} className="text-red-500" />
                                            <p className="text-xs text-zinc-500">Heart Rate</p>
                                        </div>
                                        <p className="text-lg font-semibold text-zinc-900">{result.heartRate} <span className="text-xs text-zinc-500">bpm</span></p>
                                    </div>
                                    <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-3">
                                        <div className="flex items-center gap-2 mb-1">
                                            <TrendingUp size={14} className="text-green-500" />
                                            <p className="text-xs text-zinc-500">Rhythm</p>
                                        </div>
                                        <p className="text-sm font-medium text-zinc-900">Regular</p>
                                    </div>
                                </div>

                                {/* Findings */}
                                <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                                    <h4 className="text-sm font-medium text-zinc-900 mb-3">Clinical Findings</h4>
                                    <div className="space-y-3">
                                        {result.findings.map((finding, index) => (
                                            <div key={index} className="flex items-start gap-3">
                                                <CheckCircle2 size={14} className={`${getSeverityColor(finding.severity)} flex-shrink-0 mt-0.5`} />
                                                <div className="flex-1">
                                                    <p className="text-xs font-medium text-zinc-900">{finding.type}</p>
                                                    <p className="text-xs text-zinc-500 mt-0.5">{finding.description}</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Recommendation */}
                                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                                    <h4 className="text-sm font-medium text-green-700 mb-2">Clinical Recommendation</h4>
                                    <p className="text-xs text-green-600">{result.recommendation}</p>
                                </div>

                                {/* Emergency Alert (if critical) */}
                                {result.riskLevel === 'critical' && (
                                    <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4 animate-pulse">
                                        <div className="flex items-start gap-3">
                                            <AlertTriangle size={20} className="text-red-500 flex-shrink-0" />
                                            <div>
                                                <h4 className="text-sm font-bold text-red-700 mb-1">CRITICAL FINDING</h4>
                                                <p className="text-xs text-red-600">
                                                    Immediate physician review required. Consider emergency intervention.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* ECG Parameters */}
                    {result && (
                        <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
                            <h3 className="text-sm font-medium text-zinc-900 mb-3">ECG Parameters</h3>
                            <div className="space-y-2">
                                {[
                                    { label: 'PR Interval', value: '160 ms', status: 'normal' },
                                    { label: 'QRS Duration', value: '90 ms', status: 'normal' },
                                    { label: 'QT Interval', value: '380 ms', status: 'normal' },
                                    { label: 'QTc (Corrected)', value: '410 ms', status: 'normal' },
                                ].map((param, index) => (
                                    <div key={index} className="flex items-center justify-between bg-zinc-50 rounded-lg p-2 border border-zinc-100">
                                        <span className="text-xs text-zinc-500">{param.label}</span>
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs font-medium text-zinc-900">{param.value}</span>
                                            <span className={`text-xs ${param.status === 'normal' ? 'text-green-500' : 'text-yellow-500'}`}>
                                                {param.status === 'normal' ? '✓' : '⚠'}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
