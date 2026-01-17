'use client';

import { useState } from 'react';
import { Upload, Scan, AlertCircle, CheckCircle2, Loader2, Download, Share2, Activity, Zap, TrendingUp } from 'lucide-react';

interface DiagnosisResult {
    condition: string;
    confidence: number;
    severity: 'normal' | 'mild' | 'moderate' | 'severe' | 'critical';
    description: string;
}

export function RadiologyAssessment() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [results, setResults] = useState<DiagnosisResult[] | null>(null);

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setResults(null);

            // Create preview URL
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setIsAnalyzing(true);

        // Simulate AI analysis (replace with actual API call)
        await new Promise(resolve => setTimeout(resolve, 2500));

        // Mock results
        setResults([
            {
                condition: 'Pneumonia',
                confidence: 87.5,
                severity: 'moderate',
                description: 'Bilateral infiltrates detected in lower lobes. Bacterial pneumonia suspected.',
            },
            {
                condition: 'Pleural Effusion',
                confidence: 72.3,
                severity: 'mild',
                description: 'Small amount of fluid detected in right pleural space.',
            },
            {
                condition: 'Normal Heart Size',
                confidence: 94.1,
                severity: 'normal',
                description: 'Cardiothoracic ratio within normal limits.',
            },
        ]);

        setIsAnalyzing(false);
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'normal': return 'text-green-600 bg-green-50 border-green-200';
            case 'mild': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
            case 'moderate': return 'text-orange-600 bg-orange-50 border-orange-200';
            case 'severe': return 'text-red-500 bg-red-50 border-red-200';
            case 'critical': return 'text-red-700 bg-red-100 border-red-300';
            default: return 'text-zinc-500 bg-zinc-50 border-zinc-200';
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="rounded-xl border border-zinc-200 bg-white p-6">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-blue-50">
                        <Upload className="h-6 w-6 text-blue-600" strokeWidth={1.5} />
                    </div>
                    <div className="flex-1">
                        <h1 className="text-[20px] font-semibold text-zinc-900">ChestXplorer AI</h1>
                        <p className="text-[13px] text-zinc-600 mt-1">
                            AI-powered chest X-ray analysis for pneumonia, COVID-19, TB, lung cancer, and cardiac conditions
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Activity className="h-4 w-4 text-zinc-500" />
                        <span className="text-[13px] text-zinc-700">X-Ray Analysis</span>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <Zap className="h-4 w-4 text-zinc-500" />
                        <span className="text-[13px] text-zinc-700">Multi-class Detection</span>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-zinc-50 rounded-lg">
                        <TrendingUp className="h-4 w-4 text-zinc-500" />
                        <span className="text-[13px] text-zinc-700">8+ Conditions</span>
                    </div>
                </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Upload Section */}
                <div className="space-y-4">
                    <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
                        <h2 className="text-lg font-medium text-zinc-900 mb-4 flex items-center gap-2">
                            <Upload size={20} className="text-blue-600" />
                            Upload X-Ray Image
                        </h2>

                        {/* File Upload Area */}
                        <div className="relative">
                            <input
                                type="file"
                                accept="image/*,.dcm"
                                onChange={handleFileSelect}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                id="xray-upload"
                            />
                            <label
                                htmlFor="xray-upload"
                                className="flex flex-col items-center justify-center border-2 border-dashed border-zinc-300 rounded-lg p-8 hover:border-blue-500 hover:bg-blue-50 transition-all cursor-pointer bg-zinc-50"
                            >
                                <Scan size={48} className="text-zinc-400 mb-4" />
                                <p className="text-sm font-medium text-zinc-900 mb-1">
                                    Drop X-ray here or click to browse
                                </p>
                                <p className="text-xs text-zinc-500">
                                    Supports: JPEG, PNG, DICOM (Max 50MB)
                                </p>
                            </label>
                        </div>

                        {/* Preview */}
                        {previewUrl && (
                            <div className="mt-4">
                                <img
                                    src={previewUrl}
                                    alt="X-ray preview"
                                    className="w-full rounded-lg border border-zinc-200"
                                />
                                <div className="mt-3 flex items-center justify-between">
                                    <span className="text-xs text-zinc-500">{selectedFile?.name}</span>
                                    <button
                                        onClick={handleAnalyze}
                                        disabled={isAnalyzing}
                                        className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 shadow-sm"
                                    >
                                        {isAnalyzing ? (
                                            <>
                                                <Loader2 size={16} className="animate-spin" />
                                                Analyzing...
                                            </>
                                        ) : (
                                            <>
                                                <Scan size={16} />
                                                Analyze X-Ray
                                            </>
                                        )}
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Info Card */}
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                        <div className="flex items-start gap-3">
                            <AlertCircle size={20} className="text-blue-600 mt-0.5 flex-shrink-0" />
                            <div>
                                <h3 className="text-sm font-medium text-blue-900 mb-1">Clinical Guidelines</h3>
                                <p className="text-xs text-blue-700">
                                    Ensure X-ray is PA or AP view, properly exposed, and includes full lung fields.
                                    AI assists diagnosis but does not replace clinical judgment.
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

                        {!results && !isAnalyzing && (
                            <div className="text-center py-12">
                                <Scan size={48} className="text-zinc-300 mx-auto mb-4" />
                                <p className="text-sm text-zinc-500">
                                    Upload an X-ray image to begin analysis
                                </p>
                            </div>
                        )}

                        {isAnalyzing && (
                            <div className="text-center py-12">
                                <Loader2 size={48} className="text-blue-500 mx-auto mb-4 animate-spin" />
                                <p className="text-sm text-zinc-900 mb-2">Analyzing X-ray image...</p>
                                <p className="text-xs text-zinc-500">
                                    Running multi-class detection and classification
                                </p>
                            </div>
                        )}

                        {results && (
                            <div className="space-y-3">
                                {results.map((result, index) => (
                                    <div
                                        key={index}
                                        className="bg-zinc-50 border border-zinc-200 rounded-lg p-4 hover:bg-zinc-100 transition-colors"
                                    >
                                        <div className="flex items-start justify-between mb-2">
                                            <h3 className="text-sm font-medium text-zinc-900">{result.condition}</h3>
                                            <span className={`text-xs font-semibold uppercase px-2 py-1 rounded border ${getSeverityColor(result.severity)}`}>
                                                {result.severity}
                                            </span>
                                        </div>
                                        <p className="text-xs text-zinc-600 mb-3">{result.description}</p>
                                        <div className="flex items-center gap-2">
                                            <div className="flex-1 h-2 bg-zinc-200 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-500"
                                                    style={{ width: `${result.confidence}%` }}
                                                />
                                            </div>
                                            <span className="text-xs font-medium text-zinc-500">
                                                {result.confidence.toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                ))}

                                {/* Action Buttons */}
                                <div className="flex gap-3 mt-6 pt-4 border-t border-zinc-100">
                                    <button className="flex-1 px-4 py-2 bg-zinc-100 text-zinc-700 rounded-lg text-sm font-medium hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2 border border-zinc-200">
                                        <Download size={16} />
                                        Export Report
                                    </button>
                                    <button className="flex-1 px-4 py-2 bg-zinc-100 text-zinc-700 rounded-lg text-sm font-medium hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2 border border-zinc-200">
                                        <Share2 size={16} />
                                        Share Results
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Detected Conditions Summary */}
                    {results && (
                        <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
                            <h3 className="text-sm font-medium text-zinc-900 mb-3">Detected Conditions</h3>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-zinc-50 rounded-lg p-3 border border-zinc-100">
                                    <p className="text-xs text-zinc-500 mb-1">Primary Finding</p>
                                    <p className="text-sm font-medium text-zinc-900">Pneumonia</p>
                                </div>
                                <div className="bg-zinc-50 rounded-lg p-3 border border-zinc-100">
                                    <p className="text-xs text-zinc-500 mb-1">Confidence</p>
                                    <p className="text-sm font-medium text-green-600">87.5%</p>
                                </div>
                                <div className="bg-zinc-50 rounded-lg p-3 border border-zinc-100">
                                    <p className="text-xs text-zinc-500 mb-1">Severity</p>
                                    <p className="text-sm font-medium text-orange-600">Moderate</p>
                                </div>
                                <div className="bg-zinc-50 rounded-lg p-3 border border-zinc-100">
                                    <p className="text-xs text-zinc-500 mb-1">Urgency</p>
                                    <p className="text-sm font-medium text-yellow-600">Follow-up</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
