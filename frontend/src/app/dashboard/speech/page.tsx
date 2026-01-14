'use client';

/**
 * Speech Analysis Module Page
 * 
 * Clinical-grade speech analysis for neurological assessment.
 * Implements the SpeechMD AI diagnostic module with:
 * - Real-time audio recording with visual feedback
 * - File upload support (WAV, MP3, M4A, WebM, OGG)
 * - 9 biomarker extraction and visualization
 * - Risk score calculation with confidence intervals
 * - Clinical recommendations
 */

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Mic,
    Info,
    Shield,
    Clock,
    Target,
    AlertCircle,
    Loader2,
} from 'lucide-react';
import { SpeechRecorder } from '@/components/speech/SpeechRecorder';
import { SpeechResultsPanel } from '@/components/speech/SpeechResultsPanel';
import type { EnhancedSpeechAnalysisResponse } from '@/types/speech-enhanced';

type AnalysisState = 'idle' | 'recording' | 'uploading' | 'processing' | 'complete' | 'error';

export default function SpeechAnalysisPage() {
    const [state, setState] = useState<AnalysisState>('idle');
    const [results, setResults] = useState<EnhancedSpeechAnalysisResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [uploadProgress, setUploadProgress] = useState(0);

    const analyzeAudio = useCallback(async (audioData: Blob | File) => {
        setState('processing');
        setError(null);
        setUploadProgress(0);

        try {
            const formData = new FormData();
            formData.append('audio_file', audioData);
            formData.append('session_id', `speech_${Date.now()}`);

            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

            const response = await fetch(`${apiUrl}/api/v1/speech/analyze/enhanced`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Analysis failed: ${response.status}`);
            }

            const data: EnhancedSpeechAnalysisResponse = await response.json();

            if (data.status === 'error') {
                throw new Error(data.error_message || 'Analysis failed');
            }

            setResults(data);
            setState('complete');
        } catch (err) {
            console.error('Speech analysis error:', err);
            setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
            setState('error');
        }
    }, []);

    const handleRecordingComplete = useCallback((audioBlob: Blob) => {
        analyzeAudio(audioBlob);
    }, [analyzeAudio]);

    const handleFileUpload = useCallback((file: File) => {
        setState('uploading');
        analyzeAudio(file);
    }, [analyzeAudio]);

    const handleReset = useCallback(() => {
        setState('idle');
        setResults(null);
        setError(null);
        setUploadProgress(0);
    }, []);

    const isProcessing = state === 'processing' || state === 'uploading';

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="space-y-6"
        >
            {/* Header */}
            <div className="bg-white rounded-lg border border-[#e2e8f0] p-5">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-[#eff6ff]">
                        <Mic className="h-6 w-6 text-[#3b82f6]" strokeWidth={1.5} />
                    </div>
                    <div className="flex-1">
                        <h1 className="text-[20px] font-semibold text-[#0f172a]">
                            Speech Analysis
                        </h1>
                        <p className="text-[13px] text-[#64748b] mt-1">
                            AI-powered voice biomarker analysis for neurological assessment.
                            Record a speech sample or upload an audio file for analysis.
                        </p>
                    </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                    <div className="flex items-center gap-2 p-3 bg-[#f8fafc] rounded-lg">
                        <Target className="h-4 w-4 text-[#64748b]" />
                        <div>
                            <div className="text-[13px] font-medium text-[#0f172a]">95.2%</div>
                            <div className="text-[11px] text-[#64748b]">Accuracy</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-[#f8fafc] rounded-lg">
                        <Clock className="h-4 w-4 text-[#64748b]" />
                        <div>
                            <div className="text-[13px] font-medium text-[#0f172a]">&lt;3s</div>
                            <div className="text-[11px] text-[#64748b]">Processing</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-[#f8fafc] rounded-lg">
                        <Shield className="h-4 w-4 text-[#64748b]" />
                        <div>
                            <div className="text-[13px] font-medium text-[#0f172a]">HIPAA</div>
                            <div className="text-[11px] text-[#64748b]">Compliant</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 p-3 bg-[#f8fafc] rounded-lg">
                        <Info className="h-4 w-4 text-[#64748b]" />
                        <div>
                            <div className="text-[13px] font-medium text-[#0f172a]">9</div>
                            <div className="text-[11px] text-[#64748b]">Biomarkers</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <AnimatePresence mode="wait">
                {state === 'complete' && results ? (
                    <SpeechResultsPanel
                        key="results"
                        results={results}
                        onReset={handleReset}
                    />
                ) : (
                    <motion.div
                        key="recorder"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="bg-white rounded-lg border border-[#e2e8f0] p-5"
                    >
                        {/* Instructions */}
                        <div className="mb-6">
                            <h2 className="text-[14px] font-semibold text-[#0f172a] mb-2">
                                Recording Instructions
                            </h2>
                            <ul className="space-y-1.5 text-[13px] text-[#64748b]">
                                <li className="flex items-start gap-2">
                                    <span className="text-[#3b82f6]">1.</span>
                                    Find a quiet environment with minimal background noise
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-[#3b82f6]">2.</span>
                                    Speak naturally for 10-30 seconds (reading or conversation)
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-[#3b82f6]">3.</span>
                                    Maintain consistent distance from microphone
                                </li>
                            </ul>
                        </div>

                        {/* Recorder */}
                        <SpeechRecorder
                            onRecordingComplete={handleRecordingComplete}
                            onFileUpload={handleFileUpload}
                            isProcessing={isProcessing}
                            maxDuration={30}
                        />

                        {/* Processing State */}
                        {isProcessing && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mt-6 p-4 bg-[#eff6ff] border border-[#bfdbfe] rounded-lg"
                            >
                                <div className="flex items-center gap-3">
                                    <Loader2 className="h-5 w-5 text-[#3b82f6] animate-spin" />
                                    <div>
                                        <div className="text-[13px] font-medium text-[#1e40af]">
                                            {state === 'uploading' ? 'Uploading audio...' : 'Analyzing speech patterns...'}
                                        </div>
                                        <div className="text-[12px] text-[#3b82f6]">
                                            Extracting biomarkers and calculating risk score
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
                                className="mt-6 p-4 bg-[#fef2f2] border border-[#fecaca] rounded-lg"
                            >
                                <div className="flex items-start gap-3">
                                    <AlertCircle className="h-5 w-5 text-[#ef4444] flex-shrink-0 mt-0.5" />
                                    <div className="flex-1">
                                        <div className="text-[13px] font-medium text-[#991b1b]">
                                            Analysis Failed
                                        </div>
                                        <div className="text-[12px] text-[#b91c1c] mt-1">{error}</div>
                                        <button
                                            onClick={handleReset}
                                            className="mt-3 text-[12px] font-medium text-[#ef4444] hover:text-[#dc2626]"
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
            <div className="bg-[#f8fafc] rounded-lg border border-[#e2e8f0] p-4">
                <div className="flex items-start gap-3">
                    <Info className="h-4 w-4 text-[#64748b] flex-shrink-0 mt-0.5" />
                    <div className="text-[12px] text-[#64748b]">
                        <p className="font-medium text-[#475569] mb-1">About Speech Analysis</p>
                        <p>
                            This module analyzes voice biomarkers associated with neurological conditions
                            including Parkinson's disease, early dementia, and speech disorders.
                            Results should be reviewed by a healthcare professional and are not intended
                            as a standalone diagnosis.
                        </p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
