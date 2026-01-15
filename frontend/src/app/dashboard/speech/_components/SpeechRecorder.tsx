'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Square, Upload, AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';
import { AudioVisualizer } from './AudioVisualizer';

interface SpeechRecorderProps {
    onRecordingComplete: (audioBlob: Blob) => void;
    onFileUpload: (file: File) => void;
    isProcessing: boolean;
    maxDuration?: number;
}

export const SpeechRecorder: React.FC<SpeechRecorderProps> = ({
    onRecordingComplete,
    onFileUpload,
    isProcessing,
    maxDuration = 30,
}) => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [audioLevel, setAudioLevel] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const animationRef = useRef<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            if (audioContextRef.current) audioContextRef.current.close();
        };
    }, []);

    // Auto-stop at max duration
    useEffect(() => {
        if (recordingTime >= maxDuration && isRecording) {
            stopRecording();
        }
    }, [recordingTime, maxDuration, isRecording]);

    const updateAudioLevel = useCallback(() => {
        if (!analyserRef.current) return;

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(average / 255);

        if (isRecording) {
            animationRef.current = requestAnimationFrame(updateAudioLevel);
        }
    }, [isRecording]);

    const startRecording = async () => {
        try {
            setError(null);
            chunksRef.current = [];

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });

            // Setup audio analysis
            audioContextRef.current = new AudioContext({ sampleRate: 16000 });
            analyserRef.current = audioContextRef.current.createAnalyser();
            analyserRef.current.fftSize = 256;

            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyserRef.current);

            // Setup MediaRecorder
            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus',
            });

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                onRecordingComplete(blob);
                stream.getTracks().forEach((track) => track.stop());
            };

            mediaRecorderRef.current.start(100);
            setIsRecording(true);
            setRecordingTime(0);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => prev + 1);
            }, 1000);

            // Start audio level monitoring
            updateAudioLevel();
        } catch (err) {
            setError('Microphone access denied. Please allow microphone permissions.');
            console.error('Recording error:', err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);

            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }

            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }

            setAudioLevel(0);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|m4a|webm|ogg)$/i)) {
                onFileUpload(file);
            } else {
                setError('Please upload a valid audio file (WAV, MP3, M4A, WebM, OGG)');
            }
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragActive(false);

        const file = e.dataTransfer.files[0];
        if (file && (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|m4a|webm|ogg)$/i))) {
            onFileUpload(file);
        } else {
            setError('Please upload a valid audio file');
        }
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="space-y-4">
            {/* Recording Interface */}
            <div className="flex flex-col items-center py-6">
                {/* Main Record Button */}
                <div className="relative mb-4">
                    {/* Audio level ring */}
                    {isRecording && (
                        <motion.div
                            className="absolute inset-0 rounded-full bg-[#ef4444]/20"
                            animate={{
                                scale: 1 + audioLevel * 0.5,
                                opacity: 0.3 + audioLevel * 0.4,
                            }}
                            transition={{ duration: 0.1 }}
                        />
                    )}

                    <motion.button
                        whileHover={{ scale: isProcessing ? 1 : 1.05 }}
                        whileTap={{ scale: isProcessing ? 1 : 0.95 }}
                        onClick={isRecording ? stopRecording : startRecording}
                        disabled={isProcessing}
                        className={cn(
                            'relative z-10 flex h-20 w-20 items-center justify-center rounded-full transition-all duration-200',
                            isRecording
                                ? 'bg-[#ef4444] hover:bg-[#dc2626]'
                                : 'bg-[#3b82f6] hover:bg-[#2563eb]',
                            isProcessing && 'opacity-50 cursor-not-allowed'
                        )}
                        aria-label={isRecording ? 'Stop recording' : 'Start recording'}
                    >
                        {isProcessing ? (
                            <Loader2 className="h-8 w-8 text-white animate-spin" />
                        ) : isRecording ? (
                            <Square className="h-7 w-7 text-white" />
                        ) : (
                            <Mic className="h-8 w-8 text-white" />
                        )}
                    </motion.button>
                </div>

                {/* Recording status */}
                <div className="text-center">
                    {isRecording ? (
                        <div className="space-y-1">
                            <div className="flex items-center justify-center gap-2">
                                <span className="h-2 w-2 rounded-full bg-[#ef4444] animate-pulse" />
                                <span className="text-[13px] font-medium text-[#ef4444]">Recording</span>
                            </div>
                            <div className="text-[20px] font-semibold text-[#0f172a] tabular-nums">
                                {formatTime(recordingTime)} / {formatTime(maxDuration)}
                            </div>
                        </div>
                    ) : isProcessing ? (
                        <span className="text-[13px] text-[#64748b]">Processing audio...</span>
                    ) : (
                        <span className="text-[13px] text-[#64748b]">
                            Click to record or upload an audio file
                        </span>
                    )}
                </div>

                {/* Audio Visualizer */}
                {isRecording && (
                    <div className="w-full max-w-md mt-4">
                        <AudioVisualizer
                            analyser={analyserRef.current}
                            isActive={isRecording}
                            showWaveform={true}
                            showLevelBar={true}
                            onAudioLevelChange={(level) => setAudioLevel(level / 100)}
                            className="rounded-lg"
                        />
                    </div>
                )}

                {/* Progress bar during recording */}
                {isRecording && (
                    <div className="w-full max-w-xs mt-4">
                        <div className="h-1.5 bg-[#e2e8f0] rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-[#ef4444]"
                                initial={{ width: 0 }}
                                animate={{ width: `${(recordingTime / maxDuration) * 100}%` }}
                                transition={{ duration: 0.5 }}
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Divider */}
            <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-[#e2e8f0]" />
                <span className="text-[12px] text-[#94a3b8]">or</span>
                <div className="flex-1 h-px bg-[#e2e8f0]" />
            </div>

            {/* File Upload */}
            <div
                className={cn(
                    'border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer',
                    dragActive ? 'border-[#3b82f6] bg-[#eff6ff]' : 'border-[#e2e8f0] hover:border-[#cbd5e1]',
                    isProcessing && 'opacity-50 pointer-events-none'
                )}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg"
                    onChange={handleFileChange}
                    className="hidden"
                />
                <Upload className="h-6 w-6 text-[#94a3b8] mx-auto mb-2" />
                <p className="text-[13px] text-[#64748b]">
                    Drop audio file here or <span className="text-[#3b82f6]">browse</span>
                </p>
                <p className="text-[11px] text-[#94a3b8] mt-1">
                    WAV, MP3, M4A, WebM, OGG • Max 30MB • Max 60 seconds
                </p>
            </div>

            {/* Error display */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="flex items-center gap-2 p-3 bg-[#fef2f2] border border-[#fecaca] rounded-lg"
                    >
                        <AlertCircle className="h-4 w-4 text-[#ef4444] flex-shrink-0" />
                        <span className="text-[13px] text-[#991b1b]">{error}</span>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default SpeechRecorder;
