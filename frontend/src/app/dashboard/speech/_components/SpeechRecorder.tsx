'use client';

/**
 * Speech Recorder Component
 *
 * Records audio in WAV format for reliable backend processing.
 * Uses the MediaRecorder API with Web Audio API for proper format conversion.
 */

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

// Audio recording configuration
const TARGET_SAMPLE_RATE = 16000;
const BITS_PER_SAMPLE = 16;
const NUM_CHANNELS = 1;

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

    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const audioBufferRef = useRef<Float32Array[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const animationRef = useRef<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const isRecordingRef = useRef(false); // Use ref to track recording state for processor
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            cleanup();
        };
    }, []);

    // Auto-stop at max duration
    useEffect(() => {
        if (recordingTime >= maxDuration && isRecording) {
            handleStopRecording();
        }
    }, [recordingTime, maxDuration, isRecording]);

    const cleanup = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        if (processorRef.current) {
            processorRef.current.disconnect();
            processorRef.current = null;
        }
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
    };

    const updateAudioLevel = useCallback(() => {
        if (!analyserRef.current || !isRecordingRef.current) return;

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(average / 255);

        animationRef.current = requestAnimationFrame(updateAudioLevel);
    }, []);

    /**
     * Convert Float32Array audio data to WAV format Blob
     */
    const createWavBlob = (samples: Float32Array, sampleRate: number): Blob => {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        // WAV header
        const writeString = (offset: number, str: string) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // Subchunk1Size
        view.setUint16(20, 1, true); // AudioFormat (PCM)
        view.setUint16(22, NUM_CHANNELS, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * NUM_CHANNELS * (BITS_PER_SAMPLE / 8), true);
        view.setUint16(32, NUM_CHANNELS * (BITS_PER_SAMPLE / 8), true);
        view.setUint16(34, BITS_PER_SAMPLE, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        // Convert Float32 samples to Int16
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const sample = samples[i] ?? 0;
            const s = Math.max(-1, Math.min(1, sample));
            const val = s < 0 ? s * 0x8000 : s * 0x7fff;
            view.setInt16(offset, val, true);
            offset += 2;
        }

        return new Blob([buffer], { type: 'audio/wav' });
    };

    /**
     * Resample audio from source rate to target rate
     */
    const resample = (buffer: Float32Array, fromRate: number, toRate: number): Float32Array => {
        if (fromRate === toRate) return buffer;
        
        const ratio = fromRate / toRate;
        const newLength = Math.round(buffer.length / ratio);
        const result = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, buffer.length - 1);
            const t = srcIndex - srcIndexFloor;
            
            result[i] = (buffer[srcIndexFloor] ?? 0) * (1 - t) + (buffer[srcIndexCeil] ?? 0) * t;
        }
        
        return result;
    };

    const startRecording = async () => {
        try {
            setError(null);
            audioBufferRef.current = [];

            console.log('[SpeechRecorder] Starting recording...');

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: { ideal: 48000 },
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });
            streamRef.current = stream;

            // Create audio context
            audioContextRef.current = new AudioContext();
            const actualSampleRate = audioContextRef.current.sampleRate;
            console.log(`[SpeechRecorder] AudioContext sample rate: ${actualSampleRate}Hz`);

            // Setup analyser for visualization
            analyserRef.current = audioContextRef.current.createAnalyser();
            analyserRef.current.fftSize = 256;

            sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
            sourceRef.current.connect(analyserRef.current);

            // Use ScriptProcessorNode to capture raw audio data
            const bufferSize = 4096;
            processorRef.current = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);

            // Set recording flag BEFORE setting up processor
            isRecordingRef.current = true;
            setIsRecording(true);

            processorRef.current.onaudioprocess = (e) => {
                if (isRecordingRef.current) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    // Clone the data since the buffer is reused
                    audioBufferRef.current.push(new Float32Array(inputData));
                }
            };

            sourceRef.current.connect(processorRef.current);
            // Connect to destination to keep the processor running (but muted)
            processorRef.current.connect(audioContextRef.current.destination);

            setRecordingTime(0);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => prev + 1);
            }, 1000);

            // Start audio level monitoring
            updateAudioLevel();

            console.log('[SpeechRecorder] Recording started successfully');

        } catch (err) {
            console.error('[SpeechRecorder] Recording error:', err);
            if (err instanceof DOMException && err.name === 'NotAllowedError') {
                setError('Microphone access denied. Please allow microphone permissions.');
            } else {
                setError('Failed to start recording. Please check your microphone.');
            }
        }
    };

    const handleStopRecording = () => {
        console.log('[SpeechRecorder] Stopping recording...');
        console.log(`[SpeechRecorder] Buffer chunks: ${audioBufferRef.current.length}`);
        
        // Mark as not recording FIRST
        isRecordingRef.current = false;
        setIsRecording(false);

        // Stop timer
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }

        // Stop animation
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }

        // Process collected audio buffers
        const bufferChunks = audioBufferRef.current;
        const audioContext = audioContextRef.current;
        
        console.log(`[SpeechRecorder] Processing ${bufferChunks.length} chunks`);
        
        if (bufferChunks.length > 0 && audioContext) {
            // Merge all chunks
            const totalLength = bufferChunks.reduce((sum, arr) => sum + arr.length, 0);
            const mergedBuffer = new Float32Array(totalLength);
            
            let offset = 0;
            for (const chunk of bufferChunks) {
                mergedBuffer.set(chunk, offset);
                offset += chunk.length;
            }

            console.log(`[SpeechRecorder] Merged buffer: ${mergedBuffer.length} samples`);

            // Resample to 16kHz for analysis
            const originalRate = audioContext.sampleRate;
            const resampledBuffer = resample(mergedBuffer, originalRate, TARGET_SAMPLE_RATE);

            // Create WAV blob
            const wavBlob = createWavBlob(resampledBuffer, TARGET_SAMPLE_RATE);
            const duration = resampledBuffer.length / TARGET_SAMPLE_RATE;
            console.log(`[SpeechRecorder] Created WAV: ${wavBlob.size} bytes, ${duration.toFixed(2)}s duration`);

            // Clear the buffer
            audioBufferRef.current = [];

            // Cleanup audio resources
            if (processorRef.current) {
                processorRef.current.disconnect();
                processorRef.current = null;
            }
            if (sourceRef.current) {
                sourceRef.current.disconnect();
                sourceRef.current = null;
            }
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
                streamRef.current = null;
            }
            if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
                audioContextRef.current.close();
                audioContextRef.current = null;
            }

            // Call the callback with the WAV blob
            console.log('[SpeechRecorder] Calling onRecordingComplete with blob');
            onRecordingComplete(wavBlob);
        } else {
            console.error('[SpeechRecorder] No audio data captured!');
            setError('No audio was captured. Please try again.');
            cleanup();
        }

        setAudioLevel(0);
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            validateAndUploadFile(file);
        }
    };

    const validateAndUploadFile = (file: File) => {
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/x-m4a', 'audio/m4a', 'audio/ogg', 'audio/webm'];
        const validExtensions = ['.wav', '.mp3', '.m4a', '.ogg', '.webm'];
        
        const hasValidType = validTypes.some(t => file.type.includes(t.split('/')[1] ?? ''));
        const hasValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (hasValidType || hasValidExt || file.type.startsWith('audio/')) {
            onFileUpload(file);
        } else {
            setError('Please upload a valid audio file (WAV, MP3, M4A, WebM, OGG)');
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragActive(false);

        const file = e.dataTransfer.files[0];
        if (file) {
            validateAndUploadFile(file);
        }
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleButtonClick = () => {
        if (isRecording) {
            handleStopRecording();
        } else {
            startRecording();
        }
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
                        onClick={handleButtonClick}
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
                    WAV, MP3, M4A, WebM, OGG - Max 30MB - Max 60 seconds
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
