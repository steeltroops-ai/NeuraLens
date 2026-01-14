'use client';

import React, { useEffect, useRef, useCallback, useState } from 'react';

/**
 * Normalizes a raw audio level value to a percentage (0-100).
 * 
 * @param rawLevel - Raw audio level value (typically 0-255 from frequency data)
 * @param maxValue - Maximum expected raw value (default 255)
 * @returns Normalized value between 0 and 100
 */
export function normalizeAudioLevel(rawLevel: number, maxValue: number = 255): number {
    // Clamp the input to valid range
    const clampedLevel = Math.max(0, Math.min(rawLevel, maxValue));
    // Normalize to 0-100 range
    const normalized = (clampedLevel / maxValue) * 100;
    // Ensure output is within bounds (handles floating point edge cases)
    return Math.max(0, Math.min(100, normalized));
}

export interface AudioVisualizerProps {
    /** The AnalyserNode from Web Audio API to read audio data from */
    analyser: AnalyserNode | null;
    /** Whether the visualizer is currently active (recording) */
    isActive: boolean;
    /** Callback when audio level is updated */
    onAudioLevelChange?: (level: number) => void;
    /** Callback when low audio is detected for extended period */
    onLowAudioWarning?: (isLow: boolean) => void;
    /** Threshold below which audio is considered "low" (0-100, default 10) */
    lowAudioThreshold?: number;
    /** Duration in ms before triggering low audio warning (default 2000) */
    lowAudioDuration?: number;
    /** Custom className for the container */
    className?: string;
    /** Whether to show the waveform visualization */
    showWaveform?: boolean;
    /** Whether to show the level bar */
    showLevelBar?: boolean;
    /** Accessible label for screen readers */
    ariaLabel?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
    analyser,
    isActive,
    onAudioLevelChange,
    onLowAudioWarning,
    lowAudioThreshold = 10,
    lowAudioDuration = 2000,
    className = '',
    showWaveform = true,
    showLevelBar = true,
    ariaLabel = 'Audio level visualization',
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationFrameRef = useRef<number | null>(null);
    const lowAudioStartRef = useRef<number | null>(null);
    const [currentLevel, setCurrentLevel] = useState<number>(0);
    const [isLowAudio, setIsLowAudio] = useState<boolean>(false);

    /**
     * Calculate the current audio level from the analyser
     */
    const calculateAudioLevel = useCallback((dataArray: Uint8Array): number => {
        if (dataArray.length === 0) return 0;

        // Calculate average of frequency data
        const sum = dataArray.reduce((acc, val) => acc + val, 0);
        const average = sum / dataArray.length;

        // Normalize to 0-100 range
        return normalizeAudioLevel(average, 128);
    }, []);

    /**
     * Draw waveform visualization on canvas
     */
    const drawWaveform = useCallback((
        ctx: CanvasRenderingContext2D,
        dataArray: Uint8Array,
        width: number,
        height: number
    ) => {
        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Set up gradient for waveform
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#10b981'); // green-500
        gradient.addColorStop(0.5, '#22c55e'); // green-400
        gradient.addColorStop(1, '#10b981'); // green-500

        ctx.fillStyle = gradient;

        const barWidth = width / dataArray.length;
        const centerY = height / 2;

        for (let i = 0; i < dataArray.length; i++) {
            const value = dataArray[i] ?? 0;
            const normalizedValue = value / 255;
            const barHeight = normalizedValue * height * 0.8;

            const x = i * barWidth;
            const y = centerY - barHeight / 2;

            // Draw rounded bars
            ctx.beginPath();
            ctx.roundRect(x, y, Math.max(barWidth - 1, 1), barHeight, 2);
            ctx.fill();
        }
    }, []);

    /**
     * Main animation loop using requestAnimationFrame
     */
    const animate = useCallback(() => {
        if (!analyser || !isActive) {
            return;
        }

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        // Calculate and update audio level
        const level = calculateAudioLevel(dataArray);
        setCurrentLevel(level);
        onAudioLevelChange?.(level);

        // Check for low audio warning
        const now = Date.now();
        if (level < lowAudioThreshold) {
            if (lowAudioStartRef.current === null) {
                lowAudioStartRef.current = now;
            } else if (now - lowAudioStartRef.current >= lowAudioDuration) {
                if (!isLowAudio) {
                    setIsLowAudio(true);
                    onLowAudioWarning?.(true);
                }
            }
        } else {
            if (lowAudioStartRef.current !== null) {
                lowAudioStartRef.current = null;
                if (isLowAudio) {
                    setIsLowAudio(false);
                    onLowAudioWarning?.(false);
                }
            }
        }

        // Draw waveform if canvas is available
        if (showWaveform && canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                drawWaveform(ctx, dataArray, canvas.width, canvas.height);
            }
        }

        // Schedule next frame
        animationFrameRef.current = requestAnimationFrame(animate);
    }, [
        analyser,
        isActive,
        calculateAudioLevel,
        onAudioLevelChange,
        lowAudioThreshold,
        lowAudioDuration,
        isLowAudio,
        onLowAudioWarning,
        showWaveform,
        drawWaveform,
    ]);

    // Start/stop animation based on isActive
    useEffect(() => {
        if (isActive && analyser) {
            animate();
        } else {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            // Reset state when not active
            setCurrentLevel(0);
            setIsLowAudio(false);
            lowAudioStartRef.current = null;
        }

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, [isActive, analyser, animate]);

    // Get level bar color based on current level
    const getLevelBarColor = (): string => {
        if (currentLevel < 10) return 'from-red-400 to-red-500';
        if (currentLevel < 30) return 'from-yellow-400 to-yellow-500';
        return 'from-green-400 to-green-600';
    };

    return (
        <div
            className={`audio-visualizer ${className}`}
            role="img"
            aria-label={ariaLabel}
            aria-describedby="audio-level-description"
        >
            {/* Screen reader description */}
            <span id="audio-level-description" className="sr-only">
                Current audio level: {Math.round(currentLevel)} percent
                {isLowAudio && '. Warning: Audio level is too low.'}
            </span>

            {/* Waveform Canvas */}
            {showWaveform && (
                <canvas
                    ref={canvasRef}
                    width={256}
                    height={64}
                    className="w-full h-16 rounded-lg bg-gray-100"
                    aria-hidden="true"
                />
            )}

            {/* Level Bar */}
            {showLevelBar && (
                <div className="mt-2">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Audio Level</span>
                        <span>{Math.round(currentLevel)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-gray-200">
                        <div
                            className={`h-full bg-gradient-to-r ${getLevelBarColor()} transition-all duration-100`}
                            style={{ width: `${currentLevel}%` }}
                            role="progressbar"
                            aria-valuenow={Math.round(currentLevel)}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-label="Audio input level"
                        />
                    </div>
                </div>
            )}

            {/* Low Audio Warning */}
            {isLowAudio && (
                <div
                    className="mt-3 p-3 rounded-lg bg-amber-50 border border-amber-200"
                    role="alert"
                    aria-live="polite"
                >
                    <div className="flex items-center gap-2">
                        <svg
                            className="h-5 w-5 text-amber-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                            />
                        </svg>
                        <span className="text-sm font-medium text-amber-700">
                            Audio level is too low. Please speak louder or move closer to the microphone.
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AudioVisualizer;
