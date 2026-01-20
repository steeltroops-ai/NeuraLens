"use client";

import React, { useEffect, useRef, useCallback } from "react";

/**
 * Normalizes a raw audio level value to a percentage (0-100).
 */
export function normalizeAudioLevel(
  rawLevel: number,
  maxValue: number = 255,
): number {
  const clampedLevel = Math.max(0, Math.min(rawLevel, maxValue));
  const normalized = (clampedLevel / maxValue) * 100;
  return Math.max(0, Math.min(100, normalized));
}

export interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  isActive: boolean;
  onAudioLevelChange?: (level: number) => void;
  onLowAudioWarning?: (isLow: boolean) => void;
  lowAudioThreshold?: number;
  lowAudioDuration?: number;
  className?: string;
  showWaveform?: boolean;
  showLevelBar?: boolean;
  ariaLabel?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  analyser,
  isActive,
  onAudioLevelChange,
  onLowAudioWarning,
  lowAudioThreshold = 10,
  lowAudioDuration = 2000,
  className = "",
  showWaveform = true,
  showLevelBar = true,
  ariaLabel = "Audio level visualization",
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lowAudioStartRef = useRef<number | null>(null);
  const currentLevelRef = useRef<number>(0);
  const isLowAudioRef = useRef<boolean>(false);
  const levelBarRef = useRef<HTMLDivElement>(null);
  const levelTextRef = useRef<HTMLSpanElement>(null);

  // Store callbacks in refs to avoid dependency issues
  const onAudioLevelChangeRef = useRef(onAudioLevelChange);
  const onLowAudioWarningRef = useRef(onLowAudioWarning);

  useEffect(() => {
    onAudioLevelChangeRef.current = onAudioLevelChange;
    onLowAudioWarningRef.current = onLowAudioWarning;
  }, [onAudioLevelChange, onLowAudioWarning]);

  // Dark theme waveform drawing
  const drawWaveform = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      dataArray: Uint8Array,
      width: number,
      height: number,
    ) => {
      ctx.clearRect(0, 0, width, height);

      // Violet-purple gradient for dark theme
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, "#8b5cf6");
      gradient.addColorStop(0.5, "#a78bfa");
      gradient.addColorStop(1, "#8b5cf6");

      ctx.fillStyle = gradient;

      const barWidth = width / dataArray.length;
      const centerY = height / 2;

      for (let i = 0; i < dataArray.length; i++) {
        const value = dataArray[i] ?? 0;
        const normalizedValue = value / 255;
        const barHeight = normalizedValue * height * 0.8;

        const x = i * barWidth;
        const y = centerY - barHeight / 2;

        ctx.beginPath();
        ctx.roundRect(x, y, Math.max(barWidth - 1, 1), barHeight, 2);
        ctx.fill();
      }
    },
    [],
  );

  // Animation loop using refs to avoid re-creating on every render
  useEffect(() => {
    if (!isActive || !analyser) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      currentLevelRef.current = 0;
      isLowAudioRef.current = false;
      lowAudioStartRef.current = null;

      // Update DOM directly
      if (levelBarRef.current) levelBarRef.current.style.width = "0%";
      if (levelTextRef.current) levelTextRef.current.textContent = "0%";
      return;
    }

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const animate = () => {
      if (!analyser) return;

      analyser.getByteFrequencyData(dataArray);

      // Calculate audio level
      const sum = dataArray.reduce((acc, val) => acc + val, 0);
      const average = sum / dataArray.length;
      const level = normalizeAudioLevel(average, 128);

      currentLevelRef.current = level;

      // Update DOM directly without setState - Dark theme colors
      if (levelBarRef.current) {
        levelBarRef.current.style.width = `${level}%`;
        // Update color based on level - dark theme compatible
        if (level < 10) {
          levelBarRef.current.className =
            "h-full bg-gradient-to-r from-red-500 to-red-400 transition-all duration-100 rounded-full";
        } else if (level < 30) {
          levelBarRef.current.className =
            "h-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-100 rounded-full";
        } else {
          levelBarRef.current.className =
            "h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-100 rounded-full";
        }
      }
      if (levelTextRef.current) {
        levelTextRef.current.textContent = `${Math.round(level)}%`;
      }

      // Callback for level change
      onAudioLevelChangeRef.current?.(level);

      // Check for low audio warning
      const now = Date.now();
      if (level < lowAudioThreshold) {
        if (lowAudioStartRef.current === null) {
          lowAudioStartRef.current = now;
        } else if (now - lowAudioStartRef.current >= lowAudioDuration) {
          if (!isLowAudioRef.current) {
            isLowAudioRef.current = true;
            onLowAudioWarningRef.current?.(true);
          }
        }
      } else {
        if (lowAudioStartRef.current !== null) {
          lowAudioStartRef.current = null;
          if (isLowAudioRef.current) {
            isLowAudioRef.current = false;
            onLowAudioWarningRef.current?.(false);
          }
        }
      }

      // Draw waveform
      if (showWaveform && canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          drawWaveform(ctx, dataArray, canvas.width, canvas.height);
        }
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [
    isActive,
    analyser,
    lowAudioThreshold,
    lowAudioDuration,
    showWaveform,
    drawWaveform,
  ]);

  return (
    <div
      className={`audio-visualizer ${className}`}
      role="img"
      aria-label={ariaLabel}
    >
      {/* Waveform Canvas - Dark Theme */}
      {showWaveform && (
        <canvas
          ref={canvasRef}
          width={256}
          height={64}
          className="w-full h-16 rounded-lg bg-zinc-800"
          aria-hidden="true"
        />
      )}

      {/* Level Bar - Dark Theme */}
      {showLevelBar && (
        <div className="mt-3">
          <div className="flex justify-between text-[11px] text-zinc-400 mb-1.5">
            <span>Audio Level</span>
            <span ref={levelTextRef} className="text-zinc-300 font-medium">
              0%
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
            <div
              ref={levelBarRef}
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-100 rounded-full"
              style={{ width: "0%" }}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Audio input level"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioVisualizer;
