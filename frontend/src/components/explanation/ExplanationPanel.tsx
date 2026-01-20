"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Volume2,
  Pause,
  Play,
  Loader2,
  RefreshCw,
  FileText,
  Cpu,
  AudioLines,
} from "lucide-react";
import { cn } from "@/utils/cn";
import ReactMarkdown from "react-markdown";

interface ExplanationPanelProps {
  pipeline: string;
  results: unknown;
  patientContext?: {
    age?: number;
    sex?: string;
    history?: string[];
  };
  className?: string;
}

export function ExplanationPanel({
  pipeline,
  results,
  patientContext,
  className = "",
}: ExplanationPanelProps) {
  const [explanation, setExplanation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isGeneratingVoice, setIsGeneratingVoice] = useState(false);
  const [audioData, setAudioData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const explanationCompleteRef = useRef(false);

  // Play audio
  const playAudio = useCallback(() => {
    if (audioData && audioRef.current) {
      audioRef.current.play();
      setIsPlaying(true);
      setIsPaused(false);
    } else if (audioData) {
      const audio = new Audio(`data:audio/mp3;base64,${audioData}`);
      audioRef.current = audio;
      audio.onplay = () => {
        setIsPlaying(true);
        setIsPaused(false);
      };
      audio.onended = () => {
        setIsPlaying(false);
        setIsPaused(false);
      };
      audio.onerror = () => {
        setIsPlaying(false);
        setIsPaused(false);
      };
      audio.onpause = () => {
        if (audio.currentTime < audio.duration) {
          setIsPaused(true);
          setIsPlaying(false);
        }
      };
      audio.play();
    }
  }, [audioData]);

  // Pause audio
  const pauseAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
      setIsPaused(true);
    }
  }, []);

  // Stop and reset audio
  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      setIsPaused(false);
    }
  }, []);

  // Generate voice for existing explanation
  const generateVoice = useCallback(
    async (autoPlay: boolean = false) => {
      if (!explanation || explanation.trim().length === 0) {
        return;
      }

      setIsGeneratingVoice(true);
      setError(null);

      try {
        const textToSpeak = explanation.trim().slice(0, 5000);

        const response = await fetch("/api/voice", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: textToSpeak,
            voice_provider: "polly",
          }),
        });

        if (!response.ok) {
          throw new Error(`Voice generation failed: ${response.status}`);
        }

        const data = await response.json();

        if (data.audio_base64) {
          setAudioData(data.audio_base64);

          // Auto-play if requested
          if (autoPlay) {
            const audio = new Audio(
              `data:audio/mp3;base64,${data.audio_base64}`,
            );
            audioRef.current = audio;
            audio.onplay = () => {
              setIsPlaying(true);
              setIsPaused(false);
            };
            audio.onended = () => {
              setIsPlaying(false);
              setIsPaused(false);
            };
            audio.onerror = () => {
              setIsPlaying(false);
              setIsPaused(false);
            };
            audio.onpause = () => {
              if (audio.currentTime < audio.duration) {
                setIsPaused(true);
                setIsPlaying(false);
              }
            };
            audio.play().catch(() => {
              console.log(
                "[Voice] Autoplay blocked, waiting for user interaction",
              );
            });
          }
        } else {
          throw new Error("No audio data received");
        }
      } catch (err) {
        console.error("[Voice] Error:", err);
        setError(
          err instanceof Error ? err.message : "Voice generation failed",
        );
      } finally {
        setIsGeneratingVoice(false);
      }
    },
    [explanation],
  );

  // Generate explanation
  const generateExplanation = async (withAutoVoice: boolean = true) => {
    if (!results) return;

    setIsLoading(true);
    setExplanation("");
    setError(null);
    setAudioData(null);
    stopAudio();
    explanationCompleteRef.current = false;

    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pipeline,
          results,
          patient_context: patientContext,
          voice_output: false,
          voice_provider: "polly",
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate explanation");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let fullExplanation = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.text) {
                fullExplanation += data.text;
                setExplanation(fullExplanation);
              }
              if (data.audio_base64) {
                setAudioData(data.audio_base64);
              }
              if (data.error) {
                setError(data.error);
              }
              if (data.done) {
                setIsLoading(false);
                explanationCompleteRef.current = true;
              }
            } catch {
              // Ignore parsing errors for incomplete JSON
            }
          }
        }
      }

      // Auto-generate voice after explanation is complete
      if (withAutoVoice && fullExplanation.length > 0) {
        setIsLoading(false);
        setTimeout(() => {
          generateVoice(true);
        }, 100);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setIsLoading(false);
    }
  };

  // Track previous results to detect new analysis vs page reload
  const prevResultsRef = useRef<unknown>(null);
  const isInitialMount = useRef(true);

  // Generate explanation when results change (only for NEW results)
  useEffect(() => {
    if (!results) return;

    if (isInitialMount.current) {
      isInitialMount.current = false;
      generateExplanation(false);
      return;
    }

    const resultsChanged =
      JSON.stringify(results) !== JSON.stringify(prevResultsRef.current);

    if (resultsChanged) {
      prevResultsRef.current = results;
      generateExplanation(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, pipeline]);

  // Handle voice button click
  const handleVoiceClick = () => {
    if (isPlaying) {
      pauseAudio();
    } else if (isPaused && audioRef.current) {
      playAudio();
    } else if (audioData) {
      playAudio();
    } else {
      generateVoice(true);
    }
  };

  // Get voice button icon
  const getVoiceButtonContent = () => {
    if (isGeneratingVoice) {
      return <Loader2 className="w-3.5 h-3.5 animate-spin" />;
    }
    if (isPlaying) {
      return <Pause className="w-3.5 h-3.5" />;
    }
    if (isPaused || audioData) {
      return <Play className="w-3.5 h-3.5" />;
    }
    return <Volume2 className="w-3.5 h-3.5" />;
  };

  const getVoiceButtonTitle = () => {
    if (isGeneratingVoice) return "Generating voice...";
    if (isPlaying) return "Pause";
    if (isPaused) return "Resume";
    if (audioData) return "Play";
    return "Generate voice";
  };

  // No results state
  if (!results) {
    return (
      <div
        className={cn(
          "bg-zinc-900 rounded-lg border border-zinc-800",
          className,
        )}
      >
        <div className="p-4 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-violet-500/15">
            <FileText className="h-4 w-4 text-violet-400" />
          </div>
          <div>
            <h3 className="text-[13px] font-medium text-zinc-300">
              AI Clinical Explanation
            </h3>
            <p className="text-[11px] text-zinc-500">
              Will appear after analysis
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn("bg-zinc-900 rounded-lg border border-zinc-800", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-violet-500/15">
            <Cpu className="h-4 w-4 text-violet-400" />
          </div>
          <div>
            <h3 className="text-[14px] font-semibold text-zinc-100">
              AI Clinical Explanation
            </h3>
            <p className="text-[10px] text-zinc-500">
              Llama 3.3 70B | Cerebras Cloud
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-1">
          {/* Regenerate button */}
          <button
            onClick={() => generateExplanation(true)}
            disabled={isLoading || isGeneratingVoice}
            className={cn(
              "p-2 rounded-md transition-colors",
              "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
            title="Regenerate explanation"
          >
            <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
          </button>

          {/* Voice Controls */}
          {explanation && (
            <button
              onClick={handleVoiceClick}
              disabled={isLoading}
              className={cn(
                "p-2 rounded-md transition-all duration-200 flex items-center gap-1",
                isPlaying
                  ? "bg-violet-500 text-white"
                  : audioData || isPaused
                    ? "bg-violet-500/20 text-violet-400"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800",
                isLoading && "opacity-50 cursor-not-allowed",
              )}
              title={getVoiceButtonTitle()}
            >
              {getVoiceButtonContent()}
              {isPlaying && <AudioLines className="w-3 h-3 animate-pulse" />}
            </button>
          )}
        </div>
      </div>

      {/* Content Area */}
      <div className="p-4">
        {/* Error State */}
        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-[12px] text-red-400">
            {error}
          </div>
        )}

        {/* Loading State */}
        {isLoading && explanation === "" && (
          <div className="flex items-center gap-3 py-6 justify-center">
            <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />
            <span className="text-[13px] text-zinc-400">
              Generating clinical explanation...
            </span>
          </div>
        )}

        {/* Markdown Content - Dark Theme Styling */}
        {explanation && (
          <div
            className={cn(
              "prose prose-sm prose-invert max-w-none",
              // Headers
              "[&_h1]:text-[14px] [&_h1]:font-bold [&_h1]:text-zinc-100 [&_h1]:mt-4 [&_h1]:mb-2 [&_h1]:pb-2 [&_h1]:border-b [&_h1]:border-zinc-700",
              "[&_h2]:text-[13px] [&_h2]:font-semibold [&_h2]:text-zinc-200 [&_h2]:mt-4 [&_h2]:mb-2",
              "[&_h3]:text-[12px] [&_h3]:font-semibold [&_h3]:text-zinc-300 [&_h3]:mt-3 [&_h3]:mb-1.5",
              "[&_h4]:text-[11px] [&_h4]:font-medium [&_h4]:text-zinc-400 [&_h4]:mt-2 [&_h4]:mb-1 [&_h4]:uppercase [&_h4]:tracking-wide",
              // Paragraphs
              "[&_p]:text-[12px] [&_p]:leading-relaxed [&_p]:text-zinc-400 [&_p]:mb-2",
              // Text formatting
              "[&_strong]:font-semibold [&_strong]:text-zinc-200",
              "[&_em]:italic [&_em]:text-zinc-500",
              // Lists
              "[&_ul]:list-none [&_ul]:pl-0 [&_ul]:my-2 [&_ul]:space-y-1.5",
              "[&_ul>li]:text-[12px] [&_ul>li]:text-zinc-400 [&_ul>li]:pl-4 [&_ul>li]:relative",
              "[&_ul>li]:before:content-[''] [&_ul>li]:before:absolute [&_ul>li]:before:left-0 [&_ul>li]:before:top-[8px] [&_ul>li]:before:w-1.5 [&_ul>li]:before:h-1.5 [&_ul>li]:before:rounded-full [&_ul>li]:before:bg-violet-500",
              "[&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:my-2 [&_ol]:space-y-1",
              "[&_ol>li]:text-[12px] [&_ol>li]:text-zinc-400 [&_ol>li]:leading-relaxed",
              // Horizontal rules
              "[&_hr]:my-3 [&_hr]:border-zinc-700",
              // Code
              "[&_code]:px-1.5 [&_code]:py-0.5 [&_code]:bg-zinc-800 [&_code]:rounded [&_code]:text-[11px] [&_code]:text-violet-400 [&_code]:font-mono",
              // Blockquotes
              "[&_blockquote]:border-l-2 [&_blockquote]:border-violet-500/50 [&_blockquote]:pl-3 [&_blockquote]:my-2 [&_blockquote]:text-zinc-500 [&_blockquote]:italic [&_blockquote]:text-[12px]",
            )}
          >
            <ReactMarkdown>{explanation}</ReactMarkdown>
            {isLoading && (
              <span className="inline-block w-1.5 h-4 ml-0.5 animate-pulse rounded-sm bg-violet-500" />
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      {explanation && (
        <div className="px-4 py-2 border-t border-zinc-800 flex items-center justify-between text-[10px] text-zinc-600">
          <span>{explanation.split(" ").length} words</span>
          {audioData && (
            <span className="flex items-center gap-1 text-violet-500">
              <Volume2 className="w-3 h-3" />
              Voice ready
            </span>
          )}
        </div>
      )}
    </div>
  );
}
