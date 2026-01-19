"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Volume2,
  Pause,
  Play,
  Loader2,
  Sparkles,
  RefreshCw,
  Bot,
} from "lucide-react";
import { cn } from "@/utils/cn";

interface ExplanationPanelProps {
  pipeline: string;
  results: unknown;
  patientContext?: {
    age?: number;
    sex?: string;
    history?: string[];
  };
  className?: string;
  theme?: "light" | "dark";
}

export function ExplanationPanel({
  pipeline,
  results,
  patientContext,
  className = "",
  theme = "light",
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

  const isDark = theme === "dark";

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
              // Autoplay blocked by browser, user needs to click
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
        // Small delay to ensure state is updated
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

    // Skip auto-voice on initial mount (page reload with existing results)
    if (isInitialMount.current) {
      isInitialMount.current = false;
      // Still generate explanation but WITHOUT auto-voice
      generateExplanation(false);
      return;
    }

    // Check if results actually changed (new analysis)
    const resultsChanged =
      JSON.stringify(results) !== JSON.stringify(prevResultsRef.current);

    if (resultsChanged) {
      prevResultsRef.current = results;
      // New analysis - generate explanation WITH auto-voice
      generateExplanation(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, pipeline]);

  // Handle voice button click
  const handleVoiceClick = () => {
    if (isPlaying) {
      // Currently playing -> Pause
      pauseAudio();
    } else if (isPaused && audioRef.current) {
      // Currently paused -> Resume
      playAudio();
    } else if (audioData) {
      // Has audio but not playing -> Play from start
      playAudio();
    } else {
      // No audio -> Generate and play
      generateVoice(true);
    }
  };

  // Light theme styles
  const containerStyles = isDark
    ? "bg-gradient-to-br from-zinc-900 to-zinc-950 border-zinc-800"
    : "bg-white border-zinc-200 shadow-sm";

  const headerTextStyles = isDark ? "text-white" : "text-zinc-900";

  const badgeStyles = isDark
    ? "text-zinc-400 bg-zinc-800"
    : "text-zinc-500 bg-zinc-100";

  const buttonStyles = isDark
    ? "text-zinc-400 hover:text-white hover:bg-zinc-800"
    : "text-zinc-500 hover:text-zinc-700 hover:bg-zinc-100";

  const errorStyles = isDark
    ? "bg-red-500/10 border-red-500/30 text-red-400"
    : "bg-red-50 border-red-200 text-red-600";

  const loadingTextStyles = isDark ? "text-zinc-400" : "text-zinc-500";

  const contentStyles = isDark ? "text-zinc-300" : "text-zinc-700";

  const footerStyles = isDark
    ? "border-zinc-800 text-zinc-500"
    : "border-zinc-200 text-zinc-400";

  const accentColor = isDark ? "text-purple-400" : "text-blue-600";
  const secondaryAccent = isDark ? "text-blue-400" : "text-indigo-600";

  // Get voice button icon and style
  const getVoiceButtonContent = () => {
    if (isGeneratingVoice) {
      return <Loader2 className="w-4 h-4 animate-spin" />;
    }
    if (isPlaying) {
      return <Pause className="w-4 h-4" />;
    }
    if (isPaused || audioData) {
      return <Play className="w-4 h-4" />;
    }
    return <Volume2 className="w-4 h-4" />;
  };

  const getVoiceButtonTitle = () => {
    if (isGeneratingVoice) return "Generating voice...";
    if (isPlaying) return "Pause";
    if (isPaused) return "Resume";
    if (audioData) return "Play";
    return "Generate voice";
  };

  if (!results) {
    return (
      <div className={cn("border rounded-2xl p-6", containerStyles, className)}>
        <div className={cn("flex items-center gap-2", loadingTextStyles)}>
          <Sparkles className="w-5 h-5" />
          <span>AI explanation will appear after analysis</span>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("border rounded-2xl p-6", containerStyles, className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "p-1.5 rounded-lg",
              isDark ? "bg-purple-500/20" : "bg-blue-50",
            )}
          >
            <Bot className={cn("w-4 h-4", accentColor)} />
          </div>
          <h3 className={cn("text-base font-semibold", headerTextStyles)}>
            AI Explanation
          </h3>
          <span className={cn("text-xs px-2 py-0.5 rounded-full", badgeStyles)}>
            Llama 3.3 70B
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Regenerate button */}
          <button
            onClick={() => generateExplanation(true)}
            disabled={isLoading || isGeneratingVoice}
            className={cn(
              "p-2 rounded-lg transition-colors disabled:opacity-50",
              buttonStyles,
            )}
            title="Regenerate explanation"
          >
            <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
          </button>

          {/* Voice Controls - Play/Pause button */}
          {explanation && (
            <button
              onClick={handleVoiceClick}
              disabled={isLoading}
              className={cn(
                "p-2 rounded-lg transition-all duration-200",
                isPlaying
                  ? isDark
                    ? "bg-purple-500 text-white"
                    : "bg-blue-600 text-white"
                  : audioData || isPaused
                    ? isDark
                      ? "bg-purple-500/20 text-purple-400"
                      : "bg-blue-50 text-blue-600"
                    : buttonStyles,
                isLoading && "opacity-50 cursor-not-allowed",
              )}
              title={getVoiceButtonTitle()}
            >
              {getVoiceButtonContent()}
            </button>
          )}
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className={cn("mb-4 p-3 border rounded-lg text-sm", errorStyles)}>
          {error}
        </div>
      )}

      {/* Loading State */}
      {isLoading && explanation === "" && (
        <div className={cn("flex items-center gap-3 py-4", loadingTextStyles)}>
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Generating AI explanation...</span>
        </div>
      )}

      {/* Streaming Text */}
      {explanation && (
        <div className="prose prose-sm max-w-none">
          <div
            className={cn(
              "text-sm leading-relaxed whitespace-pre-wrap",
              contentStyles,
            )}
          >
            {explanation}
            {isLoading && (
              <span
                className={cn(
                  "inline-block w-2 h-4 ml-1 animate-pulse rounded-sm",
                  isDark ? "bg-purple-400" : "bg-blue-500",
                )}
              />
            )}
          </div>
        </div>
      )}

      {/* Powered By */}
      <div
        className={cn(
          "mt-4 pt-4 border-t flex items-center gap-2 text-xs",
          footerStyles,
        )}
      >
        <span>Powered by</span>
        <span className={accentColor}>Cerebras Cloud</span>
        {audioData && (
          <>
            <span>+</span>
            <span className={secondaryAccent}>Voice AI</span>
          </>
        )}
      </div>
    </div>
  );
}
