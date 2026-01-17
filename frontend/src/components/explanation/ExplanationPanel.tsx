'use client';

import { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, Loader2, Sparkles, RefreshCw, Bot } from 'lucide-react';
import { cn } from '@/utils/cn';

interface ExplanationPanelProps {
  pipeline: string;
  results: unknown;
  patientContext?: {
    age?: number;
    sex?: string;
    history?: string[];
  };
  className?: string;
  theme?: 'light' | 'dark';
}

export function ExplanationPanel({
  pipeline,
  results,
  patientContext,
  className = '',
  theme = 'light'
}: ExplanationPanelProps) {
  const [explanation, setExplanation] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioData, setAudioData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const isDark = theme === 'dark';

  const generateExplanation = async () => {
    if (!results) return;
    
    setIsLoading(true);
    setExplanation('');
    setError(null);
    setAudioData(null);

    try {
      const response = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline,
          results,
          patient_context: patientContext,
          voice_output: true,
          voice_provider: 'elevenlabs'
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate explanation');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.text) {
                setExplanation(prev => prev + data.text);
              }
              if (data.audio_base64) {
                setAudioData(data.audio_base64);
              }
              if (data.error) {
                setError(data.error);
              }
              if (data.done) {
                setIsLoading(false);
              }
            } catch {
              // Ignore parsing errors for incomplete JSON
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsLoading(false);
    }
  };

  // Generate explanation when results change
  useEffect(() => {
    if (results) {
      generateExplanation();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, pipeline]);

  const playAudio = () => {
    if (audioData) {
      const audio = new Audio(`data:audio/mp3;base64,${audioData}`);
      audioRef.current = audio;
      audio.onplay = () => setIsPlaying(true);
      audio.onended = () => setIsPlaying(false);
      audio.onerror = () => setIsPlaying(false);
      audio.play();
    }
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  // Light theme styles
  const containerStyles = isDark
    ? 'bg-gradient-to-br from-zinc-900 to-zinc-950 border-zinc-800'
    : 'bg-white border-zinc-200 shadow-sm';

  const headerTextStyles = isDark
    ? 'text-white'
    : 'text-zinc-900';

  const badgeStyles = isDark
    ? 'text-zinc-400 bg-zinc-800'
    : 'text-zinc-500 bg-zinc-100';

  const buttonStyles = isDark
    ? 'text-zinc-400 hover:text-white hover:bg-zinc-800'
    : 'text-zinc-500 hover:text-zinc-700 hover:bg-zinc-100';

  const accentButtonStyles = isDark
    ? 'bg-purple-600 hover:bg-purple-700 text-white'
    : 'bg-blue-600 hover:bg-blue-700 text-white';

  const errorStyles = isDark
    ? 'bg-red-500/10 border-red-500/30 text-red-400'
    : 'bg-red-50 border-red-200 text-red-600';

  const loadingTextStyles = isDark
    ? 'text-zinc-400'
    : 'text-zinc-500';

  const contentStyles = isDark
    ? 'text-zinc-300'
    : 'text-zinc-700';

  const footerStyles = isDark
    ? 'border-zinc-800 text-zinc-500'
    : 'border-zinc-200 text-zinc-400';

  const accentColor = isDark ? 'text-purple-400' : 'text-blue-600';
  const secondaryAccent = isDark ? 'text-blue-400' : 'text-indigo-600';

  if (!results) {
    return (
      <div className={cn(
        'border rounded-2xl p-6',
        containerStyles,
        className
      )}>
        <div className={cn('flex items-center gap-2', loadingTextStyles)}>
          <Sparkles className="w-5 h-5" />
          <span>AI explanation will appear after analysis</span>
        </div>
      </div>
    );
  }

  return (
    <div className={cn(
      'border rounded-2xl p-6',
      containerStyles,
      className
    )}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={cn(
            'p-1.5 rounded-lg',
            isDark ? 'bg-purple-500/20' : 'bg-blue-50'
          )}>
            <Bot className={cn('w-4 h-4', accentColor)} />
          </div>
          <h3 className={cn('text-base font-semibold', headerTextStyles)}>
            AI Explanation
          </h3>
          <span className={cn(
            'text-xs px-2 py-0.5 rounded-full',
            badgeStyles
          )}>
            Llama 3.3 70B
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Regenerate button */}
          <button
            onClick={generateExplanation}
            disabled={isLoading}
            className={cn(
              'p-2 rounded-lg transition-colors disabled:opacity-50',
              buttonStyles
            )}
            title="Regenerate explanation"
          >
            <RefreshCw className={cn('w-4 h-4', isLoading && 'animate-spin')} />
          </button>
          
          {/* Voice button */}
          {audioData && (
            <button
              onClick={isPlaying ? stopAudio : playAudio}
              className={cn(
                'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
                accentButtonStyles
              )}
            >
              {isPlaying ? (
                <>
                  <VolumeX className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  <Volume2 className="w-4 h-4" />
                  Listen
                </>
              )}
            </button>
          )}
        </div>
      </div>
      
      {/* Error State */}
      {error && (
        <div className={cn(
          'mb-4 p-3 border rounded-lg text-sm',
          errorStyles
        )}>
          {error}
        </div>
      )}
      
      {/* Loading State */}
      {isLoading && explanation === '' && (
        <div className={cn('flex items-center gap-3 py-4', loadingTextStyles)}>
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Generating AI explanation...</span>
        </div>
      )}
      
      {/* Streaming Text */}
      {explanation && (
        <div className="prose prose-sm max-w-none">
          <div className={cn(
            'text-sm leading-relaxed whitespace-pre-wrap',
            contentStyles
          )}>
            {explanation}
            {isLoading && (
              <span className={cn(
                'inline-block w-2 h-4 ml-1 animate-pulse rounded-sm',
                isDark ? 'bg-purple-400' : 'bg-blue-500'
              )} />
            )}
          </div>
        </div>
      )}
      
      {/* Powered By */}
      <div className={cn(
        'mt-4 pt-4 border-t flex items-center gap-2 text-xs',
        footerStyles
      )}>
        <span>Powered by</span>
        <span className={accentColor}>Cerebras Cloud</span>
        {audioData && (
          <>
            <span>+</span>
            <span className={secondaryAccent}>ElevenLabs</span>
          </>
        )}
      </div>
    </div>
  );
}
