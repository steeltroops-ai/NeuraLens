'use client';

import React, { useState, useEffect } from 'react';

/**
 * Hydration-safe geometry components that prevent server/client rendering mismatches
 * by using client-only rendering and deterministic calculations.
 */

// Client-only wrapper to prevent hydration mismatches
function ClientOnly({ children }: { children: React.ReactNode }) {
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  if (!hasMounted) {
    return (
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='h-16 w-32 animate-pulse rounded-lg bg-gray-200' />
      </div>
    );
  }

  return <>{children}</>;
}

// Hydration-safe speech waveform with deterministic heights
export function SpeechWaveform() {
  // Pre-calculate deterministic heights to avoid server/client mismatches
  const waveformHeights = Array.from({ length: 20 }, (_, i) => {
    const calculatedHeight = 20 + Math.sin(i * 0.5) * 15;
    return Math.round(calculatedHeight * 100) / 100; // Round to 2 decimal places for consistency
  });

  return (
    <ClientOnly>
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='flex items-end space-x-1'>
          {waveformHeights.map((height, i) => (
            <div
              key={i}
              className='animate-pulse rounded-sm bg-blue-500 opacity-60'
              style={{
                width: '4px',
                height: `${height}px`,
                animationDelay: `${i * 0.1}s`,
                animationDuration: '1.5s',
              }}
            />
          ))}
        </div>
      </div>
    </ClientOnly>
  );
}

export function RetinalEye() {
  return (
    <ClientOnly>
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='relative'>
          <div className='h-24 w-24 rounded-full border-4 border-blue-500 opacity-60'>
            <div className='absolute left-1/2 top-1/2 h-12 w-12 -translate-x-1/2 -translate-y-1/2 transform animate-pulse rounded-full bg-blue-500 opacity-40' />
          </div>
          <div className='absolute inset-0 animate-ping rounded-full border border-blue-300 opacity-30' />
        </div>
      </div>
    </ClientOnly>
  );
}

export function HandKinematics() {
  // Pre-calculate animation delays to ensure consistency
  const animationDelays = Array.from({ length: 9 }, (_, i) => `${i * 0.2}s`);

  return (
    <ClientOnly>
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='grid grid-cols-3 gap-2'>
          {animationDelays.map((delay, i) => (
            <div
              key={i}
              className='h-3 w-3 animate-bounce rounded-full bg-blue-500 opacity-60'
              style={{
                animationDelay: delay,
                animationDuration: '2s',
              }}
            />
          ))}
        </div>
      </div>
    </ClientOnly>
  );
}

export function BrainNeural() {
  return (
    <ClientOnly>
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='relative'>
          <div
            className='h-20 w-20 animate-spin rounded-full border-2 border-blue-500 opacity-60'
            style={{ animationDuration: '8s' }}
          >
            <div className='absolute left-2 top-2 h-2 w-2 animate-pulse rounded-full bg-blue-500' />
            <div
              className='absolute right-2 top-2 h-2 w-2 animate-pulse rounded-full bg-blue-500'
              style={{ animationDelay: '0.5s' }}
            />
            <div
              className='absolute bottom-2 left-2 h-2 w-2 animate-pulse rounded-full bg-blue-500'
              style={{ animationDelay: '1s' }}
            />
            <div
              className='absolute bottom-2 right-2 h-2 w-2 animate-pulse rounded-full bg-blue-500'
              style={{ animationDelay: '1.5s' }}
            />
          </div>
        </div>
      </div>
    </ClientOnly>
  );
}

export function NRIFusion() {
  return (
    <ClientOnly>
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='relative'>
          <div className='h-16 w-16 rotate-45 animate-pulse border-2 border-blue-500 opacity-60' />
          <div
            className='absolute left-1/2 top-1/2 h-12 w-12 -translate-x-1/2 -translate-y-1/2 rotate-45 transform animate-pulse border-2 border-blue-400 opacity-40'
            style={{ animationDelay: '0.5s' }}
          />
          <div
            className='absolute left-1/2 top-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 rotate-45 transform animate-pulse border-2 border-blue-300 opacity-30'
            style={{ animationDelay: '1s' }}
          />
        </div>
      </div>
    </ClientOnly>
  );
}

export function MultiModalNetwork() {
  // Pre-calculate animation delays to ensure consistency
  const animationDelays = Array.from({ length: 16 }, (_, i) => `${(i % 4) * 0.3}s`);

  return (
    <ClientOnly>
      <div className='absolute inset-0 flex h-full w-full items-center justify-center'>
        <div className='grid grid-cols-4 gap-3'>
          {animationDelays.map((delay, i) => (
            <div
              key={i}
              className='h-2 w-2 animate-ping rounded-full bg-blue-500 opacity-60'
              style={{
                animationDelay: delay,
                animationDuration: '2s',
              }}
            />
          ))}
        </div>
      </div>
    </ClientOnly>
  );
}

export default {
  SpeechWaveform,
  RetinalEye,
  HandKinematics,
  BrainNeural,
  NRIFusion,
  MultiModalNetwork,
};
