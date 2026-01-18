'use client';

import { Activity } from 'lucide-react';

interface LogoProps {
  /** Show the MediLens text next to the icon */
  showText?: boolean;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Additional className for the container */
  className?: string;
}

/**
 * MediLens Logo Component
 * Reusable logo with Activity icon - use this everywhere for consistency
 */
export function Logo({ showText = true, size = 'md', className = '' }: LogoProps) {
  const sizes = {
    sm: { icon: 'h-4 w-4', text: 'text-[13px]' },
    md: { icon: 'h-5 w-5', text: 'text-[15px]' },
    lg: { icon: 'h-6 w-6', text: 'text-[17px]' },
  };

  const s = sizes[size];

  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <Activity className={`${s.icon} text-red-500`} strokeWidth={2} />
      {showText && (
        <span className={`${s.text} font-semibold tracking-tight text-white`}>
          MediLens
        </span>
      )}
    </div>
  );
}

export default Logo;
