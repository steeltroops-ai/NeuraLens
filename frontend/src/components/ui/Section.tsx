'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/**
 * Section Component
 * 
 * Implements MediLens Layout Patterns:
 * - Alternating white/surface-secondary backgrounds (Requirements 18.3)
 * - 80px vertical padding on desktop, 48px on mobile (Requirements 18.4)
 * - Generous whitespace following the "Breathe" design principle (Requirements 18.5)
 * 
 * @example
 * ```tsx
 * <Section variant="white">
 *   <h2>Section Title</h2>
 *   <p>Section content...</p>
 * </Section>
 * 
 * <Section variant="secondary">
 *   <h2>Another Section</h2>
 *   <p>More content...</p>
 * </Section>
 * ```
 */

export interface SectionProps {
    /** Section content */
    children: ReactNode;
    /** Background variant - white or secondary (surface-secondary) */
    variant?: 'white' | 'secondary';
    /** Additional CSS classes */
    className?: string;
    /** Section ID for navigation */
    id?: string;
    /** ARIA label for accessibility */
    'aria-label'?: string;
    /** ARIA labelledby for accessibility */
    'aria-labelledby'?: string;
    /** Whether to use container width constraints */
    contained?: boolean;
    /** Container size when contained is true */
    containerSize?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
    /** Custom padding override */
    padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
}

/**
 * Container width classes based on MediLens Design System
 * Requirements: 18.1
 */
const containerSizeClasses = {
    sm: 'max-w-[640px]',
    md: 'max-w-[768px]',
    lg: 'max-w-[1024px]',
    xl: 'max-w-[1280px]',
    '2xl': 'max-w-[1440px]',
} as const;

/**
 * Padding classes based on MediLens Design System
 * Requirements: 18.4
 */
const paddingClasses = {
    none: '',
    sm: 'py-6 lg:py-10',
    md: 'py-8 lg:py-12',
    lg: 'py-12 lg:py-20', // 48px mobile, 80px desktop (default)
    xl: 'py-16 lg:py-24',
} as const;

export function Section({
    children,
    variant = 'white',
    className,
    id,
    'aria-label': ariaLabel,
    'aria-labelledby': ariaLabelledby,
    contained = false,
    containerSize = '2xl',
    padding = 'lg',
}: SectionProps) {
    // Background color based on variant (Requirements 18.3)
    const bgClass = variant === 'white' ? 'bg-white' : 'bg-[#F2F2F7]';

    // Padding class (Requirements 18.4)
    const paddingClass = paddingClasses[padding];

    return (
        <section
            id={id}
            className={cn(
                bgClass,
                paddingClass,
                'w-full',
                className
            )}
            aria-label={ariaLabel}
            aria-labelledby={ariaLabelledby}
        >
            {contained ? (
                <div className={cn(
                    'mx-auto px-4 sm:px-6 lg:px-8',
                    containerSizeClasses[containerSize]
                )}>
                    {children}
                </div>
            ) : (
                children
            )}
        </section>
    );
}

/**
 * SectionRhythm Component
 * 
 * Wrapper that automatically alternates section backgrounds
 * for visual rhythm (Requirements 18.3)
 * 
 * @example
 * ```tsx
 * <SectionRhythm>
 *   <section>First section (white)</section>
 *   <section>Second section (secondary)</section>
 *   <section>Third section (white)</section>
 * </SectionRhythm>
 * ```
 */
export interface SectionRhythmProps {
    children: ReactNode;
    className?: string;
    /** Starting variant for the first section */
    startWith?: 'white' | 'secondary';
}

export function SectionRhythm({
    children,
    className,
    startWith = 'white',
}: SectionRhythmProps) {
    return (
        <div
            className={cn(
                'section-rhythm',
                startWith === 'secondary' && '[&>section:nth-child(odd)]:bg-[#F2F2F7] [&>section:nth-child(even)]:bg-white',
                className
            )}
        >
            {children}
        </div>
    );
}

export default Section;
