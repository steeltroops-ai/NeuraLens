/**
 * MediLens Animation System
 * 
 * Comprehensive animation utilities following the MediLens Design System.
 * Uses Apple-inspired easing (ease-out-quint) for all transitions.
 * 
 * Requirements: 16.1, 16.2, 16.3, 16.4
 */

import type { Variants, Transition, TargetAndTransition } from 'framer-motion';

// ============================================================================
// ANIMATION EASING FUNCTIONS
// ============================================================================

/**
 * MediLens easing functions - Apple-inspired curves
 * Primary easing: ease-out-quint (cubic-bezier(0.22, 1, 0.36, 1))
 */
export const easing = {
    /** Primary MediLens easing - smooth deceleration */
    outQuint: [0.22, 1, 0.36, 1] as const,
    /** For modals and dialogs - balanced */
    inOutCubic: [0.65, 0, 0.35, 1] as const,
    /** Playful interactions - slight overshoot */
    spring: [0.175, 0.885, 0.32, 1.275] as const,
    /** Standard ease-out */
    out: [0, 0, 0.2, 1] as const,
    /** Standard ease-in */
    in: [0.4, 0, 1, 1] as const,
} as const;

/**
 * CSS easing strings for use in Tailwind/CSS
 */
export const cssEasing = {
    outQuint: 'cubic-bezier(0.22, 1, 0.36, 1)',
    inOutCubic: 'cubic-bezier(0.65, 0, 0.35, 1)',
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
} as const;

// ============================================================================
// ANIMATION DURATION SCALE
// ============================================================================

/**
 * MediLens duration scale in seconds
 * - Fast: 0.15s - Micro-interactions (hover, focus)
 * - Normal: 0.3s - Standard transitions (cards, buttons)
 * - Slow: 0.5s - Page transitions, modals
 * - Slower: 0.75s - Complex animations, reveals
 */
export const duration = {
    instant: 0,
    fast: 0.15,
    normal: 0.3,
    slow: 0.5,
    slower: 0.75,
} as const;

/**
 * Duration in milliseconds for CSS usage
 */
export const durationMs = {
    instant: 0,
    fast: 150,
    normal: 300,
    slow: 500,
    slower: 750,
} as const;

// ============================================================================
// FRAMER MOTION TRANSITION PRESETS
// ============================================================================

/**
 * Standard MediLens transition using ease-out-quint
 */
export const medilensTransition: Transition = {
    duration: duration.normal,
    ease: easing.outQuint,
};

/**
 * Fast transition for micro-interactions
 */
export const fastTransition: Transition = {
    duration: duration.fast,
    ease: easing.outQuint,
};

/**
 * Slow transition for page transitions and modals
 */
export const slowTransition: Transition = {
    duration: duration.slow,
    ease: easing.outQuint,
};

/**
 * Spring transition for playful interactions
 */
export const springTransition: Transition = {
    type: 'spring',
    stiffness: 400,
    damping: 30,
};

// ============================================================================
// FRAMER MOTION VARIANTS
// ============================================================================

/**
 * Fade In Up animation variant
 * Used for content reveals and list items
 */
export const fadeInUp: Variants = {
    initial: {
        opacity: 0,
        y: 20,
    },
    animate: {
        opacity: 1,
        y: 0,
        transition: {
            duration: duration.slow,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        y: -10,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
};

/**
 * Scale In animation variant
 * Used for modals, cards, and popups
 */
export const scaleIn: Variants = {
    initial: {
        opacity: 0,
        scale: 0.95,
    },
    animate: {
        opacity: 1,
        scale: 1,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        scale: 0.95,
        transition: {
            duration: duration.fast,
            ease: easing.outQuint,
        },
    },
};

/**
 * Fade In animation variant
 * Simple opacity transition
 */
export const fadeIn: Variants = {
    initial: {
        opacity: 0,
    },
    animate: {
        opacity: 1,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        transition: {
            duration: duration.fast,
            ease: easing.outQuint,
        },
    },
};

/**
 * Slide In from Right animation variant
 * Used for page transitions and sidebars
 */
export const slideInRight: Variants = {
    initial: {
        opacity: 0,
        x: 30,
    },
    animate: {
        opacity: 1,
        x: 0,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        x: -30,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
};

/**
 * Slide In from Left animation variant
 */
export const slideInLeft: Variants = {
    initial: {
        opacity: 0,
        x: -30,
    },
    animate: {
        opacity: 1,
        x: 0,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        x: 30,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
};

// ============================================================================
// STAGGER CHILDREN VARIANTS
// ============================================================================

/**
 * Container variant for staggered children animations
 */
export const staggerContainer: Variants = {
    initial: {},
    animate: {
        transition: {
            staggerChildren: 0.1,
            delayChildren: 0.1,
        },
    },
    exit: {
        transition: {
            staggerChildren: 0.05,
            staggerDirection: -1,
        },
    },
};

/**
 * Fast stagger container (0.05s between children)
 */
export const staggerContainerFast: Variants = {
    initial: {},
    animate: {
        transition: {
            staggerChildren: 0.05,
            delayChildren: 0.05,
        },
    },
};

/**
 * Slow stagger container (0.15s between children)
 */
export const staggerContainerSlow: Variants = {
    initial: {},
    animate: {
        transition: {
            staggerChildren: 0.15,
            delayChildren: 0.15,
        },
    },
};

/**
 * Child item variant for use with stagger containers
 */
export const staggerItem: Variants = {
    initial: {
        opacity: 0,
        y: 20,
    },
    animate: {
        opacity: 1,
        y: 0,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        y: -10,
        transition: {
            duration: duration.fast,
            ease: easing.outQuint,
        },
    },
};

// ============================================================================
// HOVER AND TAP ANIMATIONS
// ============================================================================

/**
 * Hover lift animation for buttons
 * -translate-y-0.5 (2px lift)
 */
export const hoverLiftButton: TargetAndTransition = {
    y: -2,
    transition: {
        duration: duration.fast,
        ease: easing.outQuint,
    },
};

/**
 * Hover lift animation for cards
 * -translate-y-1 (4px lift)
 */
export const hoverLiftCard: TargetAndTransition = {
    y: -4,
    transition: {
        duration: duration.normal,
        ease: easing.outQuint,
    },
};

/**
 * Tap/Active scale animation
 * scale-[0.98] feedback
 */
export const tapScale: TargetAndTransition = {
    scale: 0.98,
    transition: {
        duration: duration.fast,
        ease: easing.outQuint,
    },
};

/**
 * Combined hover and tap props for buttons
 */
export const buttonAnimationProps = {
    whileHover: hoverLiftButton,
    whileTap: tapScale,
};

/**
 * Combined hover and tap props for cards
 */
export const cardAnimationProps = {
    whileHover: hoverLiftCard,
    whileTap: { scale: 0.99 },
};

// ============================================================================
// MODAL AND OVERLAY VARIANTS
// ============================================================================

/**
 * Modal backdrop animation
 */
export const modalBackdrop: Variants = {
    initial: {
        opacity: 0,
    },
    animate: {
        opacity: 1,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        transition: {
            duration: duration.fast,
            ease: easing.outQuint,
        },
    },
};

/**
 * Modal content animation
 */
export const modalContent: Variants = {
    initial: {
        opacity: 0,
        scale: 0.95,
        y: -20,
    },
    animate: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        scale: 0.95,
        y: -10,
        transition: {
            duration: duration.fast,
            ease: easing.outQuint,
        },
    },
};

// ============================================================================
// TOAST AND NOTIFICATION VARIANTS
// ============================================================================

/**
 * Toast slide in from right
 */
export const toastSlideIn: Variants = {
    initial: {
        opacity: 0,
        x: 100,
        scale: 0.95,
    },
    animate: {
        opacity: 1,
        x: 0,
        scale: 1,
        transition: {
            duration: duration.normal,
            ease: easing.outQuint,
        },
    },
    exit: {
        opacity: 0,
        x: 100,
        scale: 0.95,
        transition: {
            duration: duration.fast,
            ease: easing.outQuint,
        },
    },
};

// ============================================================================
// NRI SCORE ANIMATION
// ============================================================================

/**
 * NRI Score reveal animation
 * Special animation for the Neurological Risk Index score display
 */
export const nriScoreReveal: Variants = {
    initial: {
        opacity: 0,
        scale: 0.8,
        y: 20,
    },
    animate: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: {
            duration: duration.slower,
            ease: easing.outQuint,
        },
    },
};

// ============================================================================
// PROGRESS ANIMATION
// ============================================================================

/**
 * Progress bar fill animation
 */
export const progressFill: Variants = {
    initial: {
        width: '0%',
    },
    animate: (progress: number) => ({
        width: `${progress}%`,
        transition: {
            duration: duration.slow,
            ease: easing.outQuint,
        },
    }),
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Create a custom fade in up variant with specified distance
 */
export function createFadeInUp(distance: number = 20): Variants {
    return {
        initial: { opacity: 0, y: distance },
        animate: {
            opacity: 1,
            y: 0,
            transition: {
                duration: duration.slow,
                ease: easing.outQuint,
            },
        },
        exit: {
            opacity: 0,
            y: -distance / 2,
            transition: {
                duration: duration.normal,
                ease: easing.outQuint,
            },
        },
    };
}

/**
 * Create a custom stagger container with specified delay
 */
export function createStaggerContainer(staggerDelay: number = 0.1): Variants {
    return {
        initial: {},
        animate: {
            transition: {
                staggerChildren: staggerDelay,
                delayChildren: staggerDelay,
            },
        },
    };
}

/**
 * Create a delayed animation variant
 */
export function withDelay<T extends Variants>(variants: T, delay: number): T {
    const result = { ...variants } as Record<string, unknown>;
    if (result.animate && typeof result.animate === 'object') {
        result.animate = {
            ...(result.animate as object),
            transition: {
                ...((result.animate as Record<string, unknown>).transition as object),
                delay,
            },
        };
    }
    return result as T;
}

// ============================================================================
// REDUCED MOTION UTILITIES
// ============================================================================

/**
 * Check if user prefers reduced motion
 * Returns true if reduced motion is preferred
 */
export function prefersReducedMotion(): boolean {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Get animation variants that respect reduced motion preference
 * Returns static variants if reduced motion is preferred
 */
export function getReducedMotionVariants(variants: Variants): Variants {
    if (prefersReducedMotion()) {
        return {
            initial: { opacity: 1 },
            animate: { opacity: 1 },
            exit: { opacity: 0 },
        };
    }
    return variants;
}

/**
 * Get transition that respects reduced motion preference
 * Returns instant transition if reduced motion is preferred
 */
export function getReducedMotionTransition(transition: Transition): Transition {
    if (prefersReducedMotion()) {
        return { duration: 0 };
    }
    return transition;
}

// ============================================================================
// EXPORTS
// ============================================================================

export const animations = {
    // Easing
    easing,
    cssEasing,

    // Duration
    duration,
    durationMs,

    // Transitions
    medilensTransition,
    fastTransition,
    slowTransition,
    springTransition,

    // Variants
    fadeInUp,
    scaleIn,
    fadeIn,
    slideInRight,
    slideInLeft,

    // Stagger
    staggerContainer,
    staggerContainerFast,
    staggerContainerSlow,
    staggerItem,

    // Hover/Tap
    hoverLiftButton,
    hoverLiftCard,
    tapScale,
    buttonAnimationProps,
    cardAnimationProps,

    // Modal
    modalBackdrop,
    modalContent,

    // Toast
    toastSlideIn,

    // Special
    nriScoreReveal,
    progressFill,

    // Utilities
    createFadeInUp,
    createStaggerContainer,
    withDelay,
    prefersReducedMotion,
    getReducedMotionVariants,
    getReducedMotionTransition,
} as const;

export default animations;
