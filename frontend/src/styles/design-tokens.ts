/**
 * MediLens Design System - TypeScript Design Tokens
 * 
 * A comprehensive design philosophy inspired by Apple's Human Interface Guidelines,
 * uniquely adapted for healthcare diagnostics.
 * 
 * Core Philosophy: Clinical Clarity — the intersection of Apple's refined minimalism
 * with healthcare's need for trust and precision.
 */

// ============================================================================
// COLOR SYSTEM
// ============================================================================

/**
 * Primary Palette — MediLens Blue
 * The signature MediLens blue conveys trust, intelligence, and medical professionalism.
 */
export const medilensBlue = {
    50: '#E8F4FF',   // Subtle backgrounds
    100: '#C5E4FF',  // Hover states
    200: '#9DD1FF',  // Light accents
    300: '#6BBAFF',  // Secondary elements
    400: '#3AA3FF',  // Interactive elements
    500: '#007AFF',  // Primary brand color
    600: '#0062CC',  // Hover on primary
    700: '#004A99',  // Active states
    800: '#003366',  // Dark accents
    900: '#001A33',  // Deep backgrounds
} as const;

/**
 * Semantic Status Colors
 * Used for clinical status indicators and feedback
 */
export const statusColors = {
    healthy: '#34C759',    // Low risk, positive results
    caution: '#FF9500',    // Moderate risk, attention needed
    alert: '#FF3B30',      // High risk, immediate attention
    info: '#5AC8FA',       // Informational, neutral
    processing: '#AF52DE', // AI processing, analysis
} as const;

/**
 * NRI Risk Gradient Colors
 * Neurological Risk Index color scale for risk visualization
 */
export const nriColors = {
    minimal: '#34C759',   // 0-25: Minimal risk
    low: '#30D158',       // 26-40: Low risk
    moderate: '#FFD60A',  // 41-55: Moderate risk
    elevated: '#FF9F0A',  // 56-70: Elevated risk
    high: '#FF6B6B',      // 71-85: High risk
    critical: '#FF3B30',  // 86-100: Critical risk
} as const;

/**
 * Surface & Background Colors
 * Light theme surface colors for cards, backgrounds, and dividers
 */
export const surfaceColors = {
    primary: '#FFFFFF',    // Cards, modals
    secondary: '#F2F2F7',  // Page backgrounds
    tertiary: '#E5E5EA',   // Dividers, borders
    elevated: '#FFFFFF',   // Elevated cards
} as const;

/**
 * Text Colors
 * Semantic text colors for different content hierarchies
 */
export const textColors = {
    primary: '#000000',    // Headlines, primary content
    secondary: '#3C3C43',  // Body text, descriptions
    tertiary: '#8E8E93',   // Captions, metadata
    quaternary: '#C7C7CC', // Placeholders, disabled
} as const;

// Combined colors export
export const colors = {
    medilensBlue,
    status: statusColors,
    nri: nriColors,
    surface: surfaceColors,
    text: textColors,
} as const;

// ============================================================================
// TYPOGRAPHY SYSTEM
// ============================================================================

/**
 * Font Family Stack
 * Apple-optimized system font stack for optimal rendering
 */
export const fontFamily = {
    system: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', system-ui, sans-serif",
    mono: "'SF Mono', 'Monaco', 'Menlo', 'JetBrains Mono', monospace",
} as const;

/**
 * Typography Scale
 * Apple-inspired type scale for consistent hierarchy
 */
export const typographyScale = {
    display: { size: '48px', weight: 700, lineHeight: 1.1 },    // Hero headlines
    title1: { size: '34px', weight: 700, lineHeight: 1.2 },     // Page titles
    title2: { size: '28px', weight: 600, lineHeight: 1.25 },    // Section headers
    title3: { size: '22px', weight: 600, lineHeight: 1.3 },     // Card titles
    headline: { size: '17px', weight: 600, lineHeight: 1.4 },   // Emphasized body
    body: { size: '17px', weight: 400, lineHeight: 1.5 },       // Primary content
    callout: { size: '16px', weight: 400, lineHeight: 1.5 },    // Secondary content
    subhead: { size: '15px', weight: 400, lineHeight: 1.4 },    // Supporting text
    footnote: { size: '13px', weight: 400, lineHeight: 1.35 },  // Captions, metadata
    caption: { size: '12px', weight: 400, lineHeight: 1.3 },    // Labels, timestamps
} as const;

/**
 * Letter Spacing
 * Tightened for headlines, normal for body text
 */
export const letterSpacing = {
    tight: '-0.02em',  // Headlines
    normal: '0',       // Body text
} as const;

/**
 * Maximum text width for optimal readability
 */
export const maxTextWidth = '65ch';

/**
 * Fluid Typography Scale
 * Uses CSS clamp() for responsive scaling
 */
export const fluidTypography = {
    display: 'clamp(32px, 5vw + 1rem, 48px)',    // 32px → 48px
    title1: 'clamp(28px, 4vw + 0.5rem, 34px)',   // 28px → 34px
    title2: 'clamp(22px, 3vw + 0.5rem, 28px)',   // 22px → 28px
    title3: 'clamp(18px, 2.5vw + 0.25rem, 22px)', // 18px → 22px
} as const;

export const typography = {
    fontFamily,
    scale: typographyScale,
    letterSpacing,
    maxWidth: maxTextWidth,
    fluid: fluidTypography,
} as const;

// ============================================================================
// SPACING SYSTEM (8px Grid)
// ============================================================================

/**
 * Spacing Scale
 * All spacing derives from an 8px base unit for visual harmony
 */
export const spacing = {
    1: '4px',    // Micro: icon padding
    2: '8px',    // Tight: inline elements
    3: '12px',   // Compact: form fields
    4: '16px',   // Standard: component padding
    5: '20px',   // Comfortable: card padding
    6: '24px',   // Relaxed: section gaps
    8: '32px',   // Generous: major sections
    10: '40px',  // Spacious: page margins
    12: '48px',  // Expansive: hero spacing
    16: '64px',  // Dramatic: section breaks
    20: '80px',  // Grand: page sections
    24: '96px',  // Monumental: hero areas
} as const;

/**
 * Component-specific spacing
 * Pre-defined spacing for common component patterns
 */
export const componentSpacing = {
    button: {
        compact: { x: '24px', y: '12px' },
        standard: { x: '32px', y: '16px' },
    },
    card: {
        standard: '24px',
        featured: '32px',
    },
    formField: {
        height: '48px', // Minimum touch-friendly height
    },
    section: {
        desktop: '80px',
        mobile: '48px',
    },
} as const;

// ============================================================================
// ANIMATION SYSTEM
// ============================================================================

/**
 * Animation Easing Functions
 * Apple-inspired easing curves for smooth, natural motion
 */
export const easing = {
    outQuint: 'cubic-bezier(0.22, 1, 0.36, 1)',      // Primary - smooth deceleration
    inOutCubic: 'cubic-bezier(0.65, 0, 0.35, 1)',   // Modals - balanced
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)', // Playful - slight overshoot
} as const;

/**
 * Animation Duration Scale
 * Consistent timing for different interaction types
 */
export const duration = {
    instant: '0ms',    // State changes
    fast: '150ms',     // Micro-interactions (hover, focus)
    normal: '300ms',   // Standard transitions (cards, buttons)
    slow: '500ms',     // Page transitions, modals
    slower: '750ms',   // Complex animations, reveals
} as const;

/**
 * Framer Motion Variants
 * Pre-defined animation variants for common patterns
 */
export const motionVariants = {
    fadeInUp: {
        initial: { opacity: 0, y: 20 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
    },
    scaleIn: {
        initial: { opacity: 0, scale: 0.95 },
        animate: { opacity: 1, scale: 1 },
        transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] },
    },
    staggerChildren: {
        show: { transition: { staggerChildren: 0.1 } },
    },
} as const;

export const animation = {
    easing,
    duration,
    variants: motionVariants,
} as const;

// ============================================================================
// BORDER RADIUS SYSTEM
// ============================================================================

/**
 * Border Radius Scale
 * Consistent rounding for different component sizes
 */
export const borderRadius = {
    sm: '4px',     // Small elements
    md: '8px',     // Buttons, small cards (rounded-lg)
    lg: '12px',    // Buttons (rounded-xl)
    xl: '16px',    // Cards (rounded-2xl)
    '2xl': '24px', // Featured cards (rounded-3xl)
    full: '9999px', // Pills, badges
} as const;

// ============================================================================
// SHADOW SYSTEM
// ============================================================================

/**
 * Shadow Scale
 * Apple-inspired shadows for depth and elevation
 */
export const shadows = {
    apple: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    appleHover: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    medical: '0 4px 14px 0 rgba(0, 122, 255, 0.15)',
    medicalHover: '0 8px 25px 0 rgba(0, 122, 255, 0.25)',
    nri: '0 8px 30px 0 rgba(0, 122, 255, 0.12)',
    glass: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
} as const;

// ============================================================================
// CONTAINER WIDTHS
// ============================================================================

/**
 * Container Width Scale
 * Responsive container widths for different layouts
 * Requirements: 18.1
 */
export const containerWidths = {
    sm: '640px',   // Narrow content
    md: '768px',   // Standard content
    lg: '1024px',  // Wide content
    xl: '1280px',  // Full-width content
    '2xl': '1440px', // Maximum width
} as const;

/**
 * Layout Widths
 * Specific widths for layout components
 * Requirements: 18.2
 */
export const layoutWidths = {
    sidebar: '280px',           // Dashboard sidebar width
    sidebarCollapsed: '64px',   // Collapsed sidebar width
} as const;

/**
 * Section Padding
 * Vertical padding for page sections
 * Requirements: 18.3, 18.4
 */
export const sectionPadding = {
    desktop: '80px',  // 80px vertical padding on desktop
    mobile: '48px',   // 48px vertical padding on mobile
} as const;

// ============================================================================
// BREAKPOINTS
// ============================================================================

/**
 * Responsive Breakpoints
 * Mobile-first breakpoint scale
 */
export const breakpoints = {
    xs: '375px',   // iPhone SE
    sm: '640px',   // Small tablets
    md: '768px',   // iPad
    lg: '1024px',  // iPad Pro
    xl: '1280px',  // Laptops
    '2xl': '1536px', // Desktops
} as const;

// ============================================================================
// COMPONENT PATTERNS
// ============================================================================

/**
 * Button Style Patterns
 * Pre-defined class combinations for button variants
 */
export const buttonPatterns = {
    primary: `
    inline-flex items-center justify-center
    px-6 py-3 min-h-[48px]
    bg-gradient-to-br from-[#007AFF] to-[#0062CC]
    text-white font-semibold text-[17px]
    rounded-xl
    shadow-[0_4px_14px_0_rgba(0,122,255,0.15)]
    hover:shadow-[0_8px_25px_0_rgba(0,122,255,0.25)]
    transition-all duration-200
    hover:-translate-y-0.5 active:scale-[0.98]
    focus-visible:outline-none focus-visible:ring-[3px]
    focus-visible:ring-[#007AFF]/40
  `.trim().replace(/\s+/g, ' '),
    secondary: `
    inline-flex items-center justify-center
    px-6 py-3 min-h-[48px]
    bg-[#F2F2F7]
    text-[#000000] font-semibold text-[17px]
    rounded-xl border border-[#E5E5EA]
    transition-all duration-200
    hover:bg-[#E5E5EA] hover:-translate-y-0.5
    active:scale-[0.98]
    focus-visible:outline-none focus-visible:ring-[3px]
    focus-visible:ring-[#007AFF]/40
  `.trim().replace(/\s+/g, ' '),
    ghost: `
    inline-flex items-center justify-center
    px-4 py-2 min-h-[44px]
    text-[#007AFF] font-medium
    rounded-lg
    transition-colors duration-150
    hover:bg-[#007AFF]/10
    active:bg-[#007AFF]/20
    focus-visible:outline-none focus-visible:ring-[3px]
    focus-visible:ring-[#007AFF]/40
  `.trim().replace(/\s+/g, ' '),
} as const;

/**
 * Card Style Patterns
 * Pre-defined class combinations for card variants
 */
export const cardPatterns = {
    standard: `
    bg-white rounded-2xl p-6
    shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),0_2px_4px_-1px_rgba(0,0,0,0.06)]
    hover:shadow-[0_10px_15px_-3px_rgba(0,0,0,0.1),0_4px_6px_-2px_rgba(0,0,0,0.05)]
    border border-black/5
    transition-all duration-300
    hover:-translate-y-1
  `.trim().replace(/\s+/g, ' '),
    featured: `
    bg-gradient-to-br from-white to-[#E8F4FF]
    rounded-3xl p-8
    shadow-[0_8px_30px_0_rgba(0,122,255,0.12)]
    border border-[#C5E4FF]
  `.trim().replace(/\s+/g, ' '),
    glass: `
    bg-white/80 backdrop-blur-[16px]
    rounded-2xl p-6
    border border-white/20
    shadow-[0_8px_32px_0_rgba(31,38,135,0.15)]
  `.trim().replace(/\s+/g, ' '),
} as const;

/**
 * Form Input Style Patterns
 * Pre-defined class combinations for form inputs
 */
export const inputPatterns = {
    text: `
    w-full h-12 px-4
    bg-[#F2F2F7]
    text-[#000000] text-[17px]
    rounded-xl border border-transparent
    transition-all duration-200
    placeholder:text-[#C7C7CC]
    focus:bg-white focus:border-[#007AFF]
    focus:ring-4 focus:ring-[#007AFF]/20
    focus:outline-none
  `.trim().replace(/\s+/g, ' '),
    select: `
    w-full h-12 px-4 pr-10
    bg-[#F2F2F7]
    text-[#000000] text-[17px]
    rounded-xl border border-transparent
    appearance-none cursor-pointer
    transition-all duration-200
    focus:bg-white focus:border-[#007AFF]
    focus:ring-4 focus:ring-[#007AFF]/20
  `.trim().replace(/\s+/g, ' '),
} as const;

// ============================================================================
// RISK LEVEL UTILITIES
// ============================================================================

/**
 * Get NRI risk level from score
 */
export function getNriRiskLevel(score: number): keyof typeof nriColors {
    if (score <= 25) return 'minimal';
    if (score <= 40) return 'low';
    if (score <= 55) return 'moderate';
    if (score <= 70) return 'elevated';
    if (score <= 85) return 'high';
    return 'critical';
}

/**
 * Get NRI risk color from score
 */
export function getNriRiskColor(score: number): string {
    return nriColors[getNriRiskLevel(score)];
}

/**
 * Risk badge class patterns
 */
export const riskBadgePatterns = {
    minimal: 'bg-[#34C759]/10 text-[#34C759]',
    low: 'bg-[#30D158]/10 text-[#30D158]',
    moderate: 'bg-[#FFD60A]/10 text-[#FFD60A]',
    elevated: 'bg-[#FF9F0A]/10 text-[#FF9F0A]',
    high: 'bg-[#FF6B6B]/10 text-[#FF6B6B]',
    critical: 'bg-[#FF3B30]/10 text-[#FF3B30]',
} as const;

// ============================================================================
// EXPORTS
// ============================================================================

export const designTokens = {
    colors,
    typography,
    spacing,
    componentSpacing,
    animation,
    borderRadius,
    shadows,
    containerWidths,
    layoutWidths,
    sectionPadding,
    breakpoints,
    buttonPatterns,
    cardPatterns,
    inputPatterns,
    riskBadgePatterns,
    getNriRiskLevel,
    getNriRiskColor,
} as const;

export default designTokens;
