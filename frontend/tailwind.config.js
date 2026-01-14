/** @type {import('tailwindcss').Config} */

const { fontFamily } = require('tailwindcss/defaultTheme');

module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/lib/**/*.{js,ts,jsx,tsx,mdx}',
    './src/utils/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'media', // Use media query for dark mode detection
  theme: {
    extend: {
      // MediLens Design System Colors
      colors: {
        // MediLens Blue - Primary Brand Color (#007AFF)
        'medilens-blue': {
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
        },

        // Primary (alias for medilens-blue)
        primary: {
          50: '#E8F4FF',
          100: '#C5E4FF',
          200: '#9DD1FF',
          300: '#6BBAFF',
          400: '#3AA3FF',
          500: '#007AFF',
          600: '#0062CC',
          700: '#004A99',
          800: '#003366',
          900: '#001A33',
          950: '#000D1A',
        },

        // Medical Blue (alias for medilens-blue)
        medical: {
          50: '#E8F4FF',
          100: '#C5E4FF',
          200: '#9DD1FF',
          300: '#6BBAFF',
          400: '#3AA3FF',
          500: '#007AFF',
          600: '#0062CC',
          700: '#004A99',
          800: '#003366',
          900: '#001A33',
          950: '#000D1A',
        },

        // NRI Risk Gradient Colors
        nri: {
          minimal: '#34C759',   // 0-25: Minimal risk
          low: '#30D158',       // 26-40: Low risk
          moderate: '#FFD60A',  // 41-55: Moderate risk
          elevated: '#FF9F0A',  // 56-70: Elevated risk
          high: '#FF6B6B',      // 71-85: High risk
          critical: '#FF3B30',  // 86-100: Critical risk
        },

        // Semantic Status Colors
        status: {
          healthy: '#34C759',    // Low risk, positive results
          caution: '#FF9500',    // Moderate risk, attention needed
          alert: '#FF3B30',      // High risk, immediate attention
          info: '#5AC8FA',       // Informational, neutral
          processing: '#AF52DE', // AI processing, analysis
        },

        // Success Green (Apple Green)
        success: {
          50: '#ECFDF5',
          100: '#D1FAE5',
          200: '#A7F3D0',
          300: '#6EE7B7',
          400: '#34D399',
          500: '#34C759', // Apple green
          600: '#059669',
          700: '#047857',
          800: '#065F46',
          900: '#064E3B',
        },

        // Warning Orange (Apple Orange)
        warning: {
          50: '#FFFBEB',
          100: '#FEF3C7',
          200: '#FDE68A',
          300: '#FCD34D',
          400: '#FBBF24',
          500: '#FF9500', // Apple orange
          600: '#D97706',
          700: '#B45309',
          800: '#92400E',
          900: '#78350F',
        },

        // Error Red (Apple Red)
        error: {
          50: '#FEF2F2',
          100: '#FEE2E2',
          200: '#FECACA',
          300: '#FCA5A5',
          400: '#F87171',
          500: '#FF3B30', // Apple red
          600: '#DC2626',
          700: '#B91C1C',
          800: '#991B1B',
          900: '#7F1D1D',
        },

        // Neural Purple (AI theme)
        neural: {
          50: '#FAF5FF',
          100: '#F3E8FF',
          200: '#E9D5FF',
          300: '#D8B4FE',
          400: '#C084FC',
          500: '#AF52DE', // Apple purple
          600: '#9333EA',
          700: '#7C3AED',
          800: '#6B21A8',
          900: '#581C87',
        },

        // Surface Colors (MediLens Design System)
        surface: {
          primary: '#FFFFFF',    // Cards, modals
          secondary: '#F2F2F7',  // Page backgrounds
          tertiary: '#E5E5EA',   // Dividers, borders
          elevated: '#FFFFFF',   // Elevated cards
        },

        // Apple-style neutrals
        gray: {
          50: '#F2F2F7', // Apple light gray
          100: '#E5E5EA',
          200: '#D1D1D6',
          300: '#C7C7CC',
          400: '#AEAEB2',
          500: '#8E8E93',
          600: '#636366',
          700: '#48484A',
          800: '#3A3A3C',
          900: '#2C2C2E',
          950: '#1C1C1E', // Apple dark
        },

        // Background colors
        background: '#F2F2F7', // Apple light background
        card: '#FFFFFF', // Pure white cards

        // Text colors (MediLens Design System)
        text: {
          primary: '#000000',    // Headlines, primary content
          secondary: '#3C3C43',  // Body text, descriptions
          tertiary: '#8E8E93',   // Captions, metadata
          quaternary: '#C7C7CC', // Placeholders, disabled
        },
      },

      // Apple-style typography
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Display',
          'SF Pro Text',
          'Inter',
          'system-ui',
          'sans-serif',
        ],
        display: [
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Display',
          'Inter',
          'system-ui',
          'sans-serif',
        ],
        mono: ['SF Mono', 'Monaco', 'Menlo', 'JetBrains Mono', 'monospace'],
      },

      fontSize: {
        xs: ['12px', { lineHeight: '16px' }], // Captions
        sm: ['14px', { lineHeight: '20px' }], // Secondary text
        base: ['17px', { lineHeight: '24px' }], // Body text (Apple standard)
        lg: ['20px', { lineHeight: '28px' }], // Subheadings
        xl: ['24px', { lineHeight: '32px' }], // Section headers
        '2xl': ['28px', { lineHeight: '36px' }], // Page titles
        '3xl': ['34px', { lineHeight: '40px' }], // Hero headlines
        '4xl': ['40px', { lineHeight: '44px' }], // Large headlines
        '5xl': ['48px', { lineHeight: '52px' }], // Display text
        '6xl': ['60px', { lineHeight: '64px' }], // Hero display
      },

      // 8px grid spacing system
      spacing: {
        0.5: '2px', // Micro spacing
        1.5: '6px', // Small spacing
        2.5: '10px', // Medium spacing
        3.5: '14px', // Standard spacing
        4.5: '18px', // Large spacing
        5.5: '22px', // XL spacing
        6.5: '26px', // XXL spacing
        7.5: '30px', // Section spacing
        15: '60px', // Major spacing
        18: '72px', // Hero spacing
        22: '88px', // Massive spacing
        // Breathe principle spacing (Requirements 18.5)
        'breathe-sm': '24px',  // Small breathe gap
        'breathe-md': '32px',  // Medium breathe gap
        'breathe-lg': '48px',  // Large breathe gap
        'breathe-xl': '64px',  // Extra large breathe gap
        'breathe-2xl': '80px', // Section breathe gap
        'section-mobile': '48px', // Section padding mobile
        'section-desktop': '80px', // Section padding desktop
      },

      // Apple-style animations
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.22, 1, 0.36, 1)',
        'scale-in': 'scaleIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
        'bounce-gentle': 'bounceGentle 0.6s ease-out',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        shimmer: 'shimmer 1.5s infinite',
        'button-press': 'buttonPress 0.15s ease-out',
        'card-hover': 'cardHover 0.3s ease-out',
        'nri-reveal': 'nriReveal 1.2s cubic-bezier(0.22, 1, 0.36, 1)',
        'progress-fill': 'progressFill 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
        'spin-slow': 'spin 3s linear infinite',
      },

      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.9)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        bounceGentle: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.05)' },
          '100%': { transform: 'scale(1)' },
        },
        buttonPress: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(0.98)' },
          '100%': { transform: 'scale(1)' },
        },
        cardHover: {
          '0%': { transform: 'translateY(0) scale(1)' },
          '100%': { transform: 'translateY(-4px) scale(1.02)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        nriReveal: {
          '0%': { opacity: '0', transform: 'scale(0.8) translateY(20px)' },
          '50%': { opacity: '0.7', transform: 'scale(1.05) translateY(-5px)' },
          '100%': { opacity: '1', transform: 'scale(1) translateY(0)' },
        },
        progressFill: {
          '0%': { width: '0%' },
          '100%': { width: 'var(--progress-width, 0%)' },
        },
      },

      // MediLens Animation Easing Functions
      transitionTimingFunction: {
        apple: 'cubic-bezier(0.25, 0.1, 0.25, 1)',
        'out-quint': 'cubic-bezier(0.22, 1, 0.36, 1)',      // Primary MediLens easing
        'in-out-cubic': 'cubic-bezier(0.65, 0, 0.35, 1)',   // Modals
        spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',  // Playful
        bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      },

      // MediLens Design System Shadows
      boxShadow: {
        // Apple-style shadows
        apple: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'apple-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'apple-pressed': '0 1px 4px rgba(0, 0, 0, 0.16)',
        // MediLens medical shadows
        medical: '0 4px 14px 0 rgba(0, 122, 255, 0.15)',
        'medical-hover': '0 8px 25px 0 rgba(0, 122, 255, 0.25)',
        // NRI and glass shadows
        nri: '0 8px 30px 0 rgba(0, 122, 255, 0.12)',
        glass: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
      },

      // Apple-style border radius
      borderRadius: {
        apple: '12px', // Standard Apple radius
        'apple-lg': '16px', // Large Apple radius
        'apple-xl': '20px', // Extra large Apple radius
        '4xl': '2rem',
        '5xl': '2.5rem',
      },

      // Backdrop blur
      backdropBlur: {
        xs: '2px',
        apple: '16px', // Apple glass effect
      },

      // Z-index scale
      zIndex: {
        60: '60',
        70: '70',
        80: '80',
        90: '90',
        100: '100',
      },

      // Touch target sizes (Apple HIG)
      minHeight: {
        touch: '44px', // Minimum touch target
        'touch-lg': '48px', // Large touch target
      },

      minWidth: {
        touch: '44px', // Minimum touch target
        'touch-lg': '48px', // Large touch target
      },

      // Maximum width for readability (MediLens Typography System)
      maxWidth: {
        prose: '65ch', // Optimal reading width for body text
        'prose-sm': '55ch', // Narrower reading width
        'prose-lg': '75ch', // Wider reading width
      },

      // Apple-style screens (breakpoints)
      screens: {
        xs: '375px', // iPhone SE
        sm: '640px', // Small tablets
        md: '768px', // iPad
        lg: '1024px', // iPad Pro / Small laptops
        xl: '1280px', // Laptops
        '2xl': '1536px', // Large screens
        '3xl': '1920px', // Ultra-wide
      },

      // MediLens Container Widths (Requirements 18.1)
      // sm (640px), md (768px), lg (1024px), xl (1280px), 2xl (1440px)
      container: {
        center: true,
        padding: {
          DEFAULT: '16px', // Apple standard
          sm: '20px',
          md: '24px',
          lg: '32px',
          xl: '40px',
          '2xl': '48px',
        },
        screens: {
          sm: '640px',   // Narrow content
          md: '768px',   // Standard content
          lg: '1024px',  // Wide content
          xl: '1280px',  // Full-width content
          '2xl': '1440px', // Maximum width
        },
      },

      // MediLens Layout Widths
      width: {
        'sidebar': '280px',        // Dashboard sidebar width (Requirements 18.2)
        'sidebar-collapsed': '64px', // Collapsed sidebar width
        'container-sm': '640px',   // Narrow content container
        'container-md': '768px',   // Standard content container
        'container-lg': '1024px',  // Wide content container
        'container-xl': '1280px',  // Full-width content container
        'container-2xl': '1440px', // Maximum width container
      },

      // MediLens Max Widths
      maxWidth: {
        'container-sm': '640px',   // Narrow content
        'container-md': '768px',   // Standard content
        'container-lg': '1024px',  // Wide content
        'container-xl': '1280px',  // Full-width content
        'container-2xl': '1440px', // Maximum width
        prose: '65ch', // Optimal reading width for body text
        'prose-sm': '55ch', // Narrower reading width
        'prose-lg': '75ch', // Wider reading width
      },

      // MediLens Grid Templates (Requirements 18.2)
      gridTemplateColumns: {
        'dashboard': '280px 1fr',           // Dashboard layout with sidebar
        'dashboard-collapsed': '64px 1fr',  // Dashboard layout with collapsed sidebar
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms')({
      strategy: 'class',
    }),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),

    // Apple-style component utilities
    function ({ addUtilities, addComponents, theme }) {
      // Apple-style focus utilities
      addUtilities({
        '.apple-focus': {
          '&:focus-visible': {
            outline: '3px solid #007AFF',
            outlineOffset: '2px',
            boxShadow: '0 0 0 6px rgba(0, 122, 255, 0.2)',
          },
        },
        '.glass-morphism': {
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)',
        },
        '.text-gradient-medical': {
          background: 'linear-gradient(135deg, #007AFF, #0056CC)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        },
        '.text-gradient-success': {
          background: 'linear-gradient(135deg, #34C759, #059669)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        },
      });

      // Apple-style components
      addComponents({
        '.btn-apple': {
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: '600',
          borderRadius: '12px',
          border: 'none',
          cursor: 'pointer',
          transition: 'all 0.15s ease-out',
          textDecoration: 'none',
          whiteSpace: 'nowrap',
          userSelect: 'none',
          position: 'relative',
          minHeight: '44px',
          minWidth: '44px',
          fontSize: '17px',
          '&:focus-visible': {
            outline: '3px solid #007AFF',
            outlineOffset: '2px',
          },
          '&:disabled': {
            opacity: '0.5',
            cursor: 'not-allowed',
            pointerEvents: 'none',
          },
          '&:active': {
            transform: 'scale(0.98)',
          },
        },
        '.btn-primary': {
          background: 'linear-gradient(135deg, #007AFF 0%, #0056CC 100%)',
          color: 'white',
          boxShadow: '0 4px 12px rgba(0, 122, 255, 0.3)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 20px rgba(0, 122, 255, 0.4)',
          },
        },
        '.btn-secondary': {
          backgroundColor: '#F2F2F7',
          color: '#000000',
          border: '1px solid #C6C6C8',
          '&:hover': {
            backgroundColor: '#E5E5EA',
            transform: 'translateY(-1px)',
          },
        },
        '.card-apple': {
          backgroundColor: 'white',
          borderRadius: '20px',
          padding: '24px',
          boxShadow: '0 2px 16px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(0, 0, 0, 0.05)',
          transition: 'all 0.3s cubic-bezier(0.22, 1, 0.36, 1)',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.12)',
          },
        },
        '.nri-score': {
          fontSize: '60px',
          fontWeight: '900',
          fontFamily: theme('fontFamily.display'),
          lineHeight: '1',
          background: 'linear-gradient(135deg, #007AFF, #0056CC)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textAlign: 'center',
        },
      });
    },
  ],
};
