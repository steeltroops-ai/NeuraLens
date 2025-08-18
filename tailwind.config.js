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
  darkMode: 'class',
  theme: {
    extend: {
      // Color system from design tokens
      colors: {
        // Primary Clinical Blue
        primary: {
          50: '#EFF6FF',
          100: '#DBEAFE',
          200: '#BFDBFE',
          300: '#93C5FD',
          400: '#60A5FA',
          500: '#3B82F6',
          600: '#2563EB',
          700: '#1D4ED8',
          800: '#1E40AF',
          900: '#1E3A8A',
          950: '#172554',
        },
        
        // Clinical Neutrals
        neutral: {
          50: '#FAFAFA',
          100: '#F5F5F5',
          200: '#E5E5E5',
          300: '#D4D4D4',
          400: '#A3A3A3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
          950: '#0A0A0A',
        },

        // Semantic Colors
        success: '#10B981',
        warning: '#F59E0B',
        error: '#EF4444',
        info: '#3B82F6',

        // Risk Assessment Colors
        risk: {
          low: '#10B981',
          moderate: '#F59E0B',
          high: '#F97316',
          critical: '#EF4444',
        },

        // Surface Colors
        surface: {
          background: '#0A0A0A',
          primary: '#1A1A1A',
          secondary: '#2A2A2A',
          tertiary: '#3A3A3A',
        },

        // Text Colors
        text: {
          primary: '#FFFFFF',
          secondary: '#A0A0A0',
          muted: '#6B7280',
          inverse: '#0F172A',
        },
      },

      // Typography system
      fontFamily: {
        sans: ['Inter', ...fontFamily.sans],
        display: ['Inter', ...fontFamily.sans],
        mono: ['JetBrains Mono', ...fontFamily.mono],
      },

      fontSize: {
        xs: ['0.75rem', { lineHeight: '1rem' }],
        sm: ['0.875rem', { lineHeight: '1.25rem' }],
        base: ['1rem', { lineHeight: '1.5rem' }],
        lg: ['1.125rem', { lineHeight: '1.75rem' }],
        xl: ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }],
        '7xl': ['4.5rem', { lineHeight: '1' }],
        '8xl': ['6rem', { lineHeight: '1' }],
        '9xl': ['8rem', { lineHeight: '1' }],
      },

      // Spacing system (8px base unit)
      spacing: {
        '0.5': '0.125rem', // 2px
        '1.5': '0.375rem', // 6px
        '2.5': '0.625rem', // 10px
        '3.5': '0.875rem', // 14px
        '18': '4.5rem',    // 72px
        '88': '22rem',     // 352px
        '100': '25rem',    // 400px
        '112': '28rem',    // 448px
        '128': '32rem',    // 512px
      },

      // Animation system
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s infinite',
        'nri-reveal': 'nriReveal 1.2s ease-out',
        'progress-fill': 'progressFill 0.5s ease-out',
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

      // Transition timing functions
      transitionTimingFunction: {
        'out-quint': 'cubic-bezier(0.22, 1, 0.36, 1)',
        'in-out-cubic': 'cubic-bezier(0.65, 0, 0.35, 1)',
        'spring': 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
      },

      // Box shadows
      boxShadow: {
        'clinical': '0 4px 14px 0 rgba(59, 130, 246, 0.25)',
        'clinical-hover': '0 6px 20px 0 rgba(59, 130, 246, 0.35)',
        'glass': '0 8px 32px rgba(31, 38, 135, 0.37)',
        'nri': '0 10px 40px rgba(59, 130, 246, 0.3)',
      },

      // Border radius
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.5rem',
      },

      // Backdrop blur
      backdropBlur: {
        xs: '2px',
      },

      // Z-index scale
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },

      // Aspect ratios
      aspectRatio: {
        '4/3': '4 / 3',
        '3/2': '3 / 2',
        '2/3': '2 / 3',
        '9/16': '9 / 16',
      },

      // Grid template columns
      gridTemplateColumns: {
        '13': 'repeat(13, minmax(0, 1fr))',
        '14': 'repeat(14, minmax(0, 1fr))',
        '15': 'repeat(15, minmax(0, 1fr))',
        '16': 'repeat(16, minmax(0, 1fr))',
      },

      // Screens (breakpoints)
      screens: {
        'xs': '475px',
        '3xl': '1600px',
      },

      // Container
      container: {
        center: true,
        padding: {
          DEFAULT: '1rem',
          sm: '2rem',
          lg: '4rem',
          xl: '5rem',
          '2xl': '6rem',
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms')({
      strategy: 'class',
    }),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
    
    // Custom plugin for clinical utilities
    function({ addUtilities, addComponents, theme }) {
      // Clinical focus utilities
      addUtilities({
        '.clinical-focus': {
          '&:focus-visible': {
            outline: '3px solid theme(colors.primary.500)',
            outlineOffset: '2px',
            boxShadow: '0 0 0 6px rgba(59, 130, 246, 0.2)',
          },
        },
        '.glass-morphism': {
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
        '.text-gradient': {
          background: 'linear-gradient(135deg, theme(colors.primary.500), theme(colors.primary.600))',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        },
      });

      // Clinical components
      addComponents({
        '.btn': {
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: theme('fontWeight.medium'),
          borderRadius: theme('borderRadius.lg'),
          border: 'none',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          textDecoration: 'none',
          whiteSpace: 'nowrap',
          userSelect: 'none',
          position: 'relative',
          overflow: 'hidden',
          '&:focus-visible': {
            outline: '2px solid theme(colors.primary.500)',
            outlineOffset: '2px',
          },
          '&:disabled': {
            opacity: '0.5',
            cursor: 'not-allowed',
            pointerEvents: 'none',
          },
        },
        '.card': {
          backgroundColor: theme('colors.surface.primary'),
          border: `1px solid ${theme('colors.neutral.800')}`,
          borderRadius: theme('borderRadius.2xl'),
          padding: theme('spacing.6'),
          boxShadow: theme('boxShadow.lg'),
          transition: 'all 0.3s ease',
        },
        '.nri-score': {
          fontSize: theme('fontSize.6xl[0]'),
          fontWeight: theme('fontWeight.black'),
          fontFamily: theme('fontFamily.display'),
          lineHeight: theme('lineHeight.none'),
          background: 'linear-gradient(135deg, theme(colors.primary.500), theme(colors.primary.600))',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textAlign: 'center',
        },
      });
    },
  ],
};
