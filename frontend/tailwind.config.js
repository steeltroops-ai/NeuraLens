/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Premium OKLCH Color System - Neural Technology Palette
      colors: {
        // Primary Colors - Neural Deep Blue & Electric Innovation
        primary: {
          50: "oklch(97.5% 0.013 250)",
          100: "oklch(94.2% 0.028 250)",
          200: "oklch(87.8% 0.056 250)",
          300: "oklch(79.1% 0.089 250)",
          400: "oklch(68.9% 0.123 250)",
          500: "oklch(58.2% 0.142 250)", // Electric Blue - Innovation Core
          600: "oklch(49.8% 0.138 250)",
          700: "oklch(42.1% 0.128 250)",
          800: "oklch(35.2% 0.115 250)",
          900: "oklch(28.8% 0.098 250)", // Deep Neural Blue - Trust Foundation
          950: "oklch(19.2% 0.065 250)",
        },
        secondary: {
          50: "oklch(97.8% 0.015 180)",
          100: "oklch(93.2% 0.042 180)",
          200: "oklch(86.1% 0.078 180)",
          300: "oklch(77.8% 0.108 180)",
          400: "oklch(68.2% 0.128 180)",
          500: "oklch(59.1% 0.142 180)", // Teal Science - Precision
          600: "oklch(51.2% 0.138 180)",
          700: "oklch(44.1% 0.128 180)",
          800: "oklch(37.8% 0.115 180)",
          900: "oklch(32.1% 0.098 180)",
          950: "oklch(21.8% 0.065 180)",
        },
        neutral: {
          50: "oklch(98.2% 0.002 270)",
          100: "oklch(96.1% 0.004 270)",
          200: "oklch(91.8% 0.008 270)",
          300: "oklch(85.2% 0.012 270)", // Soft Gray - Neutrality
          400: "oklch(71.8% 0.018 270)",
          500: "oklch(58.2% 0.024 270)",
          600: "oklch(45.8% 0.028 270)",
          700: "oklch(38.2% 0.032 270)",
          800: "oklch(28.8% 0.035 270)",
          900: "oklch(18.2% 0.038 270)",
          950: "oklch(8.8% 0.042 270)",
        },
        // Premium Glassmorphism Support
        glass: {
          light: "oklch(100% 0 0 / 0.08)",
          medium: "oklch(100% 0 0 / 0.12)",
          strong: "oklch(100% 0 0 / 0.18)",
          "border-light": "oklch(100% 0 0 / 0.15)",
          "border-medium": "oklch(100% 0 0 / 0.25)",
          "border-strong": "oklch(100% 0 0 / 0.35)",
        },
        // Neural Grid System
        neural: {
          "grid-primary": "oklch(28.8% 0.098 250 / 0.08)",
          "grid-secondary": "oklch(59.1% 0.142 180 / 0.06)",
          "node-active": "oklch(58.2% 0.142 250 / 0.8)",
          "node-inactive": "oklch(59.1% 0.142 180 / 0.4)",
          "connection-primary": "oklch(58.2% 0.142 250 / 0.3)",
          "connection-secondary": "oklch(59.1% 0.142 180 / 0.2)",
          "pulse-glow": "oklch(58.2% 0.142 250 / 0.6)",
        },
        // Success, Warning, Error States
        success: {
          50: "oklch(97.1% 0.013 142)",
          500: "oklch(64.8% 0.150 142)",
          600: "oklch(55.4% 0.148 142)",
        },
        warning: {
          50: "oklch(97.6% 0.013 83)",
          500: "oklch(75.8% 0.108 83)",
          600: "oklch(65.1% 0.118 83)",
        },
        error: {
          50: "oklch(97.2% 0.013 25)",
          500: "oklch(62.8% 0.257 25)",
          600: "oklch(54.3% 0.227 25)",
        },
      },

      // Typography - Inter Primary, Roboto Fallback
      fontFamily: {
        sans: ["Inter", "Roboto", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },

      // 8px Grid System
      spacing: {
        0.5: "2px",
        1: "4px",
        2: "8px",
        3: "12px",
        4: "16px",
        5: "20px",
        6: "24px",
        8: "32px",
        10: "40px",
        12: "48px",
        16: "64px",
        20: "80px",
        24: "96px",
        32: "128px",
      },

      // Glassmorphism & Elevation
      backdropBlur: {
        xs: "2px",
        sm: "4px",
        md: "8px",
        lg: "12px",
        xl: "16px",
        "2xl": "24px",
        glass: "10px", // Standard glassmorphism
      },

      boxShadow: {
        // Premium Glass Shadows
        "glass-sm": "0 2px 8px 0 oklch(28.8% 0.098 250 / 0.15)",
        "glass-md": "0 8px 32px 0 oklch(28.8% 0.098 250 / 0.2)",
        "glass-lg": "0 16px 64px 0 oklch(28.8% 0.098 250 / 0.25)",
        "glass-xl": "0 24px 96px 0 oklch(28.8% 0.098 250 / 0.3)",

        // Neural Glow Shadows
        "neural-sm": "0 0 8px 0 oklch(58.2% 0.142 250 / 0.3)",
        "neural-md": "0 0 16px 0 oklch(58.2% 0.142 250 / 0.4)",
        "neural-lg": "0 0 32px 0 oklch(58.2% 0.142 250 / 0.5)",
        "neural-xl": "0 0 64px 0 oklch(58.2% 0.142 250 / 0.6)",

        // Premium Elevation System
        "elevation-1":
          "0 1px 3px 0 oklch(0% 0 0 / 0.1), 0 1px 2px -1px oklch(0% 0 0 / 0.1)",
        "elevation-2":
          "0 4px 6px -1px oklch(0% 0 0 / 0.1), 0 2px 4px -2px oklch(0% 0 0 / 0.1)",
        "elevation-3":
          "0 10px 15px -3px oklch(0% 0 0 / 0.1), 0 4px 6px -4px oklch(0% 0 0 / 0.1)",
        "elevation-4":
          "0 20px 25px -5px oklch(0% 0 0 / 0.1), 0 8px 10px -6px oklch(0% 0 0 / 0.1)",
        "elevation-5": "0 25px 50px -12px oklch(0% 0 0 / 0.25)",

        // Legacy support
        glass: "0 8px 32px 0 oklch(28.8% 0.098 250 / 0.2)",
        neural: "0 4px 16px 0 oklch(28.8% 0.098 250 / 0.2)",
      },

      // Border Radius - 8px System
      borderRadius: {
        none: "0",
        sm: "4px",
        md: "8px", // Standard card radius
        lg: "12px",
        xl: "16px",
        "2xl": "24px",
        full: "9999px",
      },

      // Animation & Transitions
      animation: {
        "fade-in": "fadeIn 0.5s ease-in-out",
        "slide-up": "slideUp 0.3s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "neural-pulse": "neuralPulse 2s ease-in-out infinite",
      },

      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(10px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        neuralPulse: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "0.8" },
        },
      },

      // Responsive Breakpoints
      screens: {
        xs: "320px",
        sm: "640px",
        md: "768px",
        lg: "1024px",
        xl: "1280px",
        "2xl": "1536px",
      },

      // Accessibility & Touch Targets
      minHeight: {
        touch: "44px", // Minimum touch target
      },
      minWidth: {
        touch: "44px", // Minimum touch target
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    require("@tailwindcss/typography"),
    require("@tailwindcss/aspect-ratio"),
    // Premium Design System Plugin
    function ({ addUtilities }) {
      const premiumUtilities = {
        // Premium Glassmorphism
        ".glass-light": {
          background: "oklch(100% 0 0 / 0.08)",
          backdropFilter: "blur(8px)",
          WebkitBackdropFilter: "blur(8px)",
          border: "1px solid oklch(100% 0 0 / 0.15)",
          boxShadow: "0 2px 8px 0 oklch(28.8% 0.098 250 / 0.15)",
          transition: "all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        },
        ".glass-medium": {
          background: "oklch(100% 0 0 / 0.12)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          border: "1px solid oklch(100% 0 0 / 0.25)",
          boxShadow: "0 8px 32px 0 oklch(28.8% 0.098 250 / 0.2)",
          transition: "all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        },
        ".glass-strong": {
          background: "oklch(100% 0 0 / 0.18)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid oklch(100% 0 0 / 0.35)",
          boxShadow: "0 16px 64px 0 oklch(28.8% 0.098 250 / 0.25)",
          transition: "all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        },

        // Neural Grid System
        ".neural-grid-primary": {
          backgroundImage:
            "radial-gradient(circle at 1px 1px, oklch(28.8% 0.098 250 / 0.08) 1px, transparent 0)",
          backgroundSize: "24px 24px",
          backgroundAttachment: "fixed",
        },
        ".neural-grid-secondary": {
          backgroundImage:
            "radial-gradient(circle at 1px 1px, oklch(59.1% 0.142 180 / 0.06) 1px, transparent 0)",
          backgroundSize: "32px 32px",
          backgroundAttachment: "fixed",
        },
        ".neural-grid-animated": {
          backgroundImage:
            "radial-gradient(circle at 1px 1px, oklch(28.8% 0.098 250 / 0.08) 1px, transparent 0), radial-gradient(circle at 1px 1px, oklch(59.1% 0.142 180 / 0.06) 1px, transparent 0)",
          backgroundSize: "24px 24px, 48px 48px",
          backgroundAttachment: "fixed",
          animation: "neural-grid-pulse 8s ease-in-out infinite",
        },

        // Premium Interactive Effects
        ".interactive-premium": {
          transition: "all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          transformOrigin: "center",
          willChange: "transform, box-shadow, background",
        },
        ".interactive-premium:hover": {
          transform: "translateY(-2px) scale(1.02)",
          boxShadow: "0 16px 64px 0 oklch(28.8% 0.098 250 / 0.25)",
        },
        ".interactive-premium:active": {
          transform: "translateY(0) scale(0.98)",
          transitionDuration: "100ms",
        },

        // Text Gradients
        ".text-gradient-primary": {
          background:
            "linear-gradient(135deg, oklch(49.8% 0.138 250) 0%, oklch(58.2% 0.142 250) 50%, oklch(59.1% 0.142 180) 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
          backgroundSize: "200% 200%",
          animation: "gradient-shift 4s ease-in-out infinite",
        },
        ".text-gradient-neural": {
          background:
            "linear-gradient(135deg, oklch(49.8% 0.138 250) 0%, oklch(59.1% 0.142 180) 25%, oklch(58.2% 0.142 250) 50%, oklch(51.2% 0.138 180) 75%, oklch(42.1% 0.128 250) 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
          backgroundSize: "300% 300%",
          animation: "neural-gradient 6s ease-in-out infinite",
        },

        // Legacy Support
        ".glass": {
          background: "oklch(100% 0 0 / 0.12)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          border: "1px solid oklch(100% 0 0 / 0.25)",
          boxShadow: "0 8px 32px 0 oklch(28.8% 0.098 250 / 0.2)",
          transition: "all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        },
        ".glass-dark": {
          background: "rgba(0, 0, 0, 0.1)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          boxShadow: "0 2px 8px 0 oklch(28.8% 0.098 250 / 0.15)",
          transition: "all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        },
        ".neural-grid": {
          backgroundImage:
            "radial-gradient(circle at 1px 1px, oklch(28.8% 0.098 250 / 0.08) 1px, transparent 0)",
          backgroundSize: "24px 24px",
          backgroundAttachment: "fixed",
        },
      };
      addUtilities(premiumUtilities);
    },
  ],
};
