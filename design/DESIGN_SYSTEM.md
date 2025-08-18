# NeuroLens-X Design System

## 🎨 **CLINICAL-GRADE DESIGN FOUNDATION**

### **Design Philosophy**
- **Clinical Excellence**: Professional healthcare application aesthetics
- **Accessibility First**: WCAG 2.1 AA+ compliance for all users
- **Trust & Reliability**: Design that inspires confidence in medical professionals
- **Modern Sophistication**: Cutting-edge design trends with clinical appropriateness

---

## 🎯 **COLOR SYSTEM**

### **Primary Color Palette**
```css
/* Clinical Blue - Primary Brand */
--primary-50: #EFF6FF;
--primary-100: #DBEAFE;
--primary-200: #BFDBFE;
--primary-300: #93C5FD;
--primary-400: #60A5FA;
--primary-500: #3B82F6; /* Main brand color */
--primary-600: #2563EB;
--primary-700: #1D4ED8;
--primary-800: #1E40AF;
--primary-900: #1E3A8A;

/* Clinical Neutrals - Professional Grays */
--neutral-50: #FAFAFA;
--neutral-100: #F5F5F5;
--neutral-200: #E5E5E5;
--neutral-300: #D4D4D4;
--neutral-400: #A3A3A3;
--neutral-500: #737373;
--neutral-600: #525252;
--neutral-700: #404040;
--neutral-800: #262626;
--neutral-900: #171717;
--neutral-950: #0A0A0A;
```

### **Semantic Colors**
```css
/* Clinical Status Colors */
--success: #10B981; /* Healthy/Normal results */
--warning: #F59E0B; /* Moderate risk/Attention needed */
--error: #EF4444;   /* High risk/Critical */
--info: #3B82F6;    /* Information/Guidance */

/* Risk Assessment Colors */
--risk-low: #10B981;      /* 0-25 NRI Score */
--risk-moderate: #F59E0B; /* 26-50 NRI Score */
--risk-high: #F97316;     /* 51-75 NRI Score */
--risk-critical: #EF4444; /* 76-100 NRI Score */
```

### **Surface Colors (Dark Theme)**
```css
/* Professional Dark Interface */
--surface-background: #0A0A0A;
--surface-primary: #1A1A1A;
--surface-secondary: #2A2A2A;
--surface-tertiary: #3A3A3A;
--surface-overlay: rgba(0, 0, 0, 0.8);
--surface-glass: rgba(255, 255, 255, 0.05);

/* Text Colors */
--text-primary: #FFFFFF;
--text-secondary: #A0A0A0;
--text-muted: #6B7280;
--text-inverse: #0F172A;
```

---

## 📝 **TYPOGRAPHY SYSTEM**

### **Font Family**
```css
/* Primary Font Stack */
--font-display: 'Inter', 'SF Pro Display', system-ui, sans-serif;
--font-body: 'Inter', 'SF Pro Text', system-ui, sans-serif;
--font-mono: 'JetBrains Mono', 'Menlo', 'Monaco', monospace;
```

### **Type Scale (Mathematical 1.25 Ratio)**
```css
/* Display Typography */
--text-xs: 0.75rem;    /* 12px - Micro text */
--text-sm: 0.875rem;   /* 14px - Small text */
--text-base: 1rem;     /* 16px - Body text */
--text-lg: 1.125rem;   /* 18px - Large text */
--text-xl: 1.25rem;    /* 20px - Subheadings */
--text-2xl: 1.5rem;    /* 24px - Section headers */
--text-3xl: 1.875rem;  /* 30px - Page headers */
--text-4xl: 2.25rem;   /* 36px - Display headers */
--text-5xl: 3rem;      /* 48px - Hero text */

/* Line Heights */
--leading-tight: 1.25;
--leading-snug: 1.375;
--leading-normal: 1.5;
--leading-relaxed: 1.625;
--leading-loose: 2;
```

### **Font Weights**
```css
--font-light: 300;
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
```

---

## 📐 **SPACING SYSTEM**

### **8px Base Unit System**
```css
/* Spacing Scale */
--space-0: 0px;
--space-1: 0.25rem;  /* 4px */
--space-2: 0.5rem;   /* 8px */
--space-3: 0.75rem;  /* 12px */
--space-4: 1rem;     /* 16px */
--space-5: 1.25rem;  /* 20px */
--space-6: 1.5rem;   /* 24px */
--space-8: 2rem;     /* 32px */
--space-10: 2.5rem;  /* 40px */
--space-12: 3rem;    /* 48px */
--space-16: 4rem;    /* 64px */
--space-20: 5rem;    /* 80px */
--space-24: 6rem;    /* 96px */
```

### **Component Spacing**
```css
/* Standard Component Spacing */
--component-padding-sm: var(--space-3);  /* 12px */
--component-padding-md: var(--space-4);  /* 16px */
--component-padding-lg: var(--space-6);  /* 24px */
--component-padding-xl: var(--space-8);  /* 32px */

/* Layout Spacing */
--layout-gap-sm: var(--space-4);   /* 16px */
--layout-gap-md: var(--space-6);   /* 24px */
--layout-gap-lg: var(--space-8);   /* 32px */
--layout-gap-xl: var(--space-12);  /* 48px */
```

---

## 🎭 **ANIMATION SYSTEM**

### **Duration Scale**
```css
/* Animation Durations */
--duration-instant: 0ms;
--duration-fast: 150ms;
--duration-normal: 300ms;
--duration-slow: 500ms;
--duration-slower: 750ms;
```

### **Easing Functions**
```css
/* Premium Easing Curves */
--ease-out-quint: cubic-bezier(0.22, 1, 0.36, 1);
--ease-in-out-cubic: cubic-bezier(0.65, 0, 0.35, 1);
--ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
```

### **Motion Patterns**
```css
/* Common Animations */
.fade-in {
  animation: fadeIn var(--duration-normal) var(--ease-out-quint);
}

.slide-up {
  animation: slideUp var(--duration-normal) var(--ease-out-quint);
}

.scale-in {
  animation: scaleIn var(--duration-fast) var(--ease-spring);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.9); }
  to { opacity: 1; transform: scale(1); }
}
```

---

## 🧩 **COMPONENT SPECIFICATIONS**

### **Button System**
```css
/* Button Base */
.btn-base {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: var(--font-medium);
  border-radius: 0.5rem;
  transition: all var(--duration-fast) var(--ease-out-quint);
  focus: outline-none ring-2 ring-primary-500 ring-offset-2;
}

/* Button Variants */
.btn-primary {
  background: var(--primary-500);
  color: white;
  box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.25);
}

.btn-primary:hover {
  background: var(--primary-600);
  box-shadow: 0 6px 20px 0 rgba(59, 130, 246, 0.35);
  transform: translateY(-1px);
}

/* Button Sizes */
.btn-sm { padding: 0.5rem 1rem; font-size: var(--text-sm); }
.btn-md { padding: 0.75rem 1.5rem; font-size: var(--text-base); }
.btn-lg { padding: 1rem 2rem; font-size: var(--text-lg); }
```

### **Card System**
```css
/* Card Base */
.card {
  background: var(--surface-primary);
  border: 1px solid var(--neutral-800);
  border-radius: 1rem;
  padding: var(--space-6);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  transition: all var(--duration-normal) var(--ease-out-quint);
}

.card:hover {
  border-color: var(--neutral-700);
  box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.2);
  transform: translateY(-2px);
}

/* Glass Morphism Card */
.card-glass {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}
```

### **Input System**
```css
/* Input Base */
.input {
  width: 100%;
  padding: 0.75rem 1rem;
  background: var(--surface-secondary);
  border: 1px solid var(--neutral-700);
  border-radius: 0.5rem;
  color: var(--text-primary);
  font-size: var(--text-base);
  transition: all var(--duration-fast) var(--ease-out-quint);
}

.input:focus {
  outline: none;
  border-color: var(--primary-500);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.input::placeholder {
  color: var(--text-muted);
}
```

---

## 📊 **DATA VISUALIZATION**

### **Chart Color Palette**
```css
/* Data Visualization Colors */
--chart-primary: var(--primary-500);
--chart-secondary: var(--primary-300);
--chart-accent: #8B5CF6;
--chart-success: var(--success);
--chart-warning: var(--warning);
--chart-error: var(--error);

/* Gradient Definitions */
--gradient-primary: linear-gradient(135deg, var(--primary-500), var(--primary-600));
--gradient-risk: linear-gradient(135deg, var(--success), var(--warning), var(--error));
```

### **Progress Indicators**
```css
/* NRI Score Progress Bar */
.nri-progress {
  width: 100%;
  height: 1rem;
  background: var(--neutral-800);
  border-radius: 0.5rem;
  overflow: hidden;
}

.nri-progress-fill {
  height: 100%;
  background: var(--gradient-risk);
  transition: width var(--duration-slow) var(--ease-out-quint);
  border-radius: 0.5rem;
}
```

---

## ♿ **ACCESSIBILITY STANDARDS**

### **WCAG 2.1 AA Compliance**
```css
/* Focus Indicators */
.focus-visible {
  outline: 2px solid var(--primary-500);
  outline-offset: 2px;
}

/* High Contrast Mode Support */
@media (prefers-contrast: high) {
  :root {
    --primary-500: #0066CC;
    --text-primary: #FFFFFF;
    --surface-primary: #000000;
  }
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### **Screen Reader Support**
```css
/* Screen Reader Only Text */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

---

## 📱 **RESPONSIVE DESIGN**

### **Breakpoint System**
```css
/* Mobile-First Breakpoints */
--breakpoint-sm: 640px;   /* Small devices */
--breakpoint-md: 768px;   /* Medium devices */
--breakpoint-lg: 1024px;  /* Large devices */
--breakpoint-xl: 1280px;  /* Extra large devices */
--breakpoint-2xl: 1536px; /* 2X large devices */
```

### **Container System**
```css
/* Responsive Containers */
.container {
  width: 100%;
  margin: 0 auto;
  padding: 0 var(--space-4);
}

@media (min-width: 640px) { .container { max-width: 640px; } }
@media (min-width: 768px) { .container { max-width: 768px; } }
@media (min-width: 1024px) { .container { max-width: 1024px; } }
@media (min-width: 1280px) { .container { max-width: 1280px; } }
```

---

*This design system ensures clinical-grade aesthetics with modern sophistication, accessibility compliance, and professional healthcare application standards.*
