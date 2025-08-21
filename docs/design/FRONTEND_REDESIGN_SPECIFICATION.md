# NeuroLens-X: Frontend Redesign Specification

## 🎯 **Design Philosophy: Apple iOS 18-Inspired Medical Interface**

### **Core Principles**
- **Minimalist Elegance**: Clean white backgrounds, subtle shadows, generous whitespace
- **Intuitive Interactions**: Natural gestures, smooth transitions, satisfying feedback
- **Medical-Grade Trust**: Professional appearance with consumer-friendly warmth
- **Accessibility First**: WCAG 2.1 AA+ compliance, large touch targets, high contrast
- **Performance Excellence**: <2s load times, 60fps animations, optimized assets

### **Visual Style Guide**
- **Rounded Corners**: 12px standard, 20px for cards, 50% for buttons
- **Soft Gradients**: Subtle medical blue to white backgrounds
- **High Contrast**: 4.5:1 minimum ratio for text
- **Tactile Buttons**: Elevated appearance with press animations
- **Smooth Transitions**: 0.3s ease-in-out for all interactions

---

## 🎨 **Design System Specifications**

### **Color Palette (Medical + Apple)**
```css
/* Primary Colors */
--medical-blue: #007AFF;      /* Apple blue - trust, technology */
--success-green: #34C759;     /* Healthy results, positive outcomes */
--warning-orange: #FF9500;    /* Moderate risk, attention needed */
--error-red: #FF3B30;         /* High risk, urgent action */
--neural-purple: #AF52DE;     /* AI/brain theme, innovation */

/* Neutral Colors */
--background: #F2F2F7;        /* Light gray background */
--card-white: #FFFFFF;        /* Pure white cards */
--text-primary: #000000;      /* Maximum readability */
--text-secondary: #3C3C43;    /* Secondary text */
--border-gray: #C6C6C8;       /* Subtle borders */
--shadow: rgba(0, 0, 0, 0.1); /* Soft shadows */
```

### **Typography System (Apple-Inspired)**
```css
/* Font Stack */
font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', system-ui, sans-serif;

/* Type Scale */
--text-xs: 12px;    /* Captions, fine print */
--text-sm: 14px;    /* Secondary text */
--text-base: 17px;  /* Body text (Apple standard) */
--text-lg: 20px;    /* Subheadings */
--text-xl: 24px;    /* Section headers */
--text-2xl: 28px;   /* Page titles */
--text-3xl: 34px;   /* Hero headlines */

/* Font Weights */
--font-regular: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
```

### **Spacing System (8px Grid)**
```css
/* Spacing Scale */
--space-1: 4px;     /* Micro spacing */
--space-2: 8px;     /* Small spacing */
--space-3: 12px;    /* Medium spacing */
--space-4: 16px;    /* Standard spacing */
--space-5: 20px;    /* Large spacing */
--space-6: 24px;    /* Section spacing */
--space-8: 32px;    /* Major spacing */
--space-10: 40px;   /* Hero spacing */
--space-12: 48px;   /* Page spacing */
--space-16: 64px;   /* Massive spacing */
```

### **Animation System**
```css
/* Durations */
--duration-fast: 0.15s;     /* Micro-interactions */
--duration-normal: 0.3s;    /* Standard transitions */
--duration-slow: 0.5s;      /* Complex animations */

/* Easing Functions */
--ease-out: cubic-bezier(0, 0, 0.2, 1);
--ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
--ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
```

---

## 📱 **Component Design Specifications**

### **Button System**
```css
/* Primary Button */
.btn-primary {
  background: linear-gradient(135deg, #007AFF 0%, #0056CC 100%);
  color: white;
  padding: 16px 32px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 17px;
  box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
  transition: all 0.15s ease-out;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 122, 255, 0.4);
}

.btn-primary:active {
  transform: translateY(0);
  box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
}
```

### **Card System**
```css
.card {
  background: white;
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 2px 16px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease-out;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}
```

### **Input System**
```css
.input-field {
  background: #F2F2F7;
  border: 2px solid transparent;
  border-radius: 12px;
  padding: 16px 20px;
  font-size: 17px;
  transition: all 0.15s ease-out;
}

.input-field:focus {
  background: white;
  border-color: #007AFF;
  box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
}
```

---

## 🏗️ **Page Layout Specifications**

### **Landing Page Layout**
```
Header (80px height)
├── Logo + Navigation
└── CTA Button

Hero Section (100vh)
├── Gradient Background
├── Floating Brain Icon (3D effect)
├── Headline: "90-Second Neurological Screening"
├── Subline: "Detect risks before symptoms appear"
└── Primary CTA: "Start Assessment"

Trust Section (400px)
├── Clinical Validation Badges
├── University Logos
├── HIPAA Compliance Icon
└── Patient Testimonials

Features Section (600px)
├── 4-Modal Assessment Cards
├── Real-time Processing
├── Clinical Validation
└── Accessibility Features

Footer (200px)
├── Links & Legal
└── Contact Information
```

### **Assessment Flow Layout**
```
Progress Header (60px)
├── Step Indicator (1 of 4)
├── Progress Bar
└── Back Button

Main Content (calc(100vh - 120px))
├── Step Title
├── Instructions
├── Interactive Element
└── Continue Button

Bottom Navigation (60px)
├── Skip Option
└── Next/Continue Button
```

---

## 🎭 **Micro-Interaction Specifications**

### **Button Interactions**
- **Hover**: 2px upward translation + shadow increase
- **Press**: Scale down to 0.98 + shadow decrease
- **Loading**: Spinner animation + disabled state
- **Success**: Checkmark animation + green flash

### **Card Interactions**
- **Hover**: 4px upward translation + shadow increase
- **Tap**: Brief scale down to 0.98
- **Loading**: Skeleton animation
- **Error**: Red border flash + shake animation

### **Form Interactions**
- **Focus**: Background color change + border highlight
- **Validation**: Real-time feedback with icons
- **Error**: Red border + shake + error message
- **Success**: Green border + checkmark icon

---

## 📊 **Assessment Component Specifications**

### **Speech Assessment Interface**
```
Large Circular Record Button (120px)
├── Microphone Icon (24px)
├── Pulsing Animation When Active
├── Timer Display (30s countdown)
└── Waveform Visualization

Text Passage Display
├── Large, Readable Font (20px)
├── Highlighted Current Sentence
├── Progress Indicator
└── Scroll Lock During Recording

Controls
├── Re-record Button
├── Play Back Button
├── Continue Button (appears after completion)
└── Skip Option
```

### **Retinal Assessment Interface**
```
Camera Viewfinder (Full Screen)
├── Eye Outline Overlay
├── Alignment Guide
├── Auto-capture Indicator
└── Flash Control

Alternative Options
├── Gallery Upload Button
├── Demo Image Selection
├── Quality Check Display
└── Retake Option

Processing State
├── Analysis Animation
├── Progress Indicator
├── Quality Assessment
└── Results Preview
```

### **Risk Assessment Interface**
```
Card-Based Form Sections
├── Demographics Card
├── Medical History Card
├── Lifestyle Factors Card
└── Family History Card

Interactive Elements
├── Age Slider (smooth animation)
├── Toggle Switches (iOS style)
├── Multi-select Checkboxes
└── Smart Suggestions

Progress Tracking
├── Section Completion Indicators
├── Overall Progress Bar
├── Validation Feedback
└── Save & Continue Options
```

### **Results Dashboard Interface**
```
Hero Results Section
├── Large NRI Score Circle (150px)
├── Color-coded Risk Level
├── Percentile Comparison
└── Primary Action Button

Module Breakdown Cards
├── Speech Analysis Card
├── Retinal Analysis Card
├── Risk Factors Card
└── Recommendations Card

Detailed View
├── Expandable Sections
├── Clinical Explanations
├── Action Items
└── Share/Download Options
```

---

## 🔧 **Technical Implementation Notes**

### **CSS Framework Integration**
- Extend Tailwind CSS with custom design tokens
- Create component classes for consistency
- Implement dark mode support
- Optimize for mobile-first responsive design

### **Animation Implementation**
- Use CSS transitions for simple animations
- Implement Framer Motion for complex interactions
- Ensure 60fps performance on all devices
- Respect `prefers-reduced-motion` settings

### **Accessibility Implementation**
- Semantic HTML structure
- ARIA labels and descriptions
- Keyboard navigation support
- Screen reader optimization
- High contrast mode support

### **Performance Optimization**
- Lazy load images and components
- Optimize font loading
- Minimize CSS bundle size
- Implement efficient re-renders

This specification provides the foundation for creating a world-class, Apple-inspired medical interface that will dominate the competition while maintaining the highest standards of usability and accessibility.
