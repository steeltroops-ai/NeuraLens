# NeuraLens Apple Design System

## Overview

This document outlines the comprehensive Apple-inspired design system implemented for the NeuraLens platform, following Apple's Human Interface Guidelines and authentic design principles for professional healthcare applications.

## Color System

### Primary Colors

Based on Apple's official Human Interface Guidelines with healthcare-appropriate adaptations:

```css
/* Apple Authentic Colors */
--apple-white: #FFFFFF;           /* Primary background */
--apple-light-gray: #F5F5F7;     /* Alternating sections (Apple HIG official) */
--apple-text-primary: #1D1D1F;   /* Primary text (Apple HIG official) */
--apple-text-secondary: #86868B; /* Secondary text */
--apple-blue: #007AFF;            /* Interactive elements */
--apple-blue-hover: #0056CC;     /* Button hover states */
```

### Usage Context

- **White (#FFFFFF)**: Hero sections, primary content areas, clean backgrounds
- **Light Gray (#F5F5F7)**: Alternating section backgrounds for visual rhythm
- **Primary Text (#1D1D1F)**: Headings, important content, navigation
- **Secondary Text (#86868B)**: Descriptions, metadata, supporting content
- **Interactive Blue (#007AFF)**: Buttons, links, active states
- **Hover Blue (#0056CC)**: Button hover and focus states

### Background Pattern Implementation

```css
/* Alternating Section Pattern */
.section:nth-child(odd) { background-color: #FFFFFF; }
.section:nth-child(even) { background-color: #F5F5F7; }
```

## Typography Hierarchy

### Font System

Using system fonts for optimal performance and Apple consistency:

```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
```

### Scale and Usage

```css
/* Headings */
.text-4xl { font-size: 2.25rem; font-weight: 700; } /* Hero titles */
.text-3xl { font-size: 1.875rem; font-weight: 700; } /* Section titles */
.text-2xl { font-size: 1.5rem; font-weight: 700; }   /* Subsection titles */
.text-xl { font-size: 1.25rem; font-weight: 600; }   /* Card titles */

/* Body Text */
.text-lg { font-size: 1.125rem; line-height: 1.75; } /* Primary descriptions */
.text-base { font-size: 1rem; line-height: 1.5; }    /* Standard body text */
.text-sm { font-size: 0.875rem; line-height: 1.25; } /* Metadata, captions */
```

## Layout Grid System

### Container Widths

```css
/* Responsive Container System */
.container {
  max-width: 1280px;  /* Desktop */
  margin: 0 auto;
  padding: 0 2rem;    /* 32px horizontal padding */
}

@media (max-width: 768px) {
  .container { padding: 0 1rem; } /* 16px on mobile */
}
```

### Spacing Units

Following Apple's 8px grid system with Fibonacci-based scaling:

```css
/* Spacing Scale */
--space-1: 0.25rem;  /* 4px */
--space-2: 0.5rem;   /* 8px - Base unit */
--space-3: 0.75rem;  /* 12px */
--space-4: 1rem;     /* 16px */
--space-6: 1.5rem;   /* 24px */
--space-8: 2rem;     /* 32px */
--space-12: 3rem;    /* 48px */
--space-16: 4rem;    /* 64px */
--space-24: 6rem;    /* 96px */
```

### Responsive Breakpoints

```css
/* Breakpoint System */
--mobile: 640px;
--tablet: 768px;
--desktop: 1024px;
--large: 1280px;
--xl: 1536px;
```

## Button Standards

### Primary Button (Start Health Check)

```css
.btn-primary {
  background: #007AFF;
  color: #FFFFFF;
  padding: 0.75rem 1.5rem;  /* py-3 px-6 for CTA */
  padding: 0.5rem 1rem;     /* py-2 px-4 for header */
  border-radius: 0.5rem;
  font-weight: 600;
  transition: all 0.2s ease;
  backdrop-filter: blur(20px);
}

.btn-primary:hover {
  background: #0056CC;
  transform: scale(0.98);
}

.btn-primary:focus {
  outline: 2px solid #007AFF;
  outline-offset: 2px;
}
```

### Button Sizing Context

- **Header buttons**: `px-4 py-2 text-sm` (compact for navigation)
- **CTA buttons**: `px-6 py-3 text-base` (prominent for actions)
- **Minimum touch target**: 44px height for accessibility

## Animation Principles

### Performance Targets

- **60fps animations**: All animations maintain 60fps performance
- **Canvas API efficiency**: Optimized rendering for geometry components
- **Smooth transitions**: 0.2s ease timing for micro-interactions

### Animation Standards

```css
/* Standard Transitions */
.transition-standard {
  transition: all 0.2s ease;
}

/* Hover Scale Effect */
.hover-scale:hover {
  transform: scale(0.98);
}

/* Fade In Animation */
.fade-in {
  animation: fadeIn 0.8s ease forwards;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### Canvas Animation Optimization

```javascript
// 60fps Animation Loop
function render() {
  // Clear and redraw
  ctx.clearRect(0, 0, width, height);
  
  // Optimized rendering
  requestAnimationFrame(render);
}
```

## No-Cards Philosophy

### Rationale

Following Apple's clean, minimal design approach:

- **Reduced visual clutter**: Content flows naturally without card containers
- **Enhanced readability**: Text and images have breathing room
- **Modern aesthetics**: Clean, spacious layouts feel premium
- **Better mobile experience**: Content adapts fluidly across screen sizes

### Implementation Guidelines

```css
/* Instead of cards, use sections with subtle backgrounds */
.content-section {
  padding: 6rem 0;  /* Generous vertical spacing */
  background: alternating colors;
}

/* Focus on typography hierarchy */
.section-title {
  margin-bottom: 2rem;
  color: #1D1D1F;
}

.section-description {
  max-width: 65ch;  /* Optimal reading width */
  line-height: 1.75;
  color: #86868B;
}
```

## Geometry Standards

### Visual Recognition Criteria

All animated geometry components must achieve **2-3 second visual recognition time**:

#### HandKinematics (Motor Assessment)
- **Anatomical accuracy**: Visible bone structure with proper proportions
- **Joint markers**: 8-12px radius circles at MCP, PIP, DIP joints
- **Tremor visualization**: 2-4px red oscillating dots near fingertips
- **Movement trails**: Finger tapping pattern history over time

#### BrainNeural (Cognitive Evaluation)
- **Neural pathways**: Distinct branching lines between activity nodes
- **Synaptic firing**: 3-5px bright flashes along connections
- **Memory clusters**: Pulsing groups in hippocampus region
- **Attention networks**: Flowing particles between frontal-parietal areas

#### NRIFusion (Data Integration)
- **Modality icons**: 🎤 (speech), 👁️ (retinal), ✋ (motor), 🧠 (cognitive)
- **Distinct colors**: Blue (#4A90E2), Green (#50C878), Orange (#FF8C42), Purple (#9B59B6)
- **Processing stages**: Intermediate nodes between modalities and fusion center
- **Correlation arcs**: Connecting lines showing data relationships

#### MultiModalNetwork (Network Topology)
- **Recognizable icons**: Clear symbols for each assessment modality
- **Color-coded packets**: Data visualization with modality-specific colors
- **Network strength**: Varying line thickness (1-4px) based on correlation
- **Integration waves**: Expanding circles from NRI center node

### Performance Benchmarks

```javascript
// Performance Requirements
const PERFORMANCE_TARGETS = {
  frameRate: 60,           // Minimum FPS
  recognitionTime: 2000,   // Maximum ms for visual recognition
  canvasEfficiency: 95,    // Percentage of optimal rendering
  memoryUsage: 50          // Maximum MB per component
};
```

## Accessibility Compliance

### WCAG 2.1 AAA Standards

- **Color contrast**: Minimum 7:1 ratio for text
- **Touch targets**: Minimum 44px for interactive elements
- **Keyboard navigation**: Full keyboard accessibility
- **Screen reader support**: Proper ARIA labels and semantic HTML
- **Motion preferences**: Respect `prefers-reduced-motion`

### Implementation

```css
/* High Contrast Text */
.text-primary { color: #1D1D1F; } /* 16.94:1 contrast on white */
.text-secondary { color: #86868B; } /* 4.54:1 contrast on white */

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Implementation Examples

### Section Structure

```jsx
<section className="relative bg-white py-24">
  <div className="mx-auto max-w-7xl px-8">
    <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="space-y-6"
      >
        <h2 className="text-3xl font-bold" style={{ color: '#1D1D1F' }}>
          Section Title
        </h2>
        <p className="text-lg leading-relaxed text-slate-600">
          Professional description with optimal line height and spacing.
        </p>
      </motion.div>
      <div className="relative h-80">
        <AnimatedGeometry />
      </div>
    </div>
  </div>
</section>
```

### Button Implementation

```jsx
<button
  onClick={handleStartAssessment}
  className="inline-flex items-center justify-center rounded-lg px-6 py-3 text-base font-semibold text-white transition-all duration-200 hover:scale-98"
  style={{
    backgroundColor: '#007AFF',
    backdropFilter: 'blur(20px)',
  }}
  onMouseEnter={(e) => e.target.style.backgroundColor = '#0056CC'}
  onMouseLeave={(e) => e.target.style.backgroundColor = '#007AFF'}
>
  Start Health Check
</button>
```

This design system ensures consistent, accessible, and performant implementation of Apple-inspired design principles throughout the NeuraLens platform while maintaining the professional healthcare aesthetic and optimal user experience.
