# Component Documentation

## Overview

This directory contains comprehensive documentation for all React components in the NeuraLens application. Each component is documented with its props, usage examples, and implementation details.

## Component Categories

### 🎛️ Dashboard Components

- **[SpeechAssessment](./dashboard/SpeechAssessment.md)** - Voice recording and analysis component
- **[RetinalAssessment](./dashboard/RetinalAssessment.md)** - Retinal image upload and analysis
- **[MotorAssessment](./dashboard/MotorAssessment.md)** - Interactive motor function testing
- **[AssessmentHistory](./dashboard/AssessmentHistory.md)** - CRUD operations for assessment management
- **[DashboardCRUD](./dashboard/DashboardCRUD.md)** - Main dashboard with navigation

### 🧪 Assessment Components

- **[AssessmentStep](./assessment/AssessmentStep.md)** - Base assessment step component
- **[SpeechAssessmentStep](./assessment/SpeechAssessmentStep.md)** - Speech assessment workflow step
- **[RetinalAssessmentStep](./assessment/RetinalAssessmentStep.md)** - Retinal assessment workflow step
- **[MotorAssessmentStep](./assessment/MotorAssessmentStep.md)** - Motor assessment workflow step

### 🎨 UI Components

- **[Button](./ui/Button.md)** - Reusable button component with variants
- **[Input](./ui/Input.md)** - Form input components
- **[Modal](./ui/Modal.md)** - Modal dialog component
- **[LoadingSpinner](./ui/LoadingSpinner.md)** - Loading state indicators
- **[ProgressBar](./ui/ProgressBar.md)** - Progress indication component

### 📊 Data Visualization

- **[RiskScoreChart](./charts/RiskScoreChart.md)** - Risk score visualization
- **[BiomarkerChart](./charts/BiomarkerChart.md)** - Biomarker data charts
- **[TrendChart](./charts/TrendChart.md)** - Trend analysis visualization

## Component Standards

### Props Documentation

Each component should document:

```typescript
interface ComponentProps {
  /** Required prop description */
  requiredProp: string;
  
  /** Optional prop with default value */
  optionalProp?: boolean;
  
  /** Callback function prop */
  onAction?: (data: any) => void;
  
  /** Complex object prop */
  config?: {
    setting1: string;
    setting2: number;
  };
}
```

### Usage Examples

Include practical usage examples:

```tsx
// Basic usage
<Component requiredProp="value" />

// Advanced usage with all props
<Component
  requiredProp="value"
  optionalProp={true}
  onAction={(data) => console.log(data)}
  config={{
    setting1: "example",
    setting2: 42
  }}
/>
```

### Accessibility Notes

Document accessibility features:

- ARIA labels and roles
- Keyboard navigation support
- Screen reader compatibility
- Focus management

### Testing Guidelines

Include testing recommendations:

- Unit test examples
- Integration test scenarios
- Accessibility testing
- Performance considerations

## Design System Integration

All components follow the NeuraLens design system:

- **8px grid system** for consistent spacing
- **Color tokens** from the design system palette
- **Typography scale** with semantic font sizes
- **Motion design** with 60fps animations
- **Glassmorphism** aesthetic for premium feel

## Performance Considerations

Components are optimized for:

- **Bundle size** - Tree-shaking and code splitting
- **Runtime performance** - React.memo and useMemo where appropriate
- **Accessibility** - WCAG 2.1 AA+ compliance
- **Mobile responsiveness** - Touch-friendly interactions

## Contributing

When adding new components:

1. Create component documentation in the appropriate category
2. Include TypeScript interfaces for all props
3. Provide usage examples and accessibility notes
4. Add unit tests with good coverage
5. Follow the established design system patterns

## Component Checklist

Before marking a component as complete:

- [ ] TypeScript interfaces documented
- [ ] Props table with descriptions
- [ ] Usage examples provided
- [ ] Accessibility features documented
- [ ] Unit tests written and passing
- [ ] Design system compliance verified
- [ ] Performance optimizations applied
- [ ] Mobile responsiveness tested
