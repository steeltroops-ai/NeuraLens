// NeuroLens-X UI Component Library
// Clinical-grade components with accessibility and design system compliance

// Button Components
export {
  Button,
  IconButton,
  ButtonGroup,
  LoadingButton,
  CopyButton,
} from './Button';

// Card Components
export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  AssessmentCard,
  ResultsCard,
} from './Card';

// Input Components
export { Input, Textarea, SearchInput } from './Input';

// Progress Components
export {
  Progress,
  CircularProgress,
  StepProgress,
  NRIProgress,
} from './Progress';

// Loading Components
export { Loading, Skeleton, LoadingOverlay } from './Loading';

// Re-export types for convenience
export type {
  ButtonProps,
  CardProps,
  InputProps,
  ProgressProps,
  BaseComponentProps,
  ButtonVariant,
  ButtonSize,
  CardVariant,
  RiskLevel,
  ProcessingStatus,
} from '@/types/design-system';

// Re-export utility functions
export { cn } from '@/utils/cn';
export {
  getRiskLevel,
  getRiskColor,
  formatConfidence,
  formatDuration,
} from '@/types/design-system';
