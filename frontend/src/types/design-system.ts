// NeuroLens-X Design System Types

/* ===== COLOR SYSTEM TYPES ===== */

export type PrimaryColor =
  | 'primary-50'
  | 'primary-100'
  | 'primary-200'
  | 'primary-300'
  | 'primary-400'
  | 'primary-500'
  | 'primary-600'
  | 'primary-700'
  | 'primary-800'
  | 'primary-900'
  | 'primary-950';

export type NeutralColor =
  | 'neutral-50'
  | 'neutral-100'
  | 'neutral-200'
  | 'neutral-300'
  | 'neutral-400'
  | 'neutral-500'
  | 'neutral-600'
  | 'neutral-700'
  | 'neutral-800'
  | 'neutral-900'
  | 'neutral-950';

export type SemanticColor = 'success' | 'warning' | 'error' | 'info';

export type RiskColor = 'risk-low' | 'risk-moderate' | 'risk-high' | 'risk-critical';

export type SurfaceColor =
  | 'surface-background'
  | 'surface-primary'
  | 'surface-secondary'
  | 'surface-tertiary'
  | 'surface-overlay'
  | 'surface-glass';

export type TextColor = 'text-primary' | 'text-secondary' | 'text-muted' | 'text-inverse';

export type ColorToken =
  | PrimaryColor
  | NeutralColor
  | SemanticColor
  | RiskColor
  | SurfaceColor
  | TextColor;

/* ===== TYPOGRAPHY TYPES ===== */

export type FontFamily = 'font-display' | 'font-body' | 'font-mono';

export type FontSize =
  | 'text-xs'
  | 'text-sm'
  | 'text-base'
  | 'text-lg'
  | 'text-xl'
  | 'text-2xl'
  | 'text-3xl'
  | 'text-4xl'
  | 'text-5xl'
  | 'text-6xl'
  | 'text-7xl'
  | 'text-8xl'
  | 'text-9xl';

export type FontWeight =
  | 'font-thin'
  | 'font-extralight'
  | 'font-light'
  | 'font-normal'
  | 'font-medium'
  | 'font-semibold'
  | 'font-bold'
  | 'font-extrabold'
  | 'font-black';

export type LineHeight =
  | 'leading-none'
  | 'leading-tight'
  | 'leading-snug'
  | 'leading-normal'
  | 'leading-relaxed'
  | 'leading-loose';

export type LetterSpacing =
  | 'tracking-tighter'
  | 'tracking-tight'
  | 'tracking-normal'
  | 'tracking-wide'
  | 'tracking-wider'
  | 'tracking-widest';

/* ===== SPACING TYPES ===== */

export type SpacingToken =
  | 'space-0'
  | 'space-px'
  | 'space-0-5'
  | 'space-1'
  | 'space-1-5'
  | 'space-2'
  | 'space-2-5'
  | 'space-3'
  | 'space-3-5'
  | 'space-4'
  | 'space-5'
  | 'space-6'
  | 'space-7'
  | 'space-8'
  | 'space-9'
  | 'space-10'
  | 'space-11'
  | 'space-12'
  | 'space-14'
  | 'space-16'
  | 'space-20'
  | 'space-24'
  | 'space-28'
  | 'space-32'
  | 'space-36'
  | 'space-40'
  | 'space-44'
  | 'space-48'
  | 'space-52'
  | 'space-56'
  | 'space-60'
  | 'space-64'
  | 'space-72'
  | 'space-80'
  | 'space-96';

export type ComponentPadding =
  | 'component-padding-sm'
  | 'component-padding-md'
  | 'component-padding-lg'
  | 'component-padding-xl';

export type LayoutGap = 'layout-gap-sm' | 'layout-gap-md' | 'layout-gap-lg' | 'layout-gap-xl';

/* ===== ANIMATION TYPES ===== */

export type AnimationDuration =
  | 'duration-instant'
  | 'duration-fast'
  | 'duration-normal'
  | 'duration-slow'
  | 'duration-slower'
  | 'duration-slowest';

export type AnimationEasing =
  | 'ease-linear'
  | 'ease-in'
  | 'ease-out'
  | 'ease-in-out'
  | 'ease-out-quint'
  | 'ease-in-out-cubic'
  | 'ease-in-cubic'
  | 'ease-spring'
  | 'ease-bounce';

/* ===== COMPONENT TYPES ===== */

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'destructive';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'xl';

export type CardVariant = 'default' | 'glass' | 'clinical' | 'results';

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

export type ProcessingStatus = 'pending' | 'processing' | 'completed' | 'error';

/* ===== BORDER RADIUS TYPES ===== */

export type BorderRadius =
  | 'radius-none'
  | 'radius-sm'
  | 'radius-base'
  | 'radius-md'
  | 'radius-lg'
  | 'radius-xl'
  | 'radius-2xl'
  | 'radius-3xl'
  | 'radius-full';

/* ===== SHADOW TYPES ===== */

export type ShadowToken =
  | 'shadow-sm'
  | 'shadow-base'
  | 'shadow-md'
  | 'shadow-lg'
  | 'shadow-xl'
  | 'shadow-2xl'
  | 'shadow-inner'
  | 'shadow-clinical'
  | 'shadow-clinical-hover'
  | 'shadow-glass';

/* ===== BREAKPOINT TYPES ===== */

export type Breakpoint = 'sm' | 'md' | 'lg' | 'xl' | '2xl';

/* ===== Z-INDEX TYPES ===== */

export type ZIndex =
  | 'z-0'
  | 'z-10'
  | 'z-20'
  | 'z-30'
  | 'z-40'
  | 'z-50'
  | 'z-auto'
  | 'z-dropdown'
  | 'z-sticky'
  | 'z-fixed'
  | 'z-modal-backdrop'
  | 'z-modal'
  | 'z-popover'
  | 'z-tooltip'
  | 'z-toast';

/* ===== COMPONENT PROP INTERFACES ===== */

export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
  testId?: string;
  'aria-label'?: string;
  'aria-describedby'?: string;
  'aria-live'?: 'polite' | 'assertive' | 'off';
  role?: string;
  tabIndex?: number;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
  form?: string;
}

export interface CardProps extends BaseComponentProps {
  variant?: CardVariant;
  padding?: ComponentPadding;
  hover?: boolean;
  onClick?: () => void;
}

export interface InputProps extends BaseComponentProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url' | 'search';
  placeholder?: string;
  value?: string;
  defaultValue?: string;
  disabled?: boolean;
  required?: boolean;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  min?: string | number;
  max?: string | number;
  step?: string | number;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onBlur?: (event: React.FocusEvent<HTMLInputElement>) => void;
  onFocus?: (event: React.FocusEvent<HTMLInputElement>) => void;
}

export interface ProgressProps extends BaseComponentProps {
  value: number; // 0-100
  max?: number;
  variant?: 'default' | 'risk' | 'clinical';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  animated?: boolean;
}

export interface NRIScoreProps extends BaseComponentProps {
  score: number; // 0-100
  confidence: number; // ±percentage
  category: RiskLevel;
  animated?: boolean;
  showRecommendation?: boolean;
  processingTime?: number; // seconds
}

export interface ProcessingStatusProps extends BaseComponentProps {
  tasks: Array<{
    id: string;
    label: string;
    icon: string;
    status: ProcessingStatus;
    progress?: number; // 0-100
    duration?: number; // seconds
    error?: string;
  }>;
  estimatedTimeRemaining?: number; // seconds
  onCancel?: () => void;
}

export interface ResultsBreakdownProps extends BaseComponentProps {
  results: Array<{
    modality: 'speech' | 'retinal' | 'risk' | 'motor';
    score: number; // 0-100
    category: RiskLevel;
    findings: string[];
    confidence: number; // ±percentage
    processingTime: number; // seconds
    details?: Record<string, any>;
  }>;
  onViewDetails: (modality: string) => void;
}

export interface RecommendationsProps extends BaseComponentProps {
  riskLevel: RiskLevel;
  immediateActions: string[];
  followUpCare: string[];
  timeframe: string;
  showDisclaimer?: boolean;
  clinicalNotes?: string[];
}

export interface FileUploadProps extends BaseComponentProps {
  acceptedTypes: string[];
  maxSize: number; // bytes
  multiple?: boolean;
  onFileSelect: (files: File[]) => void;
  onError: (error: string) => void;
  showDemoOption?: boolean;
  demoFiles?: Array<{ name: string; url: string; description?: string }>;
  dragActive?: boolean;
}

export interface AudioRecorderProps extends BaseComponentProps {
  maxDuration: number; // seconds
  onRecordingComplete: (blob: Blob) => void;
  onError: (error: string) => void;
  showWaveform?: boolean;
  showQualityIndicator?: boolean;
  autoStart?: boolean;
}

/* ===== ASSESSMENT TYPES ===== */

export interface AssessmentStep {
  id: string;
  label: string;
  description: string;
  status: 'completed' | 'current' | 'pending' | 'error';
  estimatedDuration: number; // seconds
  required: boolean;
}

export interface AssessmentProgress {
  currentStep: number;
  totalSteps: number;
  steps: AssessmentStep[];
  overallProgress: number; // 0-100
  timeElapsed: number; // seconds
  estimatedTimeRemaining: number; // seconds
}

export interface AssessmentResults {
  id: string;
  timestamp: Date;
  nriScore: number;
  riskCategory: RiskLevel;
  confidence: number;
  processingTime: number;
  modalities: {
    speech?: {
      score: number;
      findings: string[];
      confidence: number;
      audioFile?: File;
      features?: Record<string, number>;
    };
    retinal?: {
      score: number;
      findings: string[];
      confidence: number;
      imageFile?: File;
      measurements?: Record<string, number>;
    };
    risk?: {
      score: number;
      factors: Record<string, any>;
      modifiableFactors: string[];
      nonModifiableFactors: string[];
    };
    motor?: {
      score: number;
      findings: string[];
      confidence: number;
      measurements?: Record<string, number>;
    };
  };
  recommendations: {
    immediate: string[];
    followUp: string[];
    lifestyle: string[];
    timeframe: string;
  };
  clinicalNotes?: string;
  reportGenerated?: boolean;
}

/* ===== THEME TYPES ===== */

export interface ThemeConfig {
  colorMode: 'light' | 'dark' | 'system';
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: 'sm' | 'base' | 'lg' | 'xl';
  language: string;
  accessibility: {
    screenReader: boolean;
    keyboardNavigation: boolean;
    focusVisible: boolean;
  };
}

/* ===== UTILITY TYPES ===== */

export type ResponsiveValue<T> = T | Partial<Record<Breakpoint, T>>;

export type VariantProps<T extends Record<string, any>> = {
  [K in keyof T]?: T[K] extends Record<string, any> ? keyof T[K] : T[K];
};

export type ComponentVariants<T> = {
  [K in keyof T]: T[K];
};

/* ===== DESIGN TOKEN MAPS ===== */

export const RISK_COLORS: Record<RiskLevel, string> = {
  low: 'var(--risk-low)',
  moderate: 'var(--risk-moderate)',
  high: 'var(--risk-high)',
  critical: 'var(--risk-critical)',
};

export const RISK_THRESHOLDS: Record<RiskLevel, [number, number]> = {
  low: [0, 25],
  moderate: [26, 50],
  high: [51, 75],
  critical: [76, 100],
};

export const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: 'btn-sm',
  md: 'btn-md',
  lg: 'btn-lg',
  xl: 'btn-xl',
};

export const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  ghost: 'btn-ghost',
  destructive: 'btn-destructive',
};

/* ===== HELPER FUNCTIONS ===== */

export const getRiskLevel = (score: number): RiskLevel => {
  if (score <= 25) return 'low';
  if (score <= 50) return 'moderate';
  if (score <= 75) return 'high';
  return 'critical';
};

export const getRiskColor = (level: RiskLevel): string => {
  return RISK_COLORS[level];
};

export const formatConfidence = (confidence: number): string => {
  return `±${confidence.toFixed(1)}%`;
};

export const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
};
