// NeuroLens-X User Flow & Navigation Types

/* ===== NAVIGATION TYPES ===== */

export type RouteId =
  | 'home'
  | 'assessment-intro'
  | 'assessment-speech'
  | 'assessment-retinal'
  | 'assessment-risk'
  | 'assessment-processing'
  | 'dashboard'
  | 'privacy'
  | 'help';

export interface Route {
  id: RouteId;
  path: string;
  title: string;
  description: string;
  component: string;
  protected?: boolean;
  showInNav?: boolean;
  estimatedDuration?: number; // seconds
}

export interface NavigationItem {
  id: string;
  label: string;
  route: RouteId;
  icon?: string;
  badge?: string | number;
  disabled?: boolean;
  external?: boolean;
  children?: NavigationItem[];
}

/* ===== USER FLOW TYPES ===== */

export interface UserFlowStep {
  id: string;
  title: string;
  description: string;
  route: RouteId;
  required: boolean;
  estimatedDuration: number; // seconds
  dependencies?: string[]; // step IDs that must be completed first
  validationRules?: ValidationRule[];
  exitPoints?: ExitPoint[];
}

export interface ValidationRule {
  id: string;
  field: string;
  type: 'required' | 'format' | 'size' | 'custom';
  message: string;
  validator?: (value: any) => boolean;
}

export interface ExitPoint {
  id: string;
  condition: string;
  targetRoute: RouteId;
  message?: string;
}

export interface UserJourney {
  id: string;
  name: string;
  description: string;
  steps: UserFlowStep[];
  totalEstimatedDuration: number; // seconds
  completionRate?: number; // 0-100
  dropOffPoints?: string[]; // step IDs where users commonly exit
}

/* ===== ASSESSMENT FLOW TYPES ===== */

export interface AssessmentFlowState {
  currentStepId: string;
  completedSteps: string[];
  stepData: Record<string, any>;
  startTime: Date;
  lastActivityTime: Date;
  estimatedTimeRemaining: number; // seconds
  canGoBack: boolean;
  canSkip: boolean;
  autoSave: boolean;
}

export interface StepTransition {
  fromStep: string;
  toStep: string;
  condition?: (state: AssessmentFlowState) => boolean;
  animation?: 'slide' | 'fade' | 'scale';
  duration?: number; // milliseconds
}

/* ===== FORM FLOW TYPES ===== */

export interface FormField {
  id: string;
  name: string;
  type:
    | 'text'
    | 'number'
    | 'select'
    | 'checkbox'
    | 'radio'
    | 'file'
    | 'textarea';
  label: string;
  placeholder?: string;
  required: boolean;
  validation?: ValidationRule[];
  options?: Array<{ value: string; label: string }>;
  helpText?: string;
  dependencies?: Array<{ field: string; value: any }>;
  accessibility?: {
    ariaLabel?: string;
    ariaDescribedBy?: string;
    screenReaderText?: string;
  };
}

export interface FormSection {
  id: string;
  title: string;
  description?: string;
  fields: FormField[];
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

export interface FormFlow {
  id: string;
  title: string;
  sections: FormSection[];
  submitLabel: string;
  cancelLabel?: string;
  showProgress: boolean;
  autoSave: boolean;
  validationMode: 'onChange' | 'onBlur' | 'onSubmit';
}

/* ===== ACCESSIBILITY FLOW TYPES ===== */

export interface AccessibilityFeature {
  id: string;
  type: 'screenReader' | 'keyboard' | 'voice' | 'motor' | 'cognitive';
  enabled: boolean;
  settings?: Record<string, any>;
}

export interface AccessibilityState {
  features: AccessibilityFeature[];
  announcements: string[];
  focusManagement: {
    currentFocus?: string;
    focusHistory: string[];
    skipLinks: Array<{ target: string; label: string }>;
  };
  preferences: {
    reducedMotion: boolean;
    highContrast: boolean;
    largeText: boolean;
    voiceNavigation: boolean;
  };
}

/* ===== ERROR HANDLING TYPES ===== */

export interface ErrorState {
  id: string;
  type: 'validation' | 'network' | 'processing' | 'permission' | 'timeout';
  message: string;
  details?: string;
  recoverable: boolean;
  retryAction?: () => void;
  fallbackRoute?: RouteId;
  timestamp: Date;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: any;
  fallbackComponent?: string;
  retryCount: number;
  maxRetries: number;
}

/* ===== PROGRESS TRACKING TYPES ===== */

export interface ProgressState {
  currentStep: number;
  totalSteps: number;
  completedSteps: number;
  percentage: number; // 0-100
  timeElapsed: number; // seconds
  estimatedTimeRemaining: number; // seconds
  milestones: Array<{
    id: string;
    label: string;
    completed: boolean;
    timestamp?: Date;
  }>;
}

/* ===== ANALYTICS TYPES ===== */

export interface UserInteraction {
  id: string;
  type: 'click' | 'focus' | 'input' | 'scroll' | 'navigation' | 'error';
  element: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface FlowAnalytics {
  sessionId: string;
  userId?: string;
  startTime: Date;
  endTime?: Date;
  interactions: UserInteraction[];
  completionRate: number; // 0-100
  dropOffPoint?: string;
  errors: ErrorState[];
  performance: {
    loadTime: number;
    renderTime: number;
    interactionTime: number;
  };
}

/* ===== CONSTANTS ===== */

export const ROUTES: Record<RouteId, Route> = {
  home: {
    id: 'home',
    path: '/',
    title: 'NeuroLens-X',
    description: 'Multi-modal neurological risk assessment platform',
    component: 'HomePage',
    showInNav: true,
  },
  'assessment-intro': {
    id: 'assessment-intro',
    path: '/assessment',
    title: 'Assessment Introduction',
    description: 'Learn about the neurological risk assessment process',
    component: 'AssessmentIntroPage',
    estimatedDuration: 60,
  },
  'assessment-speech': {
    id: 'assessment-speech',
    path: '/assessment/speech',
    title: 'Speech Analysis',
    description: 'Record speech sample for voice biomarker analysis',
    component: 'SpeechAssessmentPage',
    estimatedDuration: 120,
  },
  'assessment-retinal': {
    id: 'assessment-retinal',
    path: '/assessment/retinal',
    title: 'Retinal Imaging',
    description: 'Upload retinal image for vascular pattern analysis',
    component: 'RetinalAssessmentPage',
    estimatedDuration: 90,
  },
  'assessment-risk': {
    id: 'assessment-risk',
    path: '/assessment/risk',
    title: 'Risk Assessment',
    description: 'Complete health and lifestyle questionnaire',
    component: 'RiskAssessmentPage',
    estimatedDuration: 180,
  },
  'assessment-processing': {
    id: 'assessment-processing',
    path: '/assessment/processing',
    title: 'Processing Results',
    description: 'AI analysis of your assessment data',
    component: 'ProcessingPage',
    estimatedDuration: 45,
  },
  // Results are now shown inline in assessment flow
  dashboard: {
    id: 'dashboard',
    path: '/dashboard',
    title: 'Dashboard',
    description: 'Track your neurological health over time',
    component: 'DashboardPage',
    protected: true,
  },
  // About page removed for streamlined experience
  privacy: {
    id: 'privacy',
    path: '/privacy',
    title: 'Privacy Policy',
    description: 'How we protect your health data',
    component: 'PrivacyPage',
    showInNav: true,
  },
  help: {
    id: 'help',
    path: '/help',
    title: 'Help & Support',
    description: 'Get help with using NeuroLens-X',
    component: 'HelpPage',
    showInNav: true,
  },
};

export const ASSESSMENT_JOURNEY: UserJourney = {
  id: 'neurological-assessment',
  name: 'Neurological Risk Assessment',
  description:
    'Complete multi-modal assessment to evaluate neurological health risk',
  totalEstimatedDuration: 495, // 8 minutes 15 seconds
  steps: [
    {
      id: 'intro',
      title: 'Assessment Introduction',
      description: 'Learn about the assessment process and provide consent',
      route: 'assessment-intro',
      required: true,
      estimatedDuration: 60,
      validationRules: [
        {
          id: 'consent',
          field: 'consent',
          type: 'required',
          message: 'You must provide consent to continue with the assessment',
        },
      ],
    },
    {
      id: 'speech',
      title: 'Speech Analysis',
      description: 'Record speech sample for voice biomarker analysis',
      route: 'assessment-speech',
      required: true,
      estimatedDuration: 120,
      dependencies: ['intro'],
      validationRules: [
        {
          id: 'audio-recording',
          field: 'audioFile',
          type: 'required',
          message: 'Please record an audio sample to continue',
        },
        {
          id: 'audio-duration',
          field: 'audioFile',
          type: 'custom',
          message: 'Audio recording must be at least 30 seconds long',
          validator: (file: File) => file && file.size > 0,
        },
      ],
    },
    {
      id: 'retinal',
      title: 'Retinal Imaging',
      description: 'Upload retinal image for vascular pattern analysis',
      route: 'assessment-retinal',
      required: true,
      estimatedDuration: 90,
      dependencies: ['speech'],
      validationRules: [
        {
          id: 'retinal-image',
          field: 'imageFile',
          type: 'required',
          message: 'Please upload a retinal image or use a demo image',
        },
        {
          id: 'image-format',
          field: 'imageFile',
          type: 'format',
          message: 'Image must be in JPG, PNG, or TIFF format',
        },
      ],
      exitPoints: [
        {
          id: 'no-retinal-image',
          condition: 'user has no retinal image available',
          targetRoute: 'assessment-risk',
          message:
            'You can continue without retinal imaging, but results may be less comprehensive',
        },
      ],
    },
    {
      id: 'risk',
      title: 'Risk Assessment',
      description: 'Complete health and lifestyle questionnaire',
      route: 'assessment-risk',
      required: true,
      estimatedDuration: 180,
      dependencies: ['retinal'],
      validationRules: [
        {
          id: 'age',
          field: 'age',
          type: 'required',
          message: 'Please provide your age',
        },
        {
          id: 'medical-history',
          field: 'medicalHistory',
          type: 'required',
          message: 'Please complete the medical history section',
        },
      ],
    },
    {
      id: 'processing',
      title: 'Processing Results',
      description: 'AI analysis of your assessment data',
      route: 'assessment-processing',
      required: true,
      estimatedDuration: 45,
      dependencies: ['risk'],
    },
  ],
};

export const MAIN_NAVIGATION: NavigationItem[] = [
  {
    id: 'home',
    label: 'Home',
    route: 'home',
    icon: 'home',
  },
  {
    id: 'assessment',
    label: 'Assessment',
    route: 'assessment-intro',
    icon: 'brain',
  },
  {
    id: 'dashboard',
    label: 'Dashboard',
    route: 'dashboard',
    icon: 'dashboard',
    disabled: true, // Enabled for registered users
  },
  {
    id: 'help',
    label: 'Help',
    route: 'help',
    icon: 'help',
  },
];

/* ===== HELPER FUNCTIONS ===== */

export const getNextStep = (
  currentStepId: string,
  journey: UserJourney
): UserFlowStep | null => {
  const currentIndex = journey.steps.findIndex(
    (step) => step.id === currentStepId
  );
  if (currentIndex === -1 || currentIndex === journey.steps.length - 1) {
    return null;
  }
  return journey.steps[currentIndex + 1] || null;
};

export const getPreviousStep = (
  currentStepId: string,
  journey: UserJourney
): UserFlowStep | null => {
  const currentIndex = journey.steps.findIndex(
    (step) => step.id === currentStepId
  );
  if (currentIndex <= 0) {
    return null;
  }
  return journey.steps[currentIndex - 1] || null;
};

export const calculateProgress = (
  completedSteps: string[],
  journey: UserJourney
): ProgressState => {
  const totalSteps = journey.steps.length;
  const completed = completedSteps.length;
  const percentage = Math.round((completed / totalSteps) * 100);

  return {
    currentStep: completed + 1,
    totalSteps,
    completedSteps: completed,
    percentage,
    timeElapsed: 0, // To be calculated based on actual timing
    estimatedTimeRemaining: journey.totalEstimatedDuration,
    milestones: journey.steps.map((step) => ({
      id: step.id,
      label: step.title,
      completed: completedSteps.includes(step.id),
    })),
  };
};

export const validateStep = (
  stepId: string,
  data: any,
  journey: UserJourney
): ValidationRule[] => {
  const step = journey.steps.find((s) => s.id === stepId);
  if (!step || !step.validationRules) {
    return [];
  }

  return step.validationRules.filter((rule) => {
    if (rule.type === 'required') {
      return !data[rule.field];
    }
    if (rule.type === 'custom' && rule.validator) {
      return !rule.validator(data[rule.field]);
    }
    return false;
  });
};
