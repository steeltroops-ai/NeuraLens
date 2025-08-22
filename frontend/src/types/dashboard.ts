export type AssessmentType =
  | 'overview'
  | 'speech'
  | 'retinal'
  | 'motor'
  | 'cognitive'
  | 'multimodal'
  | 'nri-fusion';

export interface DashboardState {
  activeAssessment: AssessmentType;
  isProcessing: boolean;
  lastUpdate: Date | null;
  systemStatus: 'healthy' | 'warning' | 'error';
}

export interface PerformanceMetrics {
  speechLatency: number;
  retinalLatency: number;
  motorLatency: number;
  cognitiveLatency: number;
  nriLatency: number;
  overallAccuracy: number;
}

export interface SidebarItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}
