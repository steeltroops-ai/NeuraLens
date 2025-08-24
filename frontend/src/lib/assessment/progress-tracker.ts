/**
 * Real-time Progress Tracking System
 * WebSocket-based progress updates with persistence and recovery
 */

import { AssessmentProgress, AssessmentStep } from './workflow';

// Progress event types
export type ProgressEventType =
  | 'progress_update'
  | 'step_started'
  | 'step_completed'
  | 'step_failed'
  | 'assessment_completed'
  | 'assessment_failed'
  | 'connection_status';

// Progress event interface
export interface ProgressEvent {
  type: ProgressEventType;
  sessionId: string;
  timestamp: string;
  data: any;
}

// Connection status
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// Progress tracker configuration
export interface ProgressTrackerConfig {
  sessionId: string;
  websocketUrl?: string;
  enablePersistence?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

// Progress persistence interface
interface PersistedProgress {
  sessionId: string;
  progress: AssessmentProgress;
  timestamp: string;
  events: ProgressEvent[];
}

// Progress tracker class
export class ProgressTracker {
  private sessionId: string;
  private websocket: WebSocket | null = null;
  private connectionStatus: ConnectionStatus = 'disconnected';
  private progress: AssessmentProgress;
  private events: ProgressEvent[] = [];
  private listeners: Map<ProgressEventType, Set<(event: ProgressEvent) => void>> = new Map();
  private config: Required<ProgressTrackerConfig>;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;

  constructor(config: ProgressTrackerConfig) {
    this.sessionId = config.sessionId;
    this.config = {
      sessionId: config.sessionId,
      websocketUrl: config.websocketUrl || `ws://localhost:8000/ws/progress/${config.sessionId}`,
      enablePersistence: config.enablePersistence ?? true,
      reconnectAttempts: config.reconnectAttempts ?? 5,
      reconnectDelay: config.reconnectDelay ?? 2000,
    };

    this.progress = {
      currentStep: 'upload',
      completedSteps: [],
      totalSteps: 8,
      progressPercentage: 0,
      stepProgress: {},
      errors: {},
    };

    // Load persisted progress if available
    if (this.config.enablePersistence) {
      this.loadPersistedProgress();
    }
  }

  /**
   * Start progress tracking
   */
  async start(): Promise<void> {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      await this.connect();
    } catch (error) {
      console.warn('WebSocket connection failed, falling back to polling:', error);
      this.startPolling();
    }
  }

  /**
   * Stop progress tracking
   */
  stop(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    this.updateConnectionStatus('disconnected');
  }

  /**
   * Connect to WebSocket
   */
  private async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.updateConnectionStatus('connecting');
        this.websocket = new WebSocket(this.config.websocketUrl);

        this.websocket.onopen = () => {
          this.updateConnectionStatus('connected');
          this.reconnectAttempts = 0;

          // Send initial progress request
          this.sendMessage({
            type: 'get_progress',
            sessionId: this.sessionId,
          });

          resolve();
        };

        this.websocket.onmessage = event => {
          try {
            const progressEvent: ProgressEvent = JSON.parse(event.data);
            this.handleProgressEvent(progressEvent);
          } catch (error) {
            console.error('Failed to parse progress event:', error);
          }
        };

        this.websocket.onclose = () => {
          this.updateConnectionStatus('disconnected');
          this.attemptReconnect();
        };

        this.websocket.onerror = error => {
          this.updateConnectionStatus('error');
          reject(error);
        };

        // Connection timeout
        setTimeout(() => {
          if (this.websocket?.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket connection timeout'));
          }
        }, 5000);
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Attempt to reconnect
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.config.reconnectAttempts) {
      console.warn('Max reconnection attempts reached, falling back to polling');
      this.startPolling();
      return;
    }

    this.reconnectAttempts++;
    const delay = this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        console.warn(`Reconnection attempt ${this.reconnectAttempts} failed:`, error);
      }
    }, delay);
  }

  /**
   * Start polling fallback
   */
  private startPolling(): void {
    const pollInterval = setInterval(async () => {
      try {
        // Simulate progress polling (in real implementation, this would call an API)
        const mockProgress = this.generateMockProgress();
        this.updateProgress(mockProgress);
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000);

    // Store interval for cleanup
    (this as any).pollInterval = pollInterval;
  }

  /**
   * Send message via WebSocket
   */
  private sendMessage(message: any): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    }
  }

  /**
   * Handle progress event
   */
  private handleProgressEvent(event: ProgressEvent): void {
    this.events.push(event);

    switch (event.type) {
      case 'progress_update':
        this.updateProgress(event.data);
        break;
      case 'step_started':
        this.handleStepStarted(event.data);
        break;
      case 'step_completed':
        this.handleStepCompleted(event.data);
        break;
      case 'step_failed':
        this.handleStepFailed(event.data);
        break;
      case 'assessment_completed':
        this.handleAssessmentCompleted(event.data);
        break;
      case 'assessment_failed':
        this.handleAssessmentFailed(event.data);
        break;
    }

    // Persist progress
    if (this.config.enablePersistence) {
      this.persistProgress();
    }

    // Notify listeners
    this.notifyListeners(event.type, event);
  }

  /**
   * Update progress state
   */
  private updateProgress(newProgress: Partial<AssessmentProgress>): void {
    this.progress = { ...this.progress, ...newProgress };

    const event: ProgressEvent = {
      type: 'progress_update',
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      data: this.progress,
    };

    this.notifyListeners('progress_update', event);
  }

  /**
   * Handle step started
   */
  private handleStepStarted(data: { step: AssessmentStep }): void {
    this.progress.currentStep = data.step;
    this.progress.stepProgress[data.step] = 0;
  }

  /**
   * Handle step completed
   */
  private handleStepCompleted(data: { step: AssessmentStep }): void {
    if (!this.progress.completedSteps.includes(data.step)) {
      this.progress.completedSteps.push(data.step);
    }
    this.progress.stepProgress[data.step] = 100;

    // Update overall progress
    this.progress.progressPercentage =
      (this.progress.completedSteps.length / this.progress.totalSteps) * 100;
  }

  /**
   * Handle step failed
   */
  private handleStepFailed(data: { step: AssessmentStep; error: string }): void {
    this.progress.errors[data.step] = data.error;
  }

  /**
   * Handle assessment completed
   */
  private handleAssessmentCompleted(data: any): void {
    this.progress.progressPercentage = 100;
    this.progress.currentStep = 'complete';
  }

  /**
   * Handle assessment failed
   */
  private handleAssessmentFailed(data: { error: string }): void {
    this.progress.errors.general = data.error;
  }

  /**
   * Update connection status
   */
  private updateConnectionStatus(status: ConnectionStatus): void {
    this.connectionStatus = status;

    const event: ProgressEvent = {
      type: 'connection_status',
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      data: { status },
    };

    this.notifyListeners('connection_status', event);
  }

  /**
   * Notify event listeners
   */
  private notifyListeners(eventType: ProgressEventType, event: ProgressEvent): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(event);
        } catch (error) {
          console.error('Progress listener error:', error);
        }
      });
    }
  }

  /**
   * Add event listener
   */
  addEventListener(eventType: ProgressEventType, listener: (event: ProgressEvent) => void): void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(listener);
  }

  /**
   * Remove event listener
   */
  removeEventListener(
    eventType: ProgressEventType,
    listener: (event: ProgressEvent) => void,
  ): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.delete(listener);
    }
  }

  /**
   * Persist progress to localStorage
   */
  private persistProgress(): void {
    try {
      const persistedData: PersistedProgress = {
        sessionId: this.sessionId,
        progress: this.progress,
        timestamp: new Date().toISOString(),
        events: this.events.slice(-50), // Keep last 50 events
      };

      localStorage.setItem(`neuralens_progress_${this.sessionId}`, JSON.stringify(persistedData));
    } catch (error) {
      console.warn('Failed to persist progress:', error);
    }
  }

  /**
   * Load persisted progress
   */
  private loadPersistedProgress(): void {
    try {
      const stored = localStorage.getItem(`neuralens_progress_${this.sessionId}`);
      if (stored) {
        const persistedData: PersistedProgress = JSON.parse(stored);
        this.progress = persistedData.progress;
        this.events = persistedData.events || [];
      }
    } catch (error) {
      console.warn('Failed to load persisted progress:', error);
    }
  }

  /**
   * Clear persisted progress
   */
  clearPersistedProgress(): void {
    try {
      localStorage.removeItem(`neuralens_progress_${this.sessionId}`);
    } catch (error) {
      console.warn('Failed to clear persisted progress:', error);
    }
  }

  /**
   * Generate mock progress for polling fallback
   */
  private generateMockProgress(): Partial<AssessmentProgress> {
    const now = Date.now();
    const elapsed = now - (this.progress as any).startTime || 0;
    const mockProgress = Math.min(100, (elapsed / 30000) * 100); // 30 second mock duration

    return {
      progressPercentage: mockProgress,
      estimatedTimeRemaining: Math.max(0, 30 - Math.floor(elapsed / 1000)),
    };
  }

  /**
   * Get current progress
   */
  getProgress(): AssessmentProgress {
    return { ...this.progress };
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): ConnectionStatus {
    return this.connectionStatus;
  }

  /**
   * Get event history
   */
  getEventHistory(): ProgressEvent[] {
    return [...this.events];
  }
}
