/**
 * Demo Data Management System
 * Centralized system for managing demo scenarios, patient profiles, and clinical datasets
 */

import {
  AssessmentScenario,
  getAllScenarios,
  getScenarioById,
} from '@/data/demo/assessmentScenarios';
import { PatientProfile, getAllPatients, getPatientById } from '@/data/clinical/patientProfiles';
import { ClinicalTestDataset, getAllDatasets, getDatasetById } from '@/data/clinical/testDatasets';
import { AssessmentResults } from '@/lib/assessment/workflow';

// Demo session interface
export interface DemoSession {
  id: string;
  scenarioId: string;
  patientId: string;
  datasetId: string;
  startTime: string;
  currentStage: DemoStage;
  progress: number;
  isActive: boolean;
  settings: DemoSettings;
}

export interface DemoSettings {
  autoAdvance: boolean;
  showTechnicalDetails: boolean;
  enableAccessibilityDemo: boolean;
  playbackSpeed: number;
  audienceType: 'clinical' | 'technical' | 'general';
  language: string;
}

export type DemoStage =
  | 'introduction'
  | 'assessment_process'
  | 'results_review'
  | 'clinical_recommendations'
  | 'follow_up_planning'
  | 'conclusion';

// Demo data manager class
export class DemoDataManager {
  private static instance: DemoDataManager;
  private activeSessions: Map<string, DemoSession> = new Map();
  private scenarios: AssessmentScenario[] = [];
  private patients: PatientProfile[] = [];
  private datasets: ClinicalTestDataset[] = [];

  private constructor() {
    this.loadDemoData();
  }

  public static getInstance(): DemoDataManager {
    if (!DemoDataManager.instance) {
      DemoDataManager.instance = new DemoDataManager();
    }
    return DemoDataManager.instance;
  }

  /**
   * Load all demo data
   */
  private loadDemoData(): void {
    this.scenarios = getAllScenarios();
    this.patients = getAllPatients();
    this.datasets = getAllDatasets();
  }

  /**
   * Create a new demo session
   */
  public createDemoSession(scenarioId: string, settings: Partial<DemoSettings> = {}): DemoSession {
    const scenario = getScenarioById(scenarioId);
    if (!scenario) {
      throw new Error(`Scenario not found: ${scenarioId}`);
    }

    // Use deterministic session ID for SSR compatibility
    const sessionId =
      typeof window !== 'undefined'
        ? `demo_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
        : `demo_ssr_${Math.floor(Math.random() * 1000000)}`;

    const defaultSettings: DemoSettings = {
      autoAdvance: false,
      showTechnicalDetails: true,
      enableAccessibilityDemo: false,
      playbackSpeed: 1.0,
      audienceType: 'clinical',
      language: 'en',
    };

    const session: DemoSession = {
      id: sessionId,
      scenarioId: scenario.id,
      patientId: scenario.patientProfile,
      datasetId: scenario.testDataset,
      startTime: new Date().toISOString(),
      currentStage: 'introduction',
      progress: 0,
      isActive: true,
      settings: { ...defaultSettings, ...settings },
    };

    this.activeSessions.set(sessionId, session);
    return session;
  }

  /**
   * Get demo session by ID
   */
  public getDemoSession(sessionId: string): DemoSession | undefined {
    return this.activeSessions.get(sessionId);
  }

  /**
   * Update demo session
   */
  public updateDemoSession(sessionId: string, updates: Partial<DemoSession>): DemoSession {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Demo session not found: ${sessionId}`);
    }

    const updatedSession = { ...session, ...updates };
    this.activeSessions.set(sessionId, updatedSession);
    return updatedSession;
  }

  /**
   * Advance demo to next stage
   */
  public advanceDemoStage(sessionId: string): DemoSession {
    const session = this.getDemoSession(sessionId);
    if (!session) {
      throw new Error(`Demo session not found: ${sessionId}`);
    }

    const stages: DemoStage[] = [
      'introduction',
      'assessment_process',
      'results_review',
      'clinical_recommendations',
      'follow_up_planning',
      'conclusion',
    ];

    const currentIndex = stages.indexOf(session.currentStage);
    const nextIndex = Math.min(currentIndex + 1, stages.length - 1);
    const progress = ((nextIndex + 1) / stages.length) * 100;

    return this.updateDemoSession(sessionId, {
      currentStage: stages[nextIndex],
      progress,
      isActive: nextIndex < stages.length - 1,
    });
  }

  /**
   * Get complete demo data for session
   */
  public getDemoData(sessionId: string): {
    session: DemoSession;
    scenario: AssessmentScenario;
    patient: PatientProfile;
    dataset: ClinicalTestDataset;
    assessmentResults: AssessmentResults;
  } {
    const session = this.getDemoSession(sessionId);
    if (!session) {
      throw new Error(`Demo session not found: ${sessionId}`);
    }

    const scenario = getScenarioById(session.scenarioId);
    const patient = getPatientById(session.patientId);
    const dataset = getDatasetById(session.datasetId);

    if (!scenario || !patient || !dataset) {
      throw new Error('Demo data incomplete');
    }

    const assessmentResults = this.generateAssessmentResults(dataset, patient);

    return {
      session,
      scenario,
      patient,
      dataset,
      assessmentResults,
    };
  }

  /**
   * Generate assessment results from dataset
   */
  private generateAssessmentResults(
    dataset: ClinicalTestDataset,
    patient: PatientProfile,
  ): AssessmentResults {
    // Use deterministic timestamps for SSR compatibility
    const baseTime = typeof window !== 'undefined' ? Date.now() : 1703000000000; // Fixed timestamp for SSR
    const now = new Date(baseTime).toISOString();
    const startTime = new Date(baseTime - 45000).toISOString(); // 45 seconds ago

    return {
      sessionId: `demo_${dataset.id}`,
      completionTime: now,
      totalProcessingTime: 45000,
      overallRiskCategory: dataset.riskCategory,
      metadata: {
        startTime,
        endTime: now,
        stepsCompleted: [
          'upload',
          'validation',
          'speech_processing',
          'retinal_processing',
          'motor_processing',
          'cognitive_processing',
          'nri_fusion',
          'results',
        ],
        errors: [],
      },
      nriResult: {
        session_id: `demo_${dataset.id}`,
        nri_score: dataset.expectedOutcomes.nri_score,
        confidence: dataset.expectedOutcomes.confidence,
        risk_category: dataset.expectedOutcomes.risk_category,
        uncertainty: 0.1,
        consistency_score: 0.85,
        modality_contributions: [
          { modality: 'speech', weight: 0.3, confidence: 0.8, risk_score: 0.4 },
          { modality: 'retinal', weight: 0.25, confidence: 0.9, risk_score: 0.3 },
          { modality: 'motor', weight: 0.25, confidence: 0.85, risk_score: 0.5 },
          { modality: 'cognitive', weight: 0.2, confidence: 0.75, risk_score: 0.35 },
        ],
        processing_time: 5000,
        timestamp: new Date().toISOString(),
        recommendations: dataset.expectedOutcomes.recommended_actions,
        follow_up_actions: [dataset.expectedOutcomes.follow_up_timeline],
      },
      speechResult: {
        session_id: `demo_${dataset.id}`,
        risk_score: this.calculateModalityRisk(dataset.speechData.biomarkers),
        confidence: 0.85,
        processing_time: 12000,
        timestamp: new Date().toISOString(),
        recommendations: ['Monitor speech patterns', 'Consider speech therapy if needed'],
        quality_score: 0.92,
        biomarkers: {
          fluency_score: dataset.speechData.biomarkers.fluency_score,
          pause_pattern: dataset.speechData.biomarkers.pause_patterns,
          voice_tremor: dataset.speechData.biomarkers.voice_tremor,
          articulation_clarity: dataset.speechData.biomarkers.articulation_clarity,
          prosody_variation: dataset.speechData.biomarkers.prosody_variation,
          speaking_rate: dataset.speechData.biomarkers.speech_rate,
          pause_frequency: dataset.speechData.biomarkers.pause_patterns * 0.8, // Approximate mapping
        },
        file_info: {
          duration: dataset.speechData.duration,
          sample_rate: dataset.speechData.sampleRate,
          channels: 1,
        },
      },
      retinalResult: {
        session_id: `demo_${dataset.id}`,
        risk_score: this.calculateModalityRisk(dataset.retinalData.biomarkers),
        confidence: 0.88,
        processing_time: 18000,
        timestamp: new Date().toISOString(),
        recommendations: ['Monitor retinal health', 'Regular eye examinations recommended'],
        quality_score: dataset.retinalData.imageQuality,
        detected_conditions: ['Normal findings', 'No significant abnormalities'],
        biomarkers: {
          vessel_density: dataset.retinalData.biomarkers.vessel_density,
          vessel_tortuosity: dataset.retinalData.biomarkers.tortuosity_index,
          cup_disc_ratio: dataset.retinalData.biomarkers.cup_disc_ratio,
          av_ratio: dataset.retinalData.biomarkers.av_ratio,
          hemorrhage_count: dataset.retinalData.biomarkers.hemorrhage_count,
          exudate_area: dataset.retinalData.biomarkers.exudate_presence,
          microaneurysm_count: 0, // Default value since not in test data
        },
        image_info: {
          width: 1024,
          height: 1024,
          format: 'JPEG',
          file_size: 2048000, // 2MB
        },
      },
      motorResult: {
        session_id: `demo_${dataset.id}`,
        risk_score: this.calculateModalityRisk(dataset.motorData.biomarkers),
        confidence: 0.87,
        processing_time: 8000,
        timestamp: new Date().toISOString(),
        recommendations: ['Monitor motor function', 'Regular physical therapy recommended'],
        assessment_type: 'tremor',
        movement_quality: this.getMovementQuality(dataset.motorData.biomarkers),
        biomarkers: {
          movement_frequency: dataset.motorData.biomarkers.finger_tapping_rhythm,
          amplitude_variation: dataset.motorData.biomarkers.hand_movement_amplitude,
          coordination_index: dataset.motorData.biomarkers.coordination_index,
          tremor_severity: dataset.motorData.biomarkers.tremor_severity,
          fatigue_index: dataset.motorData.biomarkers.bradykinesia_score,
          asymmetry_score: dataset.motorData.biomarkers.rigidity_index,
        },
      },
      cognitiveResult: {
        session_id: `demo_${dataset.id}`,
        risk_score: this.calculateModalityRisk(dataset.cognitiveData.biomarkers),
        confidence: 0.83,
        processing_time: 7000,
        timestamp: new Date().toISOString(),
        recommendations: ['Monitor cognitive function', 'Consider cognitive training exercises'],
        overall_score: this.calculateOverallCognitiveScore(dataset.cognitiveData.biomarkers),
        test_battery: ['memory', 'attention', 'executive'],
        domain_scores: {
          memory: dataset.cognitiveData.biomarkers.memory_score,
          attention: dataset.cognitiveData.biomarkers.attention_score,
          executive: dataset.cognitiveData.biomarkers.executive_function,
          processing: dataset.cognitiveData.biomarkers.processing_speed,
        },
        biomarkers: {
          memory_score: dataset.cognitiveData.biomarkers.memory_score,
          attention_score: dataset.cognitiveData.biomarkers.attention_score,
          executive_score: dataset.cognitiveData.biomarkers.executive_function,
          language_score: dataset.cognitiveData.biomarkers.verbal_fluency,
          processing_speed: dataset.cognitiveData.biomarkers.processing_speed,
          cognitive_flexibility: dataset.cognitiveData.biomarkers.visuospatial_ability,
        },
      },
    };
  }

  /**
   * Calculate modality risk from biomarkers
   */
  private calculateModalityRisk(biomarkers: Record<string, number>): number {
    const values = Object.values(biomarkers);
    const riskValues = values.map(v => (v > 0.5 ? 1 - v : v));
    return riskValues.reduce((sum, val) => sum + val, 0) / riskValues.length;
  }

  /**
   * Get movement quality description
   */
  private getMovementQuality(biomarkers: Record<string, number>): string {
    const avgScore =
      Object.values(biomarkers).reduce((sum, val) => sum + val, 0) /
      Object.values(biomarkers).length;
    if (avgScore > 0.8) return 'excellent';
    if (avgScore > 0.6) return 'good';
    if (avgScore > 0.4) return 'fair';
    return 'poor';
  }

  /**
   * Calculate overall cognitive score
   */
  private calculateOverallCognitiveScore(biomarkers: Record<string, number>): number {
    return (
      Object.values(biomarkers).reduce((sum, val) => sum + val, 0) /
      Object.values(biomarkers).length
    );
  }

  /**
   * Get available scenarios
   */
  public getAvailableScenarios(): AssessmentScenario[] {
    return this.scenarios;
  }

  /**
   * Get scenarios by category
   */
  public getScenariosByCategory(category: string): AssessmentScenario[] {
    return this.scenarios.filter(
      scenario =>
        scenario.clinicalContext.toLowerCase().includes(category.toLowerCase()) ||
        scenario.description.toLowerCase().includes(category.toLowerCase()),
    );
  }

  /**
   * Get patient profiles
   */
  public getPatientProfiles(): PatientProfile[] {
    return this.patients;
  }

  /**
   * Get clinical datasets
   */
  public getClinicalDatasets(): ClinicalTestDataset[] {
    return this.datasets;
  }

  /**
   * Search scenarios by criteria
   */
  public searchScenarios(criteria: {
    riskCategory?: 'low' | 'moderate' | 'high';
    ageRange?: [number, number];
    condition?: string;
    hasAccessibilityFeatures?: boolean;
  }): AssessmentScenario[] {
    return this.scenarios.filter(scenario => {
      const patient = getPatientById(scenario.patientProfile);
      const dataset = getDatasetById(scenario.testDataset);

      if (!patient || !dataset) return false;

      if (criteria.riskCategory && dataset.riskCategory !== criteria.riskCategory) {
        return false;
      }

      if (criteria.ageRange) {
        const [minAge, maxAge] = criteria.ageRange;
        if (patient.demographics.age < minAge || patient.demographics.age > maxAge) {
          return false;
        }
      }

      if (criteria.condition) {
        const hasCondition = scenario.expectedOutcomes.primaryDiagnosis
          .toLowerCase()
          .includes(criteria.condition.toLowerCase());
        if (!hasCondition) return false;
      }

      if (criteria.hasAccessibilityFeatures && !scenario.accessibilityFeatures) {
        return false;
      }

      return true;
    });
  }

  /**
   * Get demo statistics
   */
  public getDemoStatistics(): {
    totalScenarios: number;
    totalPatients: number;
    totalDatasets: number;
    activeSessions: number;
    riskCategoryDistribution: Record<string, number>;
    ageDistribution: Record<string, number>;
  } {
    const riskCategoryDistribution = this.datasets.reduce(
      (acc, dataset) => {
        acc[dataset.riskCategory] = (acc[dataset.riskCategory] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );

    const ageDistribution = this.patients.reduce(
      (acc, patient) => {
        const ageGroup =
          patient.demographics.age < 50
            ? '<50'
            : patient.demographics.age < 65
              ? '50-64'
              : patient.demographics.age < 75
                ? '65-74'
                : '75+';
        acc[ageGroup] = (acc[ageGroup] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );

    return {
      totalScenarios: this.scenarios.length,
      totalPatients: this.patients.length,
      totalDatasets: this.datasets.length,
      activeSessions: this.activeSessions.size,
      riskCategoryDistribution,
      ageDistribution,
    };
  }

  /**
   * Export demo data for external use
   */
  public exportDemoData(format: 'json' | 'csv' = 'json'): string {
    const data = {
      scenarios: this.scenarios,
      patients: this.patients,
      datasets: this.datasets,
      statistics: this.getDemoStatistics(),
    };

    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else {
      // Simple CSV export for scenarios
      const csvHeaders = 'ID,Title,Risk Category,Patient Age,Primary Diagnosis';
      const csvRows = this.scenarios.map(scenario => {
        const patient = getPatientById(scenario.patientProfile);
        const dataset = getDatasetById(scenario.testDataset);
        return [
          scenario.id,
          scenario.title,
          dataset?.riskCategory || 'unknown',
          patient?.demographics.age || 'unknown',
          scenario.expectedOutcomes.primaryDiagnosis,
        ].join(',');
      });

      return [csvHeaders, ...csvRows].join('\n');
    }
  }

  /**
   * Reset demo session
   */
  public resetDemoSession(sessionId: string): DemoSession {
    return this.updateDemoSession(sessionId, {
      currentStage: 'introduction',
      progress: 0,
      isActive: true,
      startTime: new Date().toISOString(),
    });
  }

  /**
   * End demo session
   */
  public endDemoSession(sessionId: string): void {
    this.activeSessions.delete(sessionId);
  }

  /**
   * Get active sessions
   */
  public getActiveSessions(): DemoSession[] {
    return Array.from(this.activeSessions.values()).filter(session => session.isActive);
  }
}

// Export singleton instance
export const demoDataManager = DemoDataManager.getInstance();
