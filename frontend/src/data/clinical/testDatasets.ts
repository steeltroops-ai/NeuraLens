/**
 * Clinical Test Datasets
 * Medically accurate test data representing diverse neurological conditions and risk profiles
 */

// Clinical dataset interfaces
export interface ClinicalTestDataset {
  id: string;
  name: string;
  description: string;
  riskCategory: 'low' | 'moderate' | 'high';
  clinicalContext: string;
  speechData: SpeechTestData;
  retinalData: RetinalTestData;
  motorData: MotorTestData;
  cognitiveData: CognitiveTestData;
  expectedOutcomes: ExpectedOutcomes;
}

export interface SpeechTestData {
  audioFile: string;
  duration: number;
  sampleRate: number;
  biomarkers: {
    fluency_score: number;
    voice_tremor: number;
    articulation_clarity: number;
    speech_rate: number;
    pause_patterns: number;
    vocal_stability: number;
    prosody_variation: number;
  };
  clinicalNotes: string;
}

export interface RetinalTestData {
  imageFile: string;
  imageQuality: number;
  biomarkers: {
    vessel_density: number;
    cup_disc_ratio: number;
    av_ratio: number;
    tortuosity_index: number;
    hemorrhage_count: number;
    exudate_presence: number;
    optic_disc_pallor: number;
    macular_thickness: number;
  };
  clinicalNotes: string;
}

export interface MotorTestData {
  accelerometerData: Array<{
    timestamp: number;
    x: number;
    y: number;
    z: number;
  }>;
  biomarkers: {
    coordination_index: number;
    tremor_severity: number;
    bradykinesia_score: number;
    rigidity_index: number;
    postural_stability: number;
    finger_tapping_rhythm: number;
    hand_movement_amplitude: number;
  };
  clinicalNotes: string;
}

export interface CognitiveTestData {
  testResults: {
    response_times: number[];
    accuracy_scores: number[];
    attention_tasks: number[];
    memory_tasks: number[];
    executive_tasks: number[];
  };
  biomarkers: {
    memory_score: number;
    attention_score: number;
    processing_speed: number;
    executive_function: number;
    working_memory: number;
    verbal_fluency: number;
    visuospatial_ability: number;
  };
  clinicalNotes: string;
}

export interface ExpectedOutcomes {
  nri_score: number;
  confidence: number;
  risk_category: 'low' | 'moderate' | 'high';
  primary_concerns: string[];
  recommended_actions: string[];
  follow_up_timeline: string;
}

// Clinical test datasets
export const CLINICAL_TEST_DATASETS: ClinicalTestDataset[] = [
  {
    id: 'early_parkinson_high_risk',
    name: "Early Parkinson's Disease - High Risk",
    description:
      "Patient showing early signs of Parkinson's disease with multiple biomarker abnormalities",
    riskCategory: 'high',
    clinicalContext: "Recent onset of subtle motor symptoms, family history of Parkinson's disease",
    speechData: {
      audioFile: 'speech_samples/early_parkinson_high.wav',
      duration: 45.2,
      sampleRate: 44100,
      biomarkers: {
        fluency_score: 0.62,
        voice_tremor: 0.78,
        articulation_clarity: 0.71,
        speech_rate: 0.68,
        pause_patterns: 0.82,
        vocal_stability: 0.59,
        prosody_variation: 0.64,
      },
      clinicalNotes:
        'Notable voice tremor and reduced vocal stability. Increased pause frequency suggests early speech motor control changes.',
    },
    retinalData: {
      imageFile: 'retinal_images/early_parkinson_high.jpg',
      imageQuality: 0.91,
      biomarkers: {
        vessel_density: 0.58,
        cup_disc_ratio: 0.42,
        av_ratio: 0.61,
        tortuosity_index: 0.73,
        hemorrhage_count: 0,
        exudate_presence: 0.12,
        optic_disc_pallor: 0.28,
        macular_thickness: 0.89,
      },
      clinicalNotes:
        'Reduced vessel density and increased tortuosity consistent with early neurodegeneration. No acute pathology.',
    },
    motorData: {
      accelerometerData: generateMotorData('high_tremor'),
      biomarkers: {
        coordination_index: 0.54,
        tremor_severity: 0.81,
        bradykinesia_score: 0.72,
        rigidity_index: 0.67,
        postural_stability: 0.61,
        finger_tapping_rhythm: 0.58,
        hand_movement_amplitude: 0.63,
      },
      clinicalNotes:
        "Significant tremor at rest, mild bradykinesia, and reduced coordination. Classic early Parkinson's motor pattern.",
    },
    cognitiveData: {
      testResults: {
        response_times: generateResponseTimes('mild_impairment'),
        accuracy_scores: generateAccuracyScores('mild_impairment'),
        attention_tasks: generateTaskScores('mild_impairment'),
        memory_tasks: generateTaskScores('mild_impairment'),
        executive_tasks: generateTaskScores('moderate_impairment'),
      },
      biomarkers: {
        memory_score: 0.74,
        attention_score: 0.71,
        processing_speed: 0.68,
        executive_function: 0.62,
        working_memory: 0.69,
        verbal_fluency: 0.66,
        visuospatial_ability: 0.73,
      },
      clinicalNotes:
        'Mild executive function deficits and processing speed reduction. Memory and attention relatively preserved.',
    },
    expectedOutcomes: {
      nri_score: 0.74,
      confidence: 0.87,
      risk_category: 'high',
      primary_concerns: [
        "Early Parkinson's disease markers detected",
        'Significant motor symptoms present',
        'Mild cognitive changes observed',
      ],
      recommended_actions: [
        'Urgent neurologist consultation within 2 weeks',
        'Consider DaTscan imaging',
        'Begin symptom monitoring diary',
      ],
      follow_up_timeline: '2 weeks',
    },
  },

  {
    id: 'moderate_cognitive_decline',
    name: 'Moderate Cognitive Decline - Moderate Risk',
    description: 'Patient with mild cognitive impairment and moderate neurological risk factors',
    riskCategory: 'moderate',
    clinicalContext:
      'Subjective cognitive complaints, age 68, hypertension, family history of dementia',
    speechData: {
      audioFile: 'speech_samples/moderate_cognitive.wav',
      duration: 52.8,
      sampleRate: 44100,
      biomarkers: {
        fluency_score: 0.78,
        voice_tremor: 0.34,
        articulation_clarity: 0.82,
        speech_rate: 0.76,
        pause_patterns: 0.68,
        vocal_stability: 0.81,
        prosody_variation: 0.74,
      },
      clinicalNotes:
        'Mild word-finding difficulties and increased pause patterns. Voice quality generally preserved.',
    },
    retinalData: {
      imageFile: 'retinal_images/moderate_cognitive.jpg',
      imageQuality: 0.88,
      biomarkers: {
        vessel_density: 0.71,
        cup_disc_ratio: 0.38,
        av_ratio: 0.72,
        tortuosity_index: 0.56,
        hemorrhage_count: 1,
        exudate_presence: 0.08,
        optic_disc_pallor: 0.22,
        macular_thickness: 0.92,
      },
      clinicalNotes:
        'Mild vascular changes consistent with age and hypertension. Single microhemorrhage noted.',
    },
    motorData: {
      accelerometerData: generateMotorData('mild_impairment'),
      biomarkers: {
        coordination_index: 0.72,
        tremor_severity: 0.28,
        bradykinesia_score: 0.41,
        rigidity_index: 0.35,
        postural_stability: 0.68,
        finger_tapping_rhythm: 0.74,
        hand_movement_amplitude: 0.79,
      },
      clinicalNotes:
        'Mild coordination changes and subtle bradykinesia. No significant tremor or rigidity.',
    },
    cognitiveData: {
      testResults: {
        response_times: generateResponseTimes('mild_impairment'),
        accuracy_scores: generateAccuracyScores('mild_impairment'),
        attention_tasks: generateTaskScores('mild_impairment'),
        memory_tasks: generateTaskScores('moderate_impairment'),
        executive_tasks: generateTaskScores('mild_impairment'),
      },
      biomarkers: {
        memory_score: 0.61,
        attention_score: 0.74,
        processing_speed: 0.72,
        executive_function: 0.68,
        working_memory: 0.64,
        verbal_fluency: 0.69,
        visuospatial_ability: 0.76,
      },
      clinicalNotes:
        'Memory performance below expected for age and education. Other cognitive domains relatively preserved.',
    },
    expectedOutcomes: {
      nri_score: 0.52,
      confidence: 0.79,
      risk_category: 'moderate',
      primary_concerns: [
        'Mild cognitive impairment detected',
        'Memory performance concerns',
        'Vascular risk factors present',
      ],
      recommended_actions: [
        'Neuropsychological evaluation within 6 weeks',
        'Cardiovascular risk assessment',
        'Cognitive training program consideration',
      ],
      follow_up_timeline: '6 weeks',
    },
  },

  {
    id: 'healthy_aging_low_risk',
    name: 'Healthy Aging - Low Risk',
    description: 'Healthy older adult with normal aging changes and low neurological risk',
    riskCategory: 'low',
    clinicalContext:
      'Annual wellness check, age 72, active lifestyle, no significant medical history',
    speechData: {
      audioFile: 'speech_samples/healthy_aging.wav',
      duration: 38.5,
      sampleRate: 44100,
      biomarkers: {
        fluency_score: 0.89,
        voice_tremor: 0.18,
        articulation_clarity: 0.91,
        speech_rate: 0.85,
        pause_patterns: 0.42,
        vocal_stability: 0.88,
        prosody_variation: 0.86,
      },
      clinicalNotes:
        'Excellent speech quality with minimal age-related changes. Clear articulation and normal prosody.',
    },
    retinalData: {
      imageFile: 'retinal_images/healthy_aging.jpg',
      imageQuality: 0.94,
      biomarkers: {
        vessel_density: 0.84,
        cup_disc_ratio: 0.32,
        av_ratio: 0.78,
        tortuosity_index: 0.41,
        hemorrhage_count: 0,
        exudate_presence: 0.02,
        optic_disc_pallor: 0.15,
        macular_thickness: 0.96,
      },
      clinicalNotes:
        'Healthy retinal appearance with age-appropriate changes. No pathological findings.',
    },
    motorData: {
      accelerometerData: generateMotorData('normal'),
      biomarkers: {
        coordination_index: 0.86,
        tremor_severity: 0.15,
        bradykinesia_score: 0.22,
        rigidity_index: 0.18,
        postural_stability: 0.82,
        finger_tapping_rhythm: 0.88,
        hand_movement_amplitude: 0.91,
      },
      clinicalNotes:
        'Excellent motor function for age. Minimal tremor and good coordination throughout testing.',
    },
    cognitiveData: {
      testResults: {
        response_times: generateResponseTimes('normal'),
        accuracy_scores: generateAccuracyScores('normal'),
        attention_tasks: generateTaskScores('normal'),
        memory_tasks: generateTaskScores('normal'),
        executive_tasks: generateTaskScores('normal'),
      },
      biomarkers: {
        memory_score: 0.84,
        attention_score: 0.87,
        processing_speed: 0.81,
        executive_function: 0.83,
        working_memory: 0.86,
        verbal_fluency: 0.89,
        visuospatial_ability: 0.85,
      },
      clinicalNotes:
        'Cognitive performance within normal limits for age and education. No concerns identified.',
    },
    expectedOutcomes: {
      nri_score: 0.23,
      confidence: 0.91,
      risk_category: 'low',
      primary_concerns: ['Normal aging changes observed', 'No significant neurological concerns'],
      recommended_actions: [
        'Continue current healthy lifestyle',
        'Annual wellness monitoring',
        'Maintain physical and cognitive activity',
      ],
      follow_up_timeline: '12 months',
    },
  },

  {
    id: 'vascular_risk_moderate',
    name: 'Vascular Risk Factors - Moderate Risk',
    description:
      'Patient with diabetes and hypertension showing vascular-related neurological changes',
    riskCategory: 'moderate',
    clinicalContext: 'Type 2 diabetes (10 years), hypertension, mild stroke history, age 65',
    speechData: {
      audioFile: 'speech_samples/vascular_risk.wav',
      duration: 41.7,
      sampleRate: 44100,
      biomarkers: {
        fluency_score: 0.73,
        voice_tremor: 0.31,
        articulation_clarity: 0.79,
        speech_rate: 0.71,
        pause_patterns: 0.58,
        vocal_stability: 0.76,
        prosody_variation: 0.68,
      },
      clinicalNotes:
        'Mild dysarthria and reduced speech rate consistent with vascular changes. Articulation mildly affected.',
    },
    retinalData: {
      imageFile: 'retinal_images/vascular_risk.jpg',
      imageQuality: 0.86,
      biomarkers: {
        vessel_density: 0.64,
        cup_disc_ratio: 0.41,
        av_ratio: 0.58,
        tortuosity_index: 0.69,
        hemorrhage_count: 3,
        exudate_presence: 0.24,
        optic_disc_pallor: 0.31,
        macular_thickness: 0.87,
      },
      clinicalNotes:
        'Diabetic retinopathy changes with multiple hemorrhages and exudates. Vascular narrowing present.',
    },
    motorData: {
      accelerometerData: generateMotorData('vascular_impairment'),
      biomarkers: {
        coordination_index: 0.68,
        tremor_severity: 0.42,
        bradykinesia_score: 0.51,
        rigidity_index: 0.38,
        postural_stability: 0.59,
        finger_tapping_rhythm: 0.66,
        hand_movement_amplitude: 0.71,
      },
      clinicalNotes:
        'Mild coordination deficits and postural instability. Consistent with vascular etiology.',
    },
    cognitiveData: {
      testResults: {
        response_times: generateResponseTimes('vascular_impairment'),
        accuracy_scores: generateAccuracyScores('mild_impairment'),
        attention_tasks: generateTaskScores('mild_impairment'),
        memory_tasks: generateTaskScores('mild_impairment'),
        executive_tasks: generateTaskScores('moderate_impairment'),
      },
      biomarkers: {
        memory_score: 0.71,
        attention_score: 0.68,
        processing_speed: 0.59,
        executive_function: 0.61,
        working_memory: 0.66,
        verbal_fluency: 0.64,
        visuospatial_ability: 0.69,
      },
      clinicalNotes:
        'Processing speed and executive function deficits consistent with vascular cognitive impairment.',
    },
    expectedOutcomes: {
      nri_score: 0.58,
      confidence: 0.83,
      risk_category: 'moderate',
      primary_concerns: [
        'Vascular cognitive impairment',
        'Diabetic complications affecting brain',
        'Increased stroke risk',
      ],
      recommended_actions: [
        'Optimize diabetes and blood pressure control',
        'Neurologist consultation within 4 weeks',
        'Cardiovascular risk reduction strategies',
      ],
      follow_up_timeline: '4 weeks',
    },
  },
];

// Helper functions for generating realistic data patterns
function generateMotorData(
  impairmentLevel: string,
): Array<{ timestamp: number; x: number; y: number; z: number }> {
  const data = [];
  const baseFreq =
    impairmentLevel === 'high_tremor'
      ? 5.2
      : impairmentLevel === 'mild_impairment'
        ? 1.8
        : impairmentLevel === 'vascular_impairment'
          ? 2.1
          : 0.8;

  const amplitude =
    impairmentLevel === 'high_tremor'
      ? 0.15
      : impairmentLevel === 'mild_impairment'
        ? 0.08
        : impairmentLevel === 'vascular_impairment'
          ? 0.12
          : 0.03;

  for (let i = 0; i < 1000; i++) {
    const t = i * 0.01;
    data.push({
      timestamp: t,
      x: Math.sin(2 * Math.PI * baseFreq * t) * amplitude + (Math.random() - 0.5) * 0.02,
      y: Math.cos(2 * Math.PI * baseFreq * t) * amplitude + (Math.random() - 0.5) * 0.02,
      z:
        9.8 +
        Math.sin(2 * Math.PI * baseFreq * 0.7 * t) * amplitude * 0.5 +
        (Math.random() - 0.5) * 0.01,
    });
  }
  return data;
}

function generateResponseTimes(impairmentLevel: string): number[] {
  const baseTimes =
    impairmentLevel === 'normal'
      ? 850
      : impairmentLevel === 'mild_impairment'
        ? 1150
        : impairmentLevel === 'moderate_impairment'
          ? 1450
          : impairmentLevel === 'vascular_impairment'
            ? 1350
            : 1000;

  const variance =
    impairmentLevel === 'normal'
      ? 150
      : impairmentLevel === 'mild_impairment'
        ? 250
        : impairmentLevel === 'moderate_impairment'
          ? 350
          : impairmentLevel === 'vascular_impairment'
            ? 300
            : 200;

  return Array.from({ length: 50 }, () =>
    Math.max(400, baseTimes + (Math.random() - 0.5) * variance),
  );
}

function generateAccuracyScores(impairmentLevel: string): number[] {
  const baseAccuracy =
    impairmentLevel === 'normal'
      ? 0.92
      : impairmentLevel === 'mild_impairment'
        ? 0.84
        : impairmentLevel === 'moderate_impairment'
          ? 0.71
          : impairmentLevel === 'vascular_impairment'
            ? 0.78
            : 0.88;

  return Array.from({ length: 50 }, () =>
    Math.min(1.0, Math.max(0.0, baseAccuracy + (Math.random() - 0.5) * 0.2)),
  );
}

function generateTaskScores(impairmentLevel: string): number[] {
  const baseScore =
    impairmentLevel === 'normal'
      ? 0.88
      : impairmentLevel === 'mild_impairment'
        ? 0.76
        : impairmentLevel === 'moderate_impairment'
          ? 0.62
          : impairmentLevel === 'vascular_impairment'
            ? 0.69
            : 0.82;

  return Array.from({ length: 20 }, () =>
    Math.min(1.0, Math.max(0.0, baseScore + (Math.random() - 0.5) * 0.25)),
  );
}

// Export utility functions
export function getDatasetById(id: string): ClinicalTestDataset | undefined {
  return CLINICAL_TEST_DATASETS.find(dataset => dataset.id === id);
}

export function getDatasetsByRiskCategory(
  category: 'low' | 'moderate' | 'high',
): ClinicalTestDataset[] {
  return CLINICAL_TEST_DATASETS.filter(dataset => dataset.riskCategory === category);
}

export function getAllDatasets(): ClinicalTestDataset[] {
  return CLINICAL_TEST_DATASETS;
}
