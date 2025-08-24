/**
 * Patient Profiles for Clinical Demonstration
 * Comprehensive patient profiles with longitudinal data and clinical outcomes
 */

// Patient profile interfaces
export interface PatientProfile {
  id: string;
  demographics: PatientDemographics;
  medicalHistory: MedicalHistory;
  currentStatus: CurrentStatus;
  assessmentHistory: AssessmentHistory[];
  clinicalOutcomes: ClinicalOutcomes;
  patientStory: PatientStory;
  accessibilityNeeds?: AccessibilityNeeds;
}

export interface PatientDemographics {
  age: number;
  gender: 'male' | 'female' | 'non-binary';
  ethnicity: string;
  education: string;
  occupation: string;
  location: string;
  insuranceType: string;
  primaryLanguage: string;
}

export interface MedicalHistory {
  primaryConditions: string[];
  currentMedications: string[];
  familyHistory: string[];
  surgicalHistory: string[];
  allergies: string[];
  riskFactors: string[];
  previousNeurologicalEvaluations: string[];
}

export interface CurrentStatus {
  chiefComplaint: string;
  symptomsOnset: string;
  functionalStatus: string;
  qualityOfLife: number; // 1-10 scale
  caregiverSupport: string;
  mobilityAids: string[];
}

export interface AssessmentHistory {
  date: string;
  sessionId: string;
  nriScore: number;
  riskCategory: 'low' | 'moderate' | 'high';
  speechScore: number;
  retinalScore: number;
  motorScore: number;
  cognitiveScore: number;
  clinicalNotes: string;
  followUpActions: string[];
  interventionsImplemented: string[];
}

export interface ClinicalOutcomes {
  diagnosisConfirmed?: string;
  timeToConfirmation?: string;
  treatmentResponse?: string;
  functionalImprovement?: string;
  qualityOfLifeChange?: number;
  caregiverSatisfaction?: number;
  costSavings?: string;
}

export interface PatientStory {
  background: string;
  initialConcerns: string;
  discoveryProcess: string;
  clinicalJourney: string;
  currentOutcome: string;
  patientQuote: string;
  familyImpact: string;
}

export interface AccessibilityNeeds {
  visualImpairment?: string;
  hearingImpairment?: string;
  motorImpairment?: string;
  cognitiveSupport?: string;
  languageSupport?: string;
  assistiveTechnology?: string[];
}

// Comprehensive patient profiles
export const PATIENT_PROFILES: PatientProfile[] = [
  {
    id: 'patient_001_margaret_chen',
    demographics: {
      age: 67,
      gender: 'female',
      ethnicity: 'Asian American',
      education: "Master's Degree",
      occupation: 'Retired Teacher',
      location: 'San Francisco, CA',
      insuranceType: 'Medicare + Supplemental',
      primaryLanguage: 'English',
    },
    medicalHistory: {
      primaryConditions: ['Hypertension', 'Osteoarthritis', 'Mild Depression'],
      currentMedications: ['Lisinopril 10mg', 'Ibuprofen 400mg PRN', 'Sertraline 50mg'],
      familyHistory: [
        "Father: Parkinson's Disease (age 72)",
        "Mother: Alzheimer's Disease (age 78)",
      ],
      surgicalHistory: ['Cataract Surgery (2019)', 'Knee Arthroscopy (2017)'],
      allergies: ['Penicillin', 'Shellfish'],
      riskFactors: ['Family history of neurodegeneration', 'Age', 'Mild depression'],
      previousNeurologicalEvaluations: ['None prior to NeuraLens assessment'],
    },
    currentStatus: {
      chiefComplaint: 'Subtle hand tremor and occasional word-finding difficulties',
      symptomsOnset: '6 months ago',
      functionalStatus: 'Independent in all activities of daily living',
      qualityOfLife: 7,
      caregiverSupport: 'Supportive spouse, adult children nearby',
      mobilityAids: [],
    },
    assessmentHistory: [
      {
        date: '2024-01-15',
        sessionId: 'session_001_baseline',
        nriScore: 0.72,
        riskCategory: 'high',
        speechScore: 0.68,
        retinalScore: 0.71,
        motorScore: 0.78,
        cognitiveScore: 0.74,
        clinicalNotes:
          'Initial assessment revealed elevated risk across multiple modalities. Speech tremor and motor abnormalities most prominent.',
        followUpActions: [
          'Neurologist referral within 2 weeks',
          'DaTscan scheduled',
          'Symptom diary initiated',
        ],
        interventionsImplemented: [],
      },
      {
        date: '2024-04-15',
        sessionId: 'session_001_followup_3mo',
        nriScore: 0.69,
        riskCategory: 'high',
        speechScore: 0.71,
        retinalScore: 0.69,
        motorScore: 0.74,
        cognitiveScore: 0.76,
        clinicalNotes:
          'Slight improvement in speech and cognitive scores following medication initiation. Motor symptoms stable.',
        followUpActions: [
          'Continue current treatment',
          'Physical therapy referral',
          'Next assessment in 3 months',
        ],
        interventionsImplemented: [
          'Carbidopa-Levodopa 25/100mg TID',
          'Physical therapy 2x/week',
          'Speech therapy 1x/week',
        ],
      },
      {
        date: '2024-07-15',
        sessionId: 'session_001_followup_6mo',
        nriScore: 0.58,
        riskCategory: 'moderate',
        speechScore: 0.76,
        retinalScore: 0.68,
        motorScore: 0.65,
        cognitiveScore: 0.79,
        clinicalNotes:
          'Significant improvement in motor and speech scores. Treatment response excellent. Risk category reduced to moderate.',
        followUpActions: [
          'Continue current regimen',
          'Gradual therapy reduction',
          'Monitor for medication side effects',
        ],
        interventionsImplemented: [
          'Medication optimization',
          'Continued therapy',
          'Exercise program',
        ],
      },
    ],
    clinicalOutcomes: {
      diagnosisConfirmed: "Early-stage Parkinson's Disease",
      timeToConfirmation: '3 weeks from initial NeuraLens assessment',
      treatmentResponse: 'Excellent response to dopaminergic therapy',
      functionalImprovement: 'Maintained independence, improved motor function',
      qualityOfLifeChange: 2, // Improved from 7 to 9
      caregiverSatisfaction: 9,
      costSavings: 'Avoided 6-month diagnostic delay, estimated $15,000 in unnecessary testing',
    },
    patientStory: {
      background:
        'Margaret is a retired elementary school teacher who noticed subtle changes in her handwriting and occasional tremor while gardening.',
      initialConcerns:
        'Worried about family history of neurological conditions but hesitant to seek medical attention for "minor" symptoms.',
      discoveryProcess:
        "NeuraLens assessment at annual wellness visit detected early Parkinson's markers before clinical diagnosis was obvious.",
      clinicalJourney:
        'Rapid referral to movement disorder specialist, confirmed diagnosis, early treatment initiation with excellent response.',
      currentOutcome:
        'Maintains active lifestyle with gardening, volunteering, and travel. Symptoms well-controlled with medication.',
      patientQuote:
        "I'm so grateful we caught this early. I was afraid I'd lose my independence, but with early treatment, I'm still doing everything I love.",
      familyImpact:
        'Family relieved to have answers and treatment plan. Spouse educated about condition and supportive of treatment regimen.',
    },
  },

  {
    id: 'patient_002_robert_johnson',
    demographics: {
      age: 72,
      gender: 'male',
      ethnicity: 'African American',
      education: 'High School',
      occupation: 'Retired Factory Worker',
      location: 'Detroit, MI',
      insuranceType: 'Medicare',
      primaryLanguage: 'English',
    },
    medicalHistory: {
      primaryConditions: ['Type 2 Diabetes', 'Hypertension', 'Chronic Kidney Disease Stage 3'],
      currentMedications: ['Metformin 1000mg BID', 'Amlodipine 10mg', 'Lisinopril 20mg'],
      familyHistory: ['Mother: Stroke (age 68)', 'Brother: Diabetes', 'Sister: Hypertension'],
      surgicalHistory: ['Appendectomy (1985)', 'Cataract Surgery (2020)'],
      allergies: ['NKDA'],
      riskFactors: ['Diabetes', 'Hypertension', 'Age', 'African American ethnicity'],
      previousNeurologicalEvaluations: ['None'],
    },
    currentStatus: {
      chiefComplaint: 'Memory problems and difficulty concentrating',
      symptomsOnset: '1 year ago, gradually worsening',
      functionalStatus: 'Needs assistance with medication management and finances',
      qualityOfLife: 5,
      caregiverSupport: 'Daughter provides daily support',
      mobilityAids: ['Walking cane'],
    },
    assessmentHistory: [
      {
        date: '2024-02-10',
        sessionId: 'session_002_baseline',
        nriScore: 0.61,
        riskCategory: 'moderate',
        speechScore: 0.58,
        retinalScore: 0.52,
        motorScore: 0.67,
        cognitiveScore: 0.48,
        clinicalNotes:
          'Moderate risk with prominent cognitive and retinal vascular changes. Consistent with vascular cognitive impairment.',
        followUpActions: [
          'Neuropsychological evaluation',
          'Optimize diabetes control',
          'Cardiovascular risk assessment',
        ],
        interventionsImplemented: [],
      },
      {
        date: '2024-05-10',
        sessionId: 'session_002_followup_3mo',
        nriScore: 0.55,
        riskCategory: 'moderate',
        speechScore: 0.62,
        retinalScore: 0.58,
        motorScore: 0.69,
        cognitiveScore: 0.54,
        clinicalNotes:
          'Improvement in all domains following diabetes optimization and cognitive training. Retinal changes stabilized.',
        followUpActions: [
          'Continue interventions',
          'Add cholinesterase inhibitor',
          'Family education',
        ],
        interventionsImplemented: [
          'Diabetes medication adjustment',
          'Cognitive training program',
          'Dietary counseling',
        ],
      },
    ],
    clinicalOutcomes: {
      diagnosisConfirmed: 'Vascular Cognitive Impairment',
      timeToConfirmation: '4 weeks from initial assessment',
      treatmentResponse: 'Moderate improvement with vascular risk factor management',
      functionalImprovement: 'Stabilized cognitive function, improved diabetes control',
      qualityOfLifeChange: 1, // Improved from 5 to 6
      caregiverSatisfaction: 7,
      costSavings: 'Early intervention prevented further cognitive decline',
    },
    patientStory: {
      background:
        'Robert worked in manufacturing for 40 years and has struggled with diabetes management since retirement.',
      initialConcerns:
        'Family noticed increasing forgetfulness and difficulty managing medications independently.',
      discoveryProcess:
        'NeuraLens assessment during routine diabetes visit identified vascular cognitive changes.',
      clinicalJourney:
        'Comprehensive vascular risk factor management and cognitive interventions implemented.',
      currentOutcome:
        'Cognitive function stabilized, better diabetes control, maintained in community with family support.',
      patientQuote:
        "I was scared I was getting dementia like my mother. Now I understand it's related to my diabetes, and we can do something about it.",
      familyImpact:
        "Daughter educated about vascular dementia prevention and actively supports father's care plan.",
    },
    accessibilityNeeds: {
      visualImpairment: 'Mild diabetic retinopathy, requires large print materials',
      languageSupport: 'Prefers simple, clear explanations',
      assistiveTechnology: ['Medication reminder system', 'Large button phone'],
    },
  },

  {
    id: 'patient_003_sarah_williams',
    demographics: {
      age: 45,
      gender: 'female',
      ethnicity: 'Caucasian',
      education: 'PhD',
      occupation: 'Research Scientist',
      location: 'Boston, MA',
      insuranceType: 'Private Insurance',
      primaryLanguage: 'English',
    },
    medicalHistory: {
      primaryConditions: ['Migraine', 'Anxiety'],
      currentMedications: ['Sumatriptan 50mg PRN', 'Propranolol 40mg BID'],
      familyHistory: ["Mother: Early-onset Alzheimer's (age 58)", 'Maternal grandmother: Dementia'],
      surgicalHistory: ['None'],
      allergies: ['Latex'],
      riskFactors: ['Strong family history of early-onset dementia', 'High-stress occupation'],
      previousNeurologicalEvaluations: ['Genetic counseling for APOE testing (declined)'],
    },
    currentStatus: {
      chiefComplaint: 'Concerned about subtle memory changes and family history',
      symptomsOnset: 'Subjective concerns over past 6 months',
      functionalStatus: 'Fully independent, high-functioning professional',
      qualityOfLife: 8,
      caregiverSupport: 'Supportive partner, no children',
      mobilityAids: [],
    },
    assessmentHistory: [
      {
        date: '2024-03-20',
        sessionId: 'session_003_baseline',
        nriScore: 0.28,
        riskCategory: 'low',
        speechScore: 0.91,
        retinalScore: 0.88,
        motorScore: 0.92,
        cognitiveScore: 0.85,
        clinicalNotes:
          'Low risk assessment with excellent performance across all modalities. Cognitive concerns likely related to anxiety and family history worry.',
        followUpActions: ['Reassurance and education', 'Stress management', 'Annual monitoring'],
        interventionsImplemented: [],
      },
      {
        date: '2024-09-20',
        sessionId: 'session_003_followup_6mo',
        nriScore: 0.25,
        riskCategory: 'low',
        speechScore: 0.93,
        retinalScore: 0.89,
        motorScore: 0.94,
        cognitiveScore: 0.88,
        clinicalNotes:
          'Continued low risk with slight improvement in cognitive scores following stress management interventions.',
        followUpActions: [
          'Continue stress management',
          'Next assessment in 12 months',
          'Consider genetic counseling if desired',
        ],
        interventionsImplemented: [
          'Mindfulness-based stress reduction',
          'Regular exercise program',
          'Therapy for anxiety',
        ],
      },
    ],
    clinicalOutcomes: {
      diagnosisConfirmed: 'No neurological pathology detected',
      treatmentResponse: 'Excellent response to anxiety management',
      functionalImprovement: 'Reduced anxiety about cognitive function',
      qualityOfLifeChange: 1, // Improved from 8 to 9
      caregiverSatisfaction: 9,
      costSavings: 'Avoided unnecessary neurological workup and imaging',
    },
    patientStory: {
      background:
        'Sarah is a successful research scientist with a strong family history of early-onset dementia.',
      initialConcerns:
        'Hypervigilant about any memory lapses due to family history, causing significant anxiety.',
      discoveryProcess:
        'NeuraLens assessment provided objective reassurance about cognitive function.',
      clinicalJourney:
        'Focus shifted from neurological concerns to anxiety management and stress reduction.',
      currentOutcome:
        'Reduced anxiety, improved work-life balance, continues high-level professional functioning.',
      patientQuote:
        'Having objective data about my brain health gave me peace of mind. I can focus on living my life instead of worrying about every forgotten word.',
      familyImpact:
        'Partner relieved to see reduced anxiety and return to normal activities and travel.',
    },
  },

  {
    id: 'patient_004_carlos_rodriguez',
    demographics: {
      age: 58,
      gender: 'male',
      ethnicity: 'Hispanic',
      education: 'Some College',
      occupation: 'Construction Foreman',
      location: 'Phoenix, AZ',
      insuranceType: 'Employer Insurance',
      primaryLanguage: 'Spanish (English as second language)',
    },
    medicalHistory: {
      primaryConditions: ['Traumatic Brain Injury (2018)', 'Chronic Pain', 'Sleep Apnea'],
      currentMedications: ['Gabapentin 300mg TID', 'CPAP therapy'],
      familyHistory: ['Father: Stroke (age 65)', 'Mother: Diabetes'],
      surgicalHistory: ['Craniotomy following TBI (2018)'],
      allergies: ['Codeine'],
      riskFactors: ['History of TBI', 'Occupational head trauma exposure', 'Sleep disorders'],
      previousNeurologicalEvaluations: ['Post-TBI neuropsychological testing (2019)'],
    },
    currentStatus: {
      chiefComplaint: 'Worsening memory and concentration since head injury',
      symptomsOnset: 'Gradual decline over 2 years post-injury',
      functionalStatus: 'Returned to work with accommodations',
      qualityOfLife: 6,
      caregiverSupport: 'Wife provides support, three adult children',
      mobilityAids: [],
    },
    assessmentHistory: [
      {
        date: '2024-01-25',
        sessionId: 'session_004_baseline',
        nriScore: 0.48,
        riskCategory: 'moderate',
        speechScore: 0.71,
        retinalScore: 0.76,
        motorScore: 0.52,
        cognitiveScore: 0.41,
        clinicalNotes:
          'Moderate risk with prominent cognitive and motor deficits consistent with post-TBI syndrome. Speech mildly affected.',
        followUpActions: [
          'Neuropsychological re-evaluation',
          'Occupational therapy',
          'Sleep study optimization',
        ],
        interventionsImplemented: [],
      },
      {
        date: '2024-07-25',
        sessionId: 'session_004_followup_6mo',
        nriScore: 0.42,
        riskCategory: 'moderate',
        speechScore: 0.76,
        retinalScore: 0.78,
        motorScore: 0.58,
        cognitiveScore: 0.48,
        clinicalNotes:
          'Improvement across all domains following comprehensive rehabilitation. Sleep optimization particularly beneficial.',
        followUpActions: ['Continue rehabilitation', 'Vocational counseling', 'Family education'],
        interventionsImplemented: [
          'CPAP optimization',
          'Cognitive rehabilitation',
          'Occupational therapy',
          'Pain management',
        ],
      },
    ],
    clinicalOutcomes: {
      diagnosisConfirmed: 'Post-Traumatic Brain Injury with Cognitive Sequelae',
      treatmentResponse: 'Good response to comprehensive rehabilitation',
      functionalImprovement: 'Improved work performance and daily functioning',
      qualityOfLifeChange: 2, // Improved from 6 to 8
      caregiverSatisfaction: 8,
      costSavings: 'Optimized rehabilitation focus, avoided unnecessary testing',
    },
    patientStory: {
      background:
        'Carlos sustained a traumatic brain injury in a construction accident and has struggled with cognitive changes.',
      initialConcerns:
        'Worried about progressive decline and ability to continue working to support family.',
      discoveryProcess:
        'NeuraLens assessment helped differentiate TBI sequelae from neurodegenerative disease.',
      clinicalJourney:
        'Comprehensive rehabilitation approach with focus on sleep, cognition, and occupational function.',
      currentOutcome:
        'Successfully returned to modified work duties, improved family relationships and quality of life.',
      patientQuote:
        'I thought my brain was getting worse and worse. Now I understand what happened and how to manage it. I can work and provide for my family.',
      familyImpact:
        'Family educated about TBI effects and recovery, reduced anxiety about progressive decline.',
    },
    accessibilityNeeds: {
      languageSupport: 'Spanish interpreter for complex medical discussions',
      cognitiveSupport: 'Written instructions and visual aids for complex information',
      assistiveTechnology: ['Smartphone apps for memory and organization'],
    },
  },
];

// Utility functions
export function getPatientById(id: string): PatientProfile | undefined {
  return PATIENT_PROFILES.find(patient => patient.id === id);
}

export function getPatientsByRiskCategory(category: 'low' | 'moderate' | 'high'): PatientProfile[] {
  return PATIENT_PROFILES.filter(
    patient =>
      patient.assessmentHistory[patient.assessmentHistory.length - 1]?.riskCategory === category,
  );
}

export function getPatientsByAge(minAge: number, maxAge: number): PatientProfile[] {
  return PATIENT_PROFILES.filter(
    patient => patient.demographics.age >= minAge && patient.demographics.age <= maxAge,
  );
}

export function getPatientsWithOutcomes(): PatientProfile[] {
  return PATIENT_PROFILES.filter(patient => patient.clinicalOutcomes.diagnosisConfirmed);
}

export function getAllPatients(): PatientProfile[] {
  return PATIENT_PROFILES;
}
