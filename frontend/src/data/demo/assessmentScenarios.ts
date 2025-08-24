/**
 * Assessment Scenarios for Clinical Demonstration
 * Complete assessment scenarios showcasing different clinical pathways and outcomes
 */

import { PatientProfile } from '../clinical/patientProfiles';
import { ClinicalTestDataset } from '../clinical/testDatasets';

// Assessment scenario interfaces
export interface AssessmentScenario {
  id: string;
  title: string;
  description: string;
  clinicalContext: string;
  patientProfile: string; // Reference to patient profile ID
  testDataset: string; // Reference to test dataset ID
  demonstrationScript: DemonstrationScript;
  learningObjectives: string[];
  clinicalDecisionPoints: ClinicalDecisionPoint[];
  expectedOutcomes: ScenarioOutcomes;
  accessibilityFeatures?: AccessibilityDemonstration;
}

export interface DemonstrationScript {
  introduction: ScriptSection;
  assessmentProcess: ScriptSection;
  resultsReview: ScriptSection;
  clinicalRecommendations: ScriptSection;
  followUpPlanning: ScriptSection;
  conclusion: ScriptSection;
}

export interface ScriptSection {
  title: string;
  duration: string;
  narratorText: string;
  keyPoints: string[];
  visualElements: string[];
  interactionPoints: string[];
}

export interface ClinicalDecisionPoint {
  stage: string;
  decision: string;
  rationale: string;
  alternatives: string[];
  clinicalEvidence: string;
}

export interface ScenarioOutcomes {
  primaryDiagnosis: string;
  timeToDetection: string;
  interventionSuccess: string;
  patientSatisfaction: number;
  clinicalValue: string;
  costEffectiveness: string;
}

export interface AccessibilityDemonstration {
  screenReaderDemo: boolean;
  keyboardNavigationDemo: boolean;
  visualImpairmentSupport: boolean;
  languageSupport: boolean;
  cognitiveSupport: boolean;
}

// Comprehensive assessment scenarios
export const ASSESSMENT_SCENARIOS: AssessmentScenario[] = [
  {
    id: 'scenario_001_early_parkinson_detection',
    title: "Early Parkinson's Disease Detection",
    description:
      "Demonstration of early-stage Parkinson's disease detection in a high-functioning patient with subtle symptoms",
    clinicalContext:
      "Annual wellness visit for retired teacher with family history of Parkinson's disease and subtle motor symptoms",
    patientProfile: 'patient_001_margaret_chen',
    testDataset: 'early_parkinson_high_risk',
    demonstrationScript: {
      introduction: {
        title: 'Patient Introduction and Clinical Context',
        duration: '2 minutes',
        narratorText:
          'Meet Margaret Chen, a 67-year-old retired teacher who comes in for her annual wellness visit. She mentions subtle hand tremor and occasional word-finding difficulties that started 6 months ago.',
        keyPoints: [
          "Family history of Parkinson's disease (father diagnosed at 72)",
          'Subtle symptoms that might be dismissed as normal aging',
          'Patient is highly functional and independent',
          'Early detection opportunity for better outcomes',
        ],
        visualElements: [
          'Patient demographic display',
          'Medical history timeline',
          'Family history visualization',
          'Current symptom assessment',
        ],
        interactionPoints: [
          'Review patient background',
          'Discuss symptom onset and progression',
          'Explain assessment rationale',
        ],
      },
      assessmentProcess: {
        title: 'Multi-Modal Assessment Execution',
        duration: '5 minutes',
        narratorText:
          'Watch as Margaret completes the comprehensive NeuraLens assessment, including speech analysis, retinal imaging, motor function testing, and cognitive evaluation.',
        keyPoints: [
          'Speech analysis detects voice tremor and reduced vocal stability',
          'Retinal imaging shows reduced vessel density and increased tortuosity',
          'Motor assessment reveals significant tremor and mild bradykinesia',
          'Cognitive testing shows mild executive function changes',
        ],
        visualElements: [
          'Real-time assessment progress indicators',
          'Live biomarker visualization',
          'Quality metrics display',
          'Processing status updates',
        ],
        interactionPoints: [
          'Observe real-time data collection',
          'Monitor quality indicators',
          'Review preliminary findings',
        ],
      },
      resultsReview: {
        title: 'Comprehensive Results Analysis',
        duration: '3 minutes',
        narratorText:
          'The NeuraLens system generates a comprehensive risk assessment with NRI score of 0.72, indicating high neurological risk requiring immediate attention.',
        keyPoints: [
          'NRI score of 0.72 indicates high risk (>0.6 threshold)',
          'Confidence level of 87% provides reliable assessment',
          "Multiple biomarkers converge on Parkinson's disease pattern",
          'Early detection before clinical diagnosis is obvious',
        ],
        visualElements: [
          'NRI score visualization with risk categorization',
          'Biomarker correlation matrix',
          'Trend analysis and comparison to normative data',
          'Confidence intervals and uncertainty quantification',
        ],
        interactionPoints: [
          'Explore detailed biomarker results',
          'Compare to age-matched controls',
          'Review confidence metrics',
        ],
      },
      clinicalRecommendations: {
        title: 'Evidence-Based Clinical Recommendations',
        duration: '2 minutes',
        narratorText:
          "The system generates prioritized clinical recommendations based on established guidelines and Margaret's specific risk profile.",
        keyPoints: [
          'Urgent neurologist consultation within 2 weeks (Critical priority)',
          'DaTscan imaging consideration for confirmation',
          'Symptom monitoring diary initiation',
          "Family education about Parkinson's disease",
        ],
        visualElements: [
          'Prioritized recommendation list',
          'Evidence sources and clinical guidelines',
          'Timeline for follow-up actions',
          'Patient education materials',
        ],
        interactionPoints: [
          'Review recommendation priorities',
          'Access clinical evidence sources',
          'Schedule follow-up appointments',
        ],
      },
      followUpPlanning: {
        title: 'Longitudinal Care Coordination',
        duration: '2 minutes',
        narratorText:
          "Follow Margaret's journey over 6 months, showing how early detection and treatment led to excellent outcomes and maintained quality of life.",
        keyPoints: [
          "Confirmed Parkinson's diagnosis within 3 weeks",
          'Early treatment initiation with excellent response',
          'NRI score improvement from 0.72 to 0.58 over 6 months',
          'Maintained independence and active lifestyle',
        ],
        visualElements: [
          'Longitudinal assessment timeline',
          'Treatment response visualization',
          'Quality of life metrics',
          'Cost-effectiveness analysis',
        ],
        interactionPoints: [
          'Review treatment timeline',
          'Analyze outcome metrics',
          'Explore cost savings',
        ],
      },
      conclusion: {
        title: 'Clinical Impact and Value Demonstration',
        duration: '1 minute',
        narratorText:
          "Margaret's case demonstrates the transformative impact of early neurological detection, leading to better outcomes and preserved quality of life.",
        keyPoints: [
          'Early detection enabled optimal treatment timing',
          'Avoided 6-month diagnostic delay',
          'Estimated $15,000 in cost savings',
          'Patient satisfaction score: 9/10',
        ],
        visualElements: [
          'Outcome summary dashboard',
          'Patient testimonial video',
          'Clinical value metrics',
          'Return on investment analysis',
        ],
        interactionPoints: [
          'Review final outcomes',
          'Listen to patient testimonial',
          'Analyze clinical value',
        ],
      },
    },
    learningObjectives: [
      "Demonstrate early Parkinson's disease detection capabilities",
      'Show multi-modal biomarker integration and analysis',
      'Illustrate clinical decision support and recommendation generation',
      'Highlight the value of early intervention and treatment',
      'Showcase longitudinal monitoring and outcome tracking',
    ],
    clinicalDecisionPoints: [
      {
        stage: 'Initial Assessment',
        decision: 'Proceed with comprehensive neurological assessment',
        rationale:
          'Family history and subtle symptoms warrant evaluation despite normal appearance',
        alternatives: ['Wait and watch approach', 'Refer directly to neurology'],
        clinicalEvidence:
          "Early detection improves outcomes in Parkinson's disease (Movement Disorders Society, 2015)",
      },
      {
        stage: 'Results Interpretation',
        decision: 'Urgent neurologist referral based on high NRI score',
        rationale: "NRI score >0.6 has 85% correlation with early Parkinson's markers",
        alternatives: ['Routine referral', 'Repeat assessment in 6 months'],
        clinicalEvidence:
          'High sensitivity and specificity of multi-modal biomarkers for early PD detection',
      },
    ],
    expectedOutcomes: {
      primaryDiagnosis: "Early-stage Parkinson's Disease",
      timeToDetection: '3 weeks from initial assessment',
      interventionSuccess: 'Excellent response to dopaminergic therapy',
      patientSatisfaction: 9,
      clinicalValue: 'Maintained independence and quality of life',
      costEffectiveness: '$15,000 savings from avoided diagnostic delay',
    },
  },

  {
    id: 'scenario_002_vascular_cognitive_impairment',
    title: 'Vascular Cognitive Impairment Detection',
    description:
      'Identification and management of vascular cognitive impairment in a patient with diabetes and hypertension',
    clinicalContext:
      'Routine diabetes visit for patient with memory complaints and vascular risk factors',
    patientProfile: 'patient_002_robert_johnson',
    testDataset: 'vascular_risk_moderate',
    demonstrationScript: {
      introduction: {
        title: 'Patient with Vascular Risk Factors',
        duration: '2 minutes',
        narratorText:
          'Robert Johnson, a 72-year-old retired factory worker with diabetes and hypertension, presents with memory concerns and difficulty managing medications.',
        keyPoints: [
          'Long-standing diabetes and hypertension',
          'Family history of stroke and vascular disease',
          'Gradual cognitive decline over 1 year',
          'Functional impact on medication management',
        ],
        visualElements: [
          'Vascular risk factor assessment',
          'Cognitive complaint timeline',
          'Functional status evaluation',
          'Caregiver support assessment',
        ],
        interactionPoints: [
          'Review vascular risk profile',
          'Assess functional impact',
          'Discuss family concerns',
        ],
      },
      assessmentProcess: {
        title: 'Vascular-Focused Assessment',
        duration: '4 minutes',
        narratorText:
          'The assessment reveals prominent retinal vascular changes and cognitive deficits consistent with vascular etiology.',
        keyPoints: [
          'Retinal imaging shows diabetic retinopathy and vascular narrowing',
          'Cognitive testing reveals processing speed and executive deficits',
          'Speech analysis shows mild dysarthria',
          'Motor assessment indicates mild coordination problems',
        ],
        visualElements: [
          'Retinal vascular analysis',
          'Cognitive domain breakdown',
          'Vascular biomarker correlation',
          'Risk stratification display',
        ],
        interactionPoints: [
          'Examine retinal findings',
          'Analyze cognitive patterns',
          'Review vascular correlations',
        ],
      },
      resultsReview: {
        title: 'Vascular Cognitive Impairment Identification',
        duration: '3 minutes',
        narratorText:
          "NRI score of 0.61 indicates moderate risk with pattern consistent with vascular cognitive impairment rather than Alzheimer's disease.",
        keyPoints: [
          'Moderate risk category with vascular pattern',
          'Retinal and cognitive scores most affected',
          "Pattern distinct from Alzheimer's disease",
          'Potentially modifiable risk factors identified',
        ],
        visualElements: [
          "Vascular vs. Alzheimer's pattern comparison",
          'Modifiable risk factor identification',
          'Progression risk assessment',
          'Treatment target visualization',
        ],
        interactionPoints: [
          'Compare diagnostic patterns',
          'Identify intervention targets',
          'Review prognosis',
        ],
      },
      clinicalRecommendations: {
        title: 'Vascular Risk Management Strategy',
        duration: '2 minutes',
        narratorText:
          'Recommendations focus on vascular risk factor optimization and cognitive support interventions.',
        keyPoints: [
          'Optimize diabetes and blood pressure control',
          'Neuropsychological evaluation for cognitive training',
          'Cardiovascular risk reduction strategies',
          'Family education and support services',
        ],
        visualElements: [
          'Vascular risk optimization plan',
          'Cognitive intervention options',
          'Family support resources',
          'Monitoring schedule',
        ],
        interactionPoints: [
          'Review intervention plan',
          'Access educational resources',
          'Schedule follow-up care',
        ],
      },
      followUpPlanning: {
        title: 'Stabilization and Improvement',
        duration: '2 minutes',
        narratorText:
          'Six-month follow-up shows cognitive stabilization and improved vascular control with comprehensive management.',
        keyPoints: [
          'Cognitive function stabilized with interventions',
          'Improved diabetes and blood pressure control',
          'Enhanced family support and education',
          'Maintained community independence',
        ],
        visualElements: [
          'Cognitive stability metrics',
          'Vascular control improvements',
          'Quality of life measures',
          'Caregiver satisfaction scores',
        ],
        interactionPoints: [
          'Review stability metrics',
          'Analyze intervention effectiveness',
          'Plan long-term management',
        ],
      },
      conclusion: {
        title: 'Vascular Cognitive Health Management',
        duration: '1 minute',
        narratorText:
          "Robert's case demonstrates the importance of identifying and managing vascular contributions to cognitive impairment.",
        keyPoints: [
          'Early identification of vascular cognitive impairment',
          'Successful risk factor modification',
          'Prevented further cognitive decline',
          'Improved family understanding and support',
        ],
        visualElements: [
          'Vascular management success metrics',
          'Family education impact',
          'Prevention of decline visualization',
          'Healthcare utilization analysis',
        ],
        interactionPoints: [
          'Review prevention success',
          'Analyze family impact',
          'Explore cost benefits',
        ],
      },
    },
    learningObjectives: [
      'Identify vascular contributions to cognitive impairment',
      'Demonstrate retinal vascular analysis capabilities',
      'Show targeted intervention for modifiable risk factors',
      'Illustrate cognitive stabilization through vascular management',
      'Highlight importance of family education and support',
    ],
    clinicalDecisionPoints: [
      {
        stage: 'Differential Diagnosis',
        decision: "Vascular cognitive impairment vs. Alzheimer's disease",
        rationale: 'Pattern of deficits and vascular risk factors suggest vascular etiology',
        alternatives: ["Alzheimer's disease workup", 'Mixed dementia evaluation'],
        clinicalEvidence: 'Vascular cognitive impairment diagnostic criteria (AHA/ASA, 2011)',
      },
    ],
    expectedOutcomes: {
      primaryDiagnosis: 'Vascular Cognitive Impairment',
      timeToDetection: '4 weeks from initial assessment',
      interventionSuccess: 'Cognitive stabilization with vascular management',
      patientSatisfaction: 7,
      clinicalValue: 'Prevented further cognitive decline',
      costEffectiveness: "Avoided unnecessary Alzheimer's workup",
    },
  },

  {
    id: 'scenario_003_healthy_aging_reassurance',
    title: 'Healthy Aging Reassurance',
    description:
      'Providing objective reassurance to a worried-well patient with family history concerns',
    clinicalContext:
      'Preventive care visit for high-functioning professional concerned about family history of dementia',
    patientProfile: 'patient_003_sarah_williams',
    testDataset: 'healthy_aging_low_risk',
    demonstrationScript: {
      introduction: {
        title: 'Worried-Well Patient Assessment',
        duration: '2 minutes',
        narratorText:
          "Sarah Williams, a 45-year-old research scientist, seeks evaluation due to strong family history of early-onset Alzheimer's disease and anxiety about subtle memory changes.",
        keyPoints: [
          'High-functioning professional with PhD education',
          'Strong family history of early-onset dementia',
          'Anxiety about normal memory lapses',
          'Seeking objective assessment for reassurance',
        ],
        visualElements: [
          'Family history pedigree',
          'Anxiety assessment scores',
          'Cognitive concern timeline',
          'Professional functioning metrics',
        ],
        interactionPoints: [
          'Review family history concerns',
          'Assess anxiety impact',
          'Discuss assessment goals',
        ],
      },
      assessmentProcess: {
        title: 'Comprehensive Cognitive Assessment',
        duration: '3 minutes',
        narratorText:
          'Sarah completes the full assessment battery, demonstrating excellent performance across all cognitive and neurological domains.',
        keyPoints: [
          'Excellent cognitive performance across all domains',
          'Normal speech patterns and vocal quality',
          'Healthy retinal appearance with no pathology',
          'Superior motor function and coordination',
        ],
        visualElements: [
          'High-performance cognitive metrics',
          'Normal biomarker patterns',
          'Age-adjusted comparison charts',
          'Quality assurance indicators',
        ],
        interactionPoints: [
          'Observe excellent performance',
          'Compare to age norms',
          'Review quality metrics',
        ],
      },
      resultsReview: {
        title: 'Objective Reassurance Through Data',
        duration: '3 minutes',
        narratorText:
          'NRI score of 0.28 indicates low neurological risk with performance well above age-expected norms across all domains.',
        keyPoints: [
          'Low risk category with high confidence (91%)',
          'Performance above 90th percentile for age',
          'No evidence of neurological pathology',
          'Objective data supports cognitive health',
        ],
        visualElements: [
          'Low risk visualization',
          'Percentile ranking display',
          'Normal aging comparison',
          'Confidence interval analysis',
        ],
        interactionPoints: [
          'Review low risk indicators',
          'Compare to normative data',
          'Analyze confidence metrics',
        ],
      },
      clinicalRecommendations: {
        title: 'Wellness and Prevention Focus',
        duration: '2 minutes',
        narratorText:
          'Recommendations focus on maintaining cognitive health, managing anxiety, and continuing preventive care.',
        keyPoints: [
          'Continue current healthy lifestyle practices',
          'Stress management and anxiety reduction',
          'Annual monitoring for peace of mind',
          'Genetic counseling option if desired',
        ],
        visualElements: [
          'Wellness maintenance plan',
          'Stress reduction strategies',
          'Monitoring schedule',
          'Genetic counseling resources',
        ],
        interactionPoints: [
          'Review wellness strategies',
          'Access stress management tools',
          'Plan monitoring schedule',
        ],
      },
      followUpPlanning: {
        title: 'Anxiety Reduction and Continued Health',
        duration: '2 minutes',
        narratorText:
          'Six-month follow-up shows reduced anxiety, improved work-life balance, and continued excellent cognitive performance.',
        keyPoints: [
          'Significant anxiety reduction about cognitive health',
          'Improved work-life balance and stress management',
          'Continued high-level professional functioning',
          'Enhanced quality of life and relationships',
        ],
        visualElements: [
          'Anxiety reduction metrics',
          'Quality of life improvements',
          'Stress management success',
          'Relationship satisfaction scores',
        ],
        interactionPoints: [
          'Review anxiety improvements',
          'Analyze quality of life gains',
          'Plan continued wellness',
        ],
      },
      conclusion: {
        title: 'Value of Objective Reassurance',
        duration: '1 minute',
        narratorText:
          "Sarah's case demonstrates the value of objective cognitive assessment in providing reassurance and reducing healthcare anxiety.",
        keyPoints: [
          'Objective data provided meaningful reassurance',
          'Reduced unnecessary healthcare utilization',
          'Improved mental health and quality of life',
          'Cost-effective anxiety management',
        ],
        visualElements: [
          'Reassurance value metrics',
          'Healthcare utilization reduction',
          'Mental health improvements',
          'Cost-effectiveness analysis',
        ],
        interactionPoints: [
          'Review reassurance impact',
          'Analyze cost savings',
          'Explore prevention value',
        ],
      },
    },
    learningObjectives: [
      'Demonstrate value of objective cognitive assessment',
      'Show appropriate use of technology for worried-well patients',
      'Illustrate anxiety reduction through data-driven reassurance',
      'Highlight prevention and wellness focus',
      'Showcase cost-effective healthcare utilization',
    ],
    clinicalDecisionPoints: [
      {
        stage: 'Assessment Indication',
        decision: 'Comprehensive assessment vs. clinical reassurance alone',
        rationale: 'Objective data more effective than clinical reassurance for anxiety reduction',
        alternatives: ['Clinical interview only', 'Psychiatric referral'],
        clinicalEvidence:
          'Objective testing reduces health anxiety more effectively than reassurance alone',
      },
    ],
    expectedOutcomes: {
      primaryDiagnosis: 'No neurological pathology - normal cognitive aging',
      timeToDetection: 'Immediate reassurance provided',
      interventionSuccess: 'Significant anxiety reduction and improved quality of life',
      patientSatisfaction: 9,
      clinicalValue: 'Prevented unnecessary neurological workup',
      costEffectiveness: 'Avoided extensive and expensive neurological evaluation',
    },
    accessibilityFeatures: {
      screenReaderDemo: true,
      keyboardNavigationDemo: true,
      visualImpairmentSupport: false,
      languageSupport: false,
      cognitiveSupport: false,
    },
  },
];

// Utility functions
export function getScenarioById(id: string): AssessmentScenario | undefined {
  return ASSESSMENT_SCENARIOS.find(scenario => scenario.id === id);
}

export function getScenariosByOutcome(outcome: string): AssessmentScenario[] {
  return ASSESSMENT_SCENARIOS.filter(scenario =>
    scenario.expectedOutcomes.primaryDiagnosis.toLowerCase().includes(outcome.toLowerCase()),
  );
}

export function getScenariosWithAccessibility(): AssessmentScenario[] {
  return ASSESSMENT_SCENARIOS.filter(scenario => scenario.accessibilityFeatures);
}

export function getAllScenarios(): AssessmentScenario[] {
  return ASSESSMENT_SCENARIOS;
}
