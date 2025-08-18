// NeuroLens-X Risk Assessment Algorithm
// Comprehensive neurological risk scoring incorporating multiple factors

export interface DemographicData {
  age: number;                    // Age in years
  sex: 'male' | 'female' | 'other';
  ethnicity: string;              // Ethnicity/race
  education: number;              // Years of education
  occupation: string;             // Current/previous occupation
  handedness: 'left' | 'right' | 'ambidextrous';
}

export interface MedicalHistory {
  // Cardiovascular
  hypertension: boolean;
  diabetes: boolean;
  heartDisease: boolean;
  stroke: boolean;
  cholesterol: number;            // mg/dL
  
  // Neurological
  headInjury: boolean;
  seizures: boolean;
  migraines: boolean;
  sleepDisorders: boolean;
  
  // Mental Health
  depression: boolean;
  anxiety: boolean;
  cognitiveComplaints: boolean;
  
  // Other
  thyroidDisease: boolean;
  kidneyDisease: boolean;
  liverDisease: boolean;
  autoimmune: boolean;
  
  // Medications
  medications: string[];
  supplements: string[];
}

export interface FamilyHistory {
  alzheimers: boolean;
  parkinsons: boolean;
  huntingtons: boolean;
  stroke: boolean;
  dementia: boolean;
  depression: boolean;
  diabetes: boolean;
  heartDisease: boolean;
  
  // Genetic factors
  apoeE4: boolean | null;         // APOE ε4 carrier status
  familyHistoryAge: number;       // Age of onset in family
}

export interface LifestyleFactors {
  // Physical Activity
  exerciseFrequency: number;      // Days per week
  exerciseIntensity: 'low' | 'moderate' | 'high';
  
  // Diet
  dietQuality: number;            // 0-10 scale
  mediterraneanDiet: boolean;
  alcohol: number;                // Drinks per week
  smoking: 'never' | 'former' | 'current';
  
  // Cognitive
  cognitiveActivity: number;      // 0-10 scale
  socialEngagement: number;       // 0-10 scale
  
  // Sleep
  sleepQuality: number;           // 0-10 scale
  sleepDuration: number;          // Hours per night
  
  // Stress
  stressLevel: number;            // 0-10 scale
  stressManagement: boolean;
  
  // Environmental
  airPollution: number;           // 0-10 scale
  occupationalExposure: boolean;
}

export interface CognitiveAssessment {
  memoryComplaints: boolean;
  concentrationIssues: boolean;
  languageProblems: boolean;
  executiveIssues: boolean;
  spatialIssues: boolean;
  
  // Functional assessment
  dailyActivities: number;        // 0-10 scale
  instrumentalActivities: number; // 0-10 scale
  
  // Mood assessment
  moodChanges: boolean;
  apathy: boolean;
  irritability: boolean;
}

export interface RiskAssessmentData {
  demographics: DemographicData;
  medicalHistory: MedicalHistory;
  familyHistory: FamilyHistory;
  lifestyle: LifestyleFactors;
  cognitive: CognitiveAssessment;
}

export interface RiskAssessmentResult {
  overallRisk: number;            // 0-100 overall risk score
  categoryRisks: {
    demographic: number;
    medical: number;
    family: number;
    lifestyle: number;
    cognitive: number;
  };
  modifiableFactors: string[];    // Factors that can be changed
  nonModifiableFactors: string[]; // Fixed risk factors
  recommendations: string[];      // Personalized recommendations
  confidence: number;             // Confidence in assessment
  processingTime: number;         // Calculation time (ms)
}

export class RiskAssessmentCalculator {
  private ageWeights = {
    under50: 0.1,
    age50to60: 0.3,
    age60to70: 0.6,
    age70to80: 0.8,
    over80: 1.0,
  };

  private ethnicityRisks = {
    'caucasian': 1.0,
    'african-american': 1.2,
    'hispanic': 1.1,
    'asian': 0.8,
    'other': 1.0,
  };

  /**
   * Calculate comprehensive neurological risk assessment
   */
  async calculateRisk(data: RiskAssessmentData): Promise<RiskAssessmentResult> {
    const startTime = performance.now();
    
    try {
      // Calculate category-specific risks
      const demographicRisk = this.calculateDemographicRisk(data.demographics);
      const medicalRisk = this.calculateMedicalRisk(data.medicalHistory);
      const familyRisk = this.calculateFamilyRisk(data.familyHistory);
      const lifestyleRisk = this.calculateLifestyleRisk(data.lifestyle);
      const cognitiveRisk = this.calculateCognitiveRisk(data.cognitive);
      
      // Calculate weighted overall risk
      const overallRisk = this.calculateOverallRisk({
        demographic: demographicRisk,
        medical: medicalRisk,
        family: familyRisk,
        lifestyle: lifestyleRisk,
        cognitive: cognitiveRisk,
      });
      
      // Identify modifiable and non-modifiable factors
      const modifiableFactors = this.identifyModifiableFactors(data);
      const nonModifiableFactors = this.identifyNonModifiableFactors(data);
      
      // Generate personalized recommendations
      const recommendations = this.generateRecommendations(data, overallRisk);
      
      // Calculate confidence
      const confidence = this.calculateConfidence(data);
      
      const processingTime = performance.now() - startTime;
      
      return {
        overallRisk,
        categoryRisks: {
          demographic: demographicRisk,
          medical: medicalRisk,
          family: familyRisk,
          lifestyle: lifestyleRisk,
          cognitive: cognitiveRisk,
        },
        modifiableFactors,
        nonModifiableFactors,
        recommendations,
        confidence,
        processingTime,
      };
    } catch (error) {
      console.error('Risk assessment failed:', error);
      throw new Error('Failed to calculate risk assessment: ' + (error as Error).message);
    }
  }

  /**
   * Calculate demographic risk factors
   */
  private calculateDemographicRisk(demographics: DemographicData): number {
    let risk = 0;
    
    // Age risk (strongest factor)
    if (demographics.age < 50) {
      risk += this.ageWeights.under50 * 10;
    } else if (demographics.age < 60) {
      risk += this.ageWeights.age50to60 * 20;
    } else if (demographics.age < 70) {
      risk += this.ageWeights.age60to70 * 40;
    } else if (demographics.age < 80) {
      risk += this.ageWeights.age70to80 * 60;
    } else {
      risk += this.ageWeights.over80 * 80;
    }
    
    // Sex risk (females have higher Alzheimer's risk, males higher vascular)
    if (demographics.sex === 'female') {
      risk += 5; // Slightly higher overall risk
    }
    
    // Ethnicity risk
    const ethnicityKey = demographics.ethnicity.toLowerCase().replace(/\s+/g, '-');
    const ethnicityMultiplier = this.ethnicityRisks[ethnicityKey as keyof typeof this.ethnicityRisks] || 1.0;
    risk *= ethnicityMultiplier;
    
    // Education protective effect
    if (demographics.education < 12) {
      risk += 10; // Less than high school
    } else if (demographics.education >= 16) {
      risk -= 5; // College education protective
    }
    
    return Math.min(Math.max(risk, 0), 100);
  }

  /**
   * Calculate medical history risk
   */
  private calculateMedicalRisk(medical: MedicalHistory): number {
    let risk = 0;
    
    // Cardiovascular risk factors
    if (medical.hypertension) risk += 15;
    if (medical.diabetes) risk += 20;
    if (medical.heartDisease) risk += 18;
    if (medical.stroke) risk += 25;
    if (medical.cholesterol > 240) risk += 10;
    
    // Neurological conditions
    if (medical.headInjury) risk += 12;
    if (medical.seizures) risk += 8;
    if (medical.sleepDisorders) risk += 10;
    
    // Mental health
    if (medical.depression) risk += 15;
    if (medical.anxiety) risk += 8;
    if (medical.cognitiveComplaints) risk += 20;
    
    // Other conditions
    if (medical.thyroidDisease) risk += 5;
    if (medical.autoimmune) risk += 8;
    
    // Medication effects
    const riskMedications = ['anticholinergics', 'benzodiazepines', 'antipsychotics'];
    const protectiveMedications = ['statins', 'ace-inhibitors', 'metformin'];
    
    medical.medications.forEach(med => {
      if (riskMedications.some(risk => med.toLowerCase().includes(risk))) {
        risk += 5;
      }
      if (protectiveMedications.some(protective => med.toLowerCase().includes(protective))) {
        risk -= 3;
      }
    });
    
    return Math.min(Math.max(risk, 0), 100);
  }

  /**
   * Calculate family history risk
   */
  private calculateFamilyRisk(family: FamilyHistory): number {
    let risk = 0;
    
    // Direct neurological family history
    if (family.alzheimers) risk += 25;
    if (family.parkinsons) risk += 20;
    if (family.huntingtons) risk += 30;
    if (family.dementia) risk += 20;
    
    // Related conditions
    if (family.stroke) risk += 15;
    if (family.depression) risk += 10;
    if (family.diabetes) risk += 8;
    if (family.heartDisease) risk += 10;
    
    // APOE ε4 genetic factor
    if (family.apoeE4 === true) {
      risk += 30; // Strong genetic risk factor
    } else if (family.apoeE4 === false) {
      risk -= 5; // Protective if known negative
    }
    
    // Age of onset in family (earlier onset = higher risk)
    if (family.familyHistoryAge > 0) {
      if (family.familyHistoryAge < 60) {
        risk += 15; // Early onset
      } else if (family.familyHistoryAge > 80) {
        risk -= 5; // Late onset
      }
    }
    
    return Math.min(Math.max(risk, 0), 100);
  }

  /**
   * Calculate lifestyle risk factors
   */
  private calculateLifestyleRisk(lifestyle: LifestyleFactors): number {
    let risk = 50; // Start at neutral
    
    // Physical activity (protective)
    if (lifestyle.exerciseFrequency >= 5) {
      risk -= 15; // Regular exercise
    } else if (lifestyle.exerciseFrequency >= 3) {
      risk -= 10; // Moderate exercise
    } else if (lifestyle.exerciseFrequency < 1) {
      risk += 15; // Sedentary
    }
    
    // Diet quality
    if (lifestyle.dietQuality >= 8) {
      risk -= 10; // Excellent diet
    } else if (lifestyle.dietQuality <= 4) {
      risk += 10; // Poor diet
    }
    
    if (lifestyle.mediterraneanDiet) {
      risk -= 8; // Mediterranean diet protective
    }
    
    // Alcohol consumption
    if (lifestyle.alcohol > 14) {
      risk += 12; // Heavy drinking
    } else if (lifestyle.alcohol >= 1 && lifestyle.alcohol <= 7) {
      risk -= 3; // Moderate drinking may be protective
    }
    
    // Smoking
    if (lifestyle.smoking === 'current') {
      risk += 20;
    } else if (lifestyle.smoking === 'former') {
      risk += 5;
    }
    
    // Cognitive activity (protective)
    if (lifestyle.cognitiveActivity >= 8) {
      risk -= 12;
    } else if (lifestyle.cognitiveActivity <= 3) {
      risk += 10;
    }
    
    // Social engagement (protective)
    if (lifestyle.socialEngagement >= 8) {
      risk -= 8;
    } else if (lifestyle.socialEngagement <= 3) {
      risk += 12;
    }
    
    // Sleep quality
    if (lifestyle.sleepQuality <= 4) {
      risk += 10; // Poor sleep
    }
    if (lifestyle.sleepDuration < 6 || lifestyle.sleepDuration > 9) {
      risk += 8; // Abnormal sleep duration
    }
    
    // Stress
    if (lifestyle.stressLevel >= 8) {
      risk += 15; // High stress
    }
    if (lifestyle.stressManagement) {
      risk -= 5; // Stress management protective
    }
    
    // Environmental factors
    if (lifestyle.airPollution >= 7) {
      risk += 8; // High pollution exposure
    }
    if (lifestyle.occupationalExposure) {
      risk += 10; // Occupational hazards
    }
    
    return Math.min(Math.max(risk, 0), 100);
  }

  /**
   * Calculate cognitive assessment risk
   */
  private calculateCognitiveRisk(cognitive: CognitiveAssessment): number {
    let risk = 0;
    
    // Cognitive complaints
    if (cognitive.memoryComplaints) risk += 20;
    if (cognitive.concentrationIssues) risk += 15;
    if (cognitive.languageProblems) risk += 18;
    if (cognitive.executiveIssues) risk += 22;
    if (cognitive.spatialIssues) risk += 16;
    
    // Functional assessment
    if (cognitive.dailyActivities <= 6) risk += 15;
    if (cognitive.instrumentalActivities <= 6) risk += 20;
    
    // Mood changes
    if (cognitive.moodChanges) risk += 12;
    if (cognitive.apathy) risk += 15;
    if (cognitive.irritability) risk += 8;
    
    return Math.min(Math.max(risk, 0), 100);
  }

  /**
   * Calculate weighted overall risk
   */
  private calculateOverallRisk(categoryRisks: {
    demographic: number;
    medical: number;
    family: number;
    lifestyle: number;
    cognitive: number;
  }): number {
    // Weighted combination of risk categories
    const weights = {
      demographic: 0.25,  // Age is strongest predictor
      medical: 0.20,      // Medical conditions important
      family: 0.20,       // Genetic factors significant
      lifestyle: 0.15,    // Modifiable factors
      cognitive: 0.20,    // Current symptoms important
    };
    
    const weightedRisk = 
      categoryRisks.demographic * weights.demographic +
      categoryRisks.medical * weights.medical +
      categoryRisks.family * weights.family +
      categoryRisks.lifestyle * weights.lifestyle +
      categoryRisks.cognitive * weights.cognitive;
    
    return Math.min(Math.max(Math.round(weightedRisk), 0), 100);
  }

  /**
   * Identify modifiable risk factors
   */
  private identifyModifiableFactors(data: RiskAssessmentData): string[] {
    const factors: string[] = [];
    
    // Lifestyle factors
    if (data.lifestyle.exerciseFrequency < 3) {
      factors.push('Increase physical activity');
    }
    if (data.lifestyle.dietQuality < 6) {
      factors.push('Improve diet quality');
    }
    if (data.lifestyle.smoking === 'current') {
      factors.push('Smoking cessation');
    }
    if (data.lifestyle.alcohol > 14) {
      factors.push('Reduce alcohol consumption');
    }
    if (data.lifestyle.sleepQuality < 6) {
      factors.push('Improve sleep quality');
    }
    if (data.lifestyle.stressLevel > 7) {
      factors.push('Stress management');
    }
    if (data.lifestyle.cognitiveActivity < 6) {
      factors.push('Increase cognitive stimulation');
    }
    if (data.lifestyle.socialEngagement < 6) {
      factors.push('Enhance social engagement');
    }
    
    // Medical factors
    if (data.medicalHistory.hypertension) {
      factors.push('Blood pressure management');
    }
    if (data.medicalHistory.diabetes) {
      factors.push('Diabetes control');
    }
    if (data.medicalHistory.cholesterol > 200) {
      factors.push('Cholesterol management');
    }
    if (data.medicalHistory.depression) {
      factors.push('Depression treatment');
    }
    
    return factors;
  }

  /**
   * Identify non-modifiable risk factors
   */
  private identifyNonModifiableFactors(data: RiskAssessmentData): string[] {
    const factors: string[] = [];
    
    // Demographics
    if (data.demographics.age > 65) {
      factors.push('Advanced age');
    }
    if (data.demographics.sex === 'female') {
      factors.push('Female sex (Alzheimer\'s risk)');
    }
    if (data.demographics.education < 12) {
      factors.push('Limited education');
    }
    
    // Family history
    if (data.familyHistory.alzheimers) {
      factors.push('Family history of Alzheimer\'s');
    }
    if (data.familyHistory.parkinsons) {
      factors.push('Family history of Parkinson\'s');
    }
    if (data.familyHistory.apoeE4) {
      factors.push('APOE ε4 genetic variant');
    }
    
    // Medical history
    if (data.medicalHistory.stroke) {
      factors.push('History of stroke');
    }
    if (data.medicalHistory.headInjury) {
      factors.push('History of head injury');
    }
    
    return factors;
  }

  /**
   * Generate personalized recommendations
   */
  private generateRecommendations(data: RiskAssessmentData, overallRisk: number): string[] {
    const recommendations: string[] = [];
    
    // Risk-based recommendations
    if (overallRisk > 75) {
      recommendations.push('Consult with neurologist for comprehensive evaluation');
      recommendations.push('Consider cognitive testing and brain imaging');
    } else if (overallRisk > 50) {
      recommendations.push('Regular monitoring with healthcare provider');
      recommendations.push('Annual cognitive screening');
    } else if (overallRisk > 25) {
      recommendations.push('Maintain healthy lifestyle practices');
      recommendations.push('Monitor for cognitive changes');
    }
    
    // Specific lifestyle recommendations
    if (data.lifestyle.exerciseFrequency < 3) {
      recommendations.push('Aim for 150 minutes of moderate exercise weekly');
    }
    if (!data.lifestyle.mediterraneanDiet) {
      recommendations.push('Consider Mediterranean-style diet');
    }
    if (data.lifestyle.cognitiveActivity < 6) {
      recommendations.push('Engage in mentally stimulating activities');
    }
    if (data.lifestyle.socialEngagement < 6) {
      recommendations.push('Maintain active social connections');
    }
    
    // Medical recommendations
    if (data.medicalHistory.hypertension || data.medicalHistory.diabetes) {
      recommendations.push('Optimize cardiovascular risk factor management');
    }
    if (data.medicalHistory.sleepDisorders) {
      recommendations.push('Address sleep disorders with healthcare provider');
    }
    
    return recommendations;
  }

  /**
   * Calculate confidence in assessment
   */
  private calculateConfidence(data: RiskAssessmentData): number {
    let confidence = 85; // Base confidence
    
    // Reduce confidence for missing data
    if (data.familyHistory.apoeE4 === null) confidence -= 10;
    if (!data.medicalHistory.cholesterol) confidence -= 5;
    
    // Increase confidence for comprehensive data
    if (data.lifestyle.cognitiveActivity > 0) confidence += 5;
    if (data.cognitive.memoryComplaints !== undefined) confidence += 5;
    
    return Math.min(Math.max(confidence, 60), 95);
  }
}

// Export singleton instance
export const riskAssessmentCalculator = new RiskAssessmentCalculator();
