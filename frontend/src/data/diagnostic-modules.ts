/**
 * Diagnostic Modules Configuration
 *
 * Defines all 11 diagnostic modules for the MediLens platform.
 * Each module includes status (available/coming-soon), icons, routes,
 * descriptions, diagnoses, and MediLens design system colors.
 *
 * Requirements: 3.1, 3.4, 3.5
 */

import {
  LucideIcon,
  Mic,
  Eye,
  Hand,
  Brain,
  Activity,
  Zap,
  Scan,
  Microscope,
  Heart,
  Wind,
  Footprints,
  Bone,
  Sparkles,
} from "lucide-react";

/**
 * Diagnostic Module Interface
 * Defines the structure for each diagnostic module in the platform.
 */
export interface DiagnosticModule {
  /** Unique identifier for the module */
  id: string;
  /** Display name of the module */
  name: string;
  /** Brief description of what the module does */
  description: string;
  /** Lucide icon component for the module */
  icon: LucideIcon;
  /** Route path for the module page */
  route: string;
  /** Current availability status */
  status: "available" | "coming-soon";
  /** Category for grouping modules */
  category: "current" | "upcoming";
  /** List of conditions the module can diagnose */
  diagnoses: string[];
  /** Brief explanation of how the module works */
  howItWorks: string;
  /** Tailwind gradient classes for the module card */
  gradient: string;
}

/**
 * All diagnostic modules available in MediLens
 *
 * Current Available Modules (9):
 * - RetinaScan AI (Ophthalmology)
 * - ChestXplorer AI (Radiology)
 * - CardioPredict AI (Cardiology)
 * - SpeechMD AI (Speech & Cognitive)
 * - SkinSense AI (Dermatology)
 * - Motor Assessment
 * - Cognitive Testing
 * - Multi-Modal Assessment
 * - NRI Fusion Engine
 *
 * Coming Soon Modules (5):
 * - Pathology (HistoVision AI)
 * - Neurology (NeuroScan AI)
 * - Pulmonology (RespiRate AI)
 * - Diabetic Foot Care (FootCare AI)
 * - Orthopedics (BoneScan AI)
 */
export const diagnosticModules: DiagnosticModule[] = [
  // ============================================
  // CURRENT AVAILABLE MODULES (9 Live Modules)
  // ============================================
  {
    id: "retinal",
    name: "RetinaScan AI",
    description: "Retinal imaging for diabetic retinopathy and eye diseases",
    icon: Eye,
    route: "/dashboard/retinal",
    status: "available",
    category: "current",
    diagnoses: [
      "Diabetic Retinopathy",
      "Glaucoma",
      "AMD",
      "Hypertensive Retinopathy",
    ],
    howItWorks: "Upload retinal fundus images for AI heatmap analysis",
    gradient: "from-cyan-500 to-teal-500",
  },
  {
    id: "radiology",
    name: "ChestXplorer AI",
    description: "Chest X-ray analysis for pneumonia, TB, and lung conditions",
    icon: Scan,
    route: "/dashboard/radiology",
    status: "available",
    category: "current",
    diagnoses: [
      "Pneumonia",
      "COVID-19",
      "TB",
      "Lung cancer",
      "Pleural effusion",
    ],
    howItWorks: "Upload chest X-ray for multi-class classification",
    gradient: "from-sky-500 to-blue-500",
  },
  {
    id: "cardiology",
    name: "CardioPredict AI",
    description: "ECG analysis for arrhythmia and cardiac conditions",
    icon: Heart,
    route: "/dashboard/cardiology",
    status: "available",
    category: "current",
    diagnoses: [
      "Arrhythmia",
      "Atrial Fibrillation",
      "Myocardial Infarction",
      "Heart murmur",
    ],
    howItWorks: "Upload ECG signal for rhythm abnormality detection",
    gradient: "from-red-500 to-pink-500",
  },
  {
    id: "speech",
    name: "SpeechMD AI",
    description: "Voice analysis for Parkinson's and cognitive assessment",
    icon: Mic,
    route: "/dashboard/speech",
    status: "available",
    category: "current",
    diagnoses: [
      "Parkinson's disease",
      "Aphasia",
      "Early dementia",
      "Depression/anxiety",
    ],
    howItWorks: "Record patient speech for acoustic feature extraction",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    id: "dermatology",
    name: "SkinSense AI",
    description: "Skin lesion classification and melanoma detection",
    icon: Sparkles,
    route: "/dashboard/dermatology",
    status: "available",
    category: "current",
    diagnoses: [
      "Melanoma",
      "Basal cell carcinoma",
      "Benign nevi",
      "Eczema",
      "Psoriasis",
    ],
    howItWorks: "Upload smartphone photo for ABCDE criteria analysis",
    gradient: "from-purple-400 to-pink-400",
  },
  {
    id: "motor",
    name: "Motor Assessment",
    description: "Movement pattern and tremor detection",
    icon: Hand,
    route: "/dashboard/motor",
    status: "available",
    category: "current",
    diagnoses: ["Bradykinesia", "Tremor", "Coordination issues"],
    howItWorks: "Interactive finger tapping and coordination tests",
    gradient: "from-purple-400 to-indigo-400",
  },
  {
    id: "cognitive",
    name: "Cognitive Testing",
    description: "Memory and executive function assessment",
    icon: Brain,
    route: "/dashboard/cognitive",
    status: "available",
    category: "current",
    diagnoses: ["MCI", "Memory impairment", "Executive dysfunction"],
    howItWorks: "Comprehensive cognitive evaluation tasks",
    gradient: "from-orange-400 to-red-400",
  },
  {
    id: "multimodal",
    name: "Multi-Modal",
    description: "Integrated multi-modal neurological assessment",
    icon: Activity,
    route: "/dashboard/multimodal",
    status: "available",
    category: "current",
    diagnoses: [
      "Combined risk assessment",
      "Cross-modal analysis",
      "Comprehensive profile",
    ],
    howItWorks: "Combines Speech, Retinal, Motor, and Cognitive analysis",
    gradient: "from-purple-500 to-violet-500",
  },
  {
    id: "nri-fusion",
    name: "NRI Fusion",
    description: "Neurological Risk Index fusion engine",
    icon: Zap,
    route: "/dashboard/nri-fusion",
    status: "available",
    category: "current",
    diagnoses: [
      "Neurological Risk Index",
      "Biomarker fusion",
      "Predictive analytics",
    ],
    howItWorks: "Bayesian fusion of multi-modal biomarkers",
    gradient: "from-yellow-400 to-orange-400",
  },

  // ============================================
  // COMING SOON MODULES
  // ============================================
  {
    id: "pathology",
    name: "HistoVision AI",
    description: "Tissue sample and blood smear analysis",
    icon: Microscope,
    route: "/dashboard/pathology",
    status: "coming-soon",
    category: "upcoming",
    diagnoses: ["Cancer detection", "Malaria", "Leukemia"],
    howItWorks: "Upload microscopy images for cell analysis",
    gradient: "from-violet-400 to-purple-400",
  },
  {
    id: "neurology",
    name: "NeuroScan AI",
    description: "Brain MRI and CT scan analysis",
    icon: Brain,
    route: "/dashboard/neurology",
    status: "coming-soon",
    category: "upcoming",
    diagnoses: ["Brain tumors", "Stroke", "MS lesions", "Hemorrhage"],
    howItWorks: "Upload MRI or CT scan (DICOM support)",
    gradient: "from-indigo-400 to-blue-400",
  },
  {
    id: "pulmonology",
    name: "RespiRate AI",
    description: "Respiratory sound and spirometry analysis",
    icon: Wind,
    route: "/dashboard/pulmonology",
    status: "coming-soon",
    category: "upcoming",
    diagnoses: ["COPD", "Asthma", "Sleep apnea"],
    howItWorks: "Audio analysis of breathing sounds",
    gradient: "from-cyan-400 to-blue-400",
  },
  {
    id: "diabetic-foot",
    name: "FootCare AI",
    description: "Diabetic foot ulcer assessment",
    icon: Footprints,
    route: "/dashboard/diabetic-foot",
    status: "coming-soon",
    category: "upcoming",
    diagnoses: ["DFU", "Wound severity", "Infection risk"],
    howItWorks: "Upload smartphone photo of foot/wound",
    gradient: "from-amber-400 to-orange-400",
  },
  {
    id: "orthopedics",
    name: "BoneScan AI",
    description: "Bone fracture and arthritis detection",
    icon: Bone,
    route: "/dashboard/orthopedics",
    status: "coming-soon",
    category: "upcoming",
    diagnoses: ["Fractures", "Osteoporosis", "Arthritis"],
    howItWorks: "Upload X-ray for fracture detection",
    gradient: "from-stone-400 to-slate-400",
  },
];

/**
 * Get all available (implemented) diagnostic modules
 */
export const getAvailableModules = (): DiagnosticModule[] => {
  return diagnosticModules.filter((module) => module.status === "available");
};

/**
 * Get all coming soon (not yet implemented) diagnostic modules
 */
export const getComingSoonModules = (): DiagnosticModule[] => {
  return diagnosticModules.filter((module) => module.status === "coming-soon");
};

/**
 * Get a diagnostic module by its ID
 */
export const getModuleById = (id: string): DiagnosticModule | undefined => {
  return diagnosticModules.find((module) => module.id === id);
};

/**
 * Get modules by category
 */
export const getModulesByCategory = (
  category: "current" | "upcoming",
): DiagnosticModule[] => {
  return diagnosticModules.filter((module) => module.category === category);
};

/**
 * Total count of all diagnostic modules
 */
export const TOTAL_MODULES_COUNT = diagnosticModules.length;

/**
 * Count of available modules
 */
export const AVAILABLE_MODULES_COUNT = getAvailableModules().length;

/**
 * Count of coming soon modules
 */
export const COMING_SOON_MODULES_COUNT = getComingSoonModules().length;

export default diagnosticModules;
