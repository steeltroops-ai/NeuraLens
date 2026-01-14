/**
 * Diagnostic Modules Configuration
 * 
 * Defines all 11 diagnostic modules for the MediLens platform.
 * Each module includes status (available/coming-soon), icons, routes,
 * descriptions, diagnoses, and MediLens design system colors.
 * 
 * Requirements: 3.1, 3.4, 3.5
 */

import { LucideIcon, Mic, Eye, Hand, Brain, Activity, Zap, Scan, Microscope, Heart, Wind, Footprints, Bone } from 'lucide-react';

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
    status: 'available' | 'coming-soon';
    /** Category for grouping modules */
    category: 'current' | 'upcoming';
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
 * Current Available Modules (4):
 * - Speech Analysis (SpeechMD AI)
 * - Retinal Imaging (RetinaScan AI)
 * - Motor Assessment
 * - Cognitive Testing
 * 
 * Coming Soon Modules (7):
 * - Radiology (ChestXplorer AI)
 * - Dermatology (SkinSense AI)
 * - Pathology (HistoVision AI)
 * - Cardiology (CardioPredict AI)
 * - Neurology (NeuroScan AI)
 * - Pulmonology (RespiRate AI)
 * - Diabetic Foot Care (FootCare AI)
 * - Orthopedics (BoneScan AI)
 */
export const diagnosticModules: DiagnosticModule[] = [
    // ============================================
    // CURRENT AVAILABLE MODULES
    // ============================================
    {
        id: 'speech',
        name: 'SpeechMD AI',
        description: 'Voice tremor analysis for Parkinson\'s detection',
        icon: Mic,
        route: '/dashboard/speech',
        status: 'available',
        category: 'current',
        diagnoses: ['Parkinson\'s disease', 'Aphasia', 'Early dementia', 'Depression/anxiety'],
        howItWorks: 'Record patient speech for acoustic feature extraction',
        gradient: 'from-medilens-blue-500 to-cyan-500',
    },
    {
        id: 'retinal',
        name: 'RetinaScan AI',
        description: 'Retinal imaging for neurological indicators',
        icon: Eye,
        route: '/dashboard/retinal',
        status: 'available',
        category: 'current',
        diagnoses: ['Diabetic Retinopathy', 'Glaucoma', 'AMD', 'Hypertensive Retinopathy'],
        howItWorks: 'Upload retinal fundus images for AI analysis',
        gradient: 'from-emerald-500 to-teal-500',
    },
    {
        id: 'motor',
        name: 'Motor Assessment',
        description: 'Movement pattern and tremor detection',
        icon: Hand,
        route: '/dashboard/motor',
        status: 'available',
        category: 'current',
        diagnoses: ['Bradykinesia', 'Tremor', 'Coordination issues'],
        howItWorks: 'Interactive finger tapping and coordination tests',
        gradient: 'from-purple-500 to-indigo-500',
    },
    {
        id: 'cognitive',
        name: 'Cognitive Testing',
        description: 'Memory and executive function assessment',
        icon: Brain,
        route: '/dashboard/cognitive',
        status: 'available',
        category: 'current',
        diagnoses: ['MCI', 'Memory impairment', 'Executive dysfunction'],
        howItWorks: 'Comprehensive cognitive evaluation tasks',
        gradient: 'from-orange-500 to-red-500',
    },

    // ============================================
    // COMING SOON MODULES
    // ============================================
    {
        id: 'radiology',
        name: 'ChestXplorer AI',
        description: 'Chest X-ray and bone fracture analysis',
        icon: Scan,
        route: '/dashboard/radiology',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['Pneumonia', 'TB', 'Lung cancer', 'Bone fractures'],
        howItWorks: 'Upload X-ray for multi-class classification',
        gradient: 'from-slate-400 to-gray-400',
    },
    {
        id: 'dermatology',
        name: 'SkinSense AI',
        description: 'Skin lesion classification and cancer detection',
        icon: Scan,
        route: '/dashboard/dermatology',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['Melanoma', 'Basal cell carcinoma', 'Eczema', 'Psoriasis'],
        howItWorks: 'Upload smartphone photo of skin lesion',
        gradient: 'from-pink-400 to-rose-400',
    },
    {
        id: 'pathology',
        name: 'HistoVision AI',
        description: 'Tissue sample and blood smear analysis',
        icon: Microscope,
        route: '/dashboard/pathology',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['Cancer detection', 'Malaria', 'Leukemia'],
        howItWorks: 'Upload microscopy images for cell analysis',
        gradient: 'from-violet-400 to-purple-400',
    },
    {
        id: 'cardiology',
        name: 'CardioPredict AI',
        description: 'ECG and heart sound analysis',
        icon: Heart,
        route: '/dashboard/cardiology',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['Arrhythmia', 'AFib', 'Heart murmur', 'MI'],
        howItWorks: 'Upload ECG signal or heart sound recording',
        gradient: 'from-red-400 to-pink-400',
    },
    {
        id: 'neurology',
        name: 'NeuroScan AI',
        description: 'Brain MRI and CT scan analysis',
        icon: Brain,
        route: '/dashboard/neurology',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['Brain tumors', 'Stroke', 'MS lesions', 'Hemorrhage'],
        howItWorks: 'Upload MRI or CT scan (DICOM support)',
        gradient: 'from-indigo-400 to-blue-400',
    },
    {
        id: 'pulmonology',
        name: 'RespiRate AI',
        description: 'Respiratory sound and spirometry analysis',
        icon: Wind,
        route: '/dashboard/pulmonology',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['COPD', 'Asthma', 'Sleep apnea'],
        howItWorks: 'Audio analysis of breathing sounds',
        gradient: 'from-cyan-400 to-blue-400',
    },
    {
        id: 'diabetic-foot',
        name: 'FootCare AI',
        description: 'Diabetic foot ulcer assessment',
        icon: Footprints,
        route: '/dashboard/diabetic-foot',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['DFU', 'Wound severity', 'Infection risk'],
        howItWorks: 'Upload smartphone photo of foot/wound',
        gradient: 'from-amber-400 to-orange-400',
    },
    {
        id: 'orthopedics',
        name: 'BoneScan AI',
        description: 'Bone fracture and arthritis detection',
        icon: Bone,
        route: '/dashboard/orthopedics',
        status: 'coming-soon',
        category: 'upcoming',
        diagnoses: ['Fractures', 'Osteoporosis', 'Arthritis'],
        howItWorks: 'Upload X-ray for fracture detection',
        gradient: 'from-stone-400 to-slate-400',
    },
];

/**
 * Get all available (implemented) diagnostic modules
 */
export const getAvailableModules = (): DiagnosticModule[] => {
    return diagnosticModules.filter(module => module.status === 'available');
};

/**
 * Get all coming soon (not yet implemented) diagnostic modules
 */
export const getComingSoonModules = (): DiagnosticModule[] => {
    return diagnosticModules.filter(module => module.status === 'coming-soon');
};

/**
 * Get a diagnostic module by its ID
 */
export const getModuleById = (id: string): DiagnosticModule | undefined => {
    return diagnosticModules.find(module => module.id === id);
};

/**
 * Get modules by category
 */
export const getModulesByCategory = (category: 'current' | 'upcoming'): DiagnosticModule[] => {
    return diagnosticModules.filter(module => module.category === category);
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
