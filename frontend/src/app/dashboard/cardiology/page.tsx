import { Metadata } from 'next';
import { CardiologyAssessment } from './_components/CardiologyAssessment';

export const metadata: Metadata = {
    title: 'CardioPredict AI - Cardiology Diagnostics | MediLens',
    description: 'AI-powered ECG analysis for arrhythmia, atrial fibrillation, myocardial infarction, and cardiac conditions',
};

export default function CardiologyPage() {
    return <CardiologyAssessment />;
}
