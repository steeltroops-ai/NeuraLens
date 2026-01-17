import { Metadata } from 'next';
import { RadiologyAssessment } from './_components/RadiologyAssessment';

export const metadata: Metadata = {
    title: 'ChestXplorer AI - Radiology Diagnostics | MediLens',
    description: 'AI-powered chest X-ray analysis for pneumonia, COVID-19, tuberculosis, lung cancer, and more',
};

export default function RadiologyPage() {
    return <RadiologyAssessment />;
}
