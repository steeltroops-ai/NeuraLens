import { Metadata } from 'next';
import { DermatologyAssessment } from './_components/DermatologyAssessment';

export const metadata: Metadata = {
    title: 'SkinSense AI - Dermatology Diagnostics | MediLens',
    description: 'AI-powered skin lesion analysis for melanoma, skin cancer, and dermatological conditions',
};

export default function DermatologyPage() {
    return <DermatologyAssessment />;
}
