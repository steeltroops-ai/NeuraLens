import type { Metadata } from 'next';
import { ValidationDashboard } from '@/components/validation/ValidationDashboard';
import { Layout } from '@/components/layout';

export const metadata: Metadata = {
  title: 'Clinical Validation',
  description: 'NeuroLens-X clinical validation results, performance metrics, and accuracy data.',
};

export default function ValidationPage() {
  return (
    <Layout>
      <div className="py-8">
        <ValidationDashboard />
      </div>
    </Layout>
  );
}
