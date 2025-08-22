import type { Metadata } from 'next';
import { ResultsDashboard } from '@/components/results/ResultsDashboard';
import { Layout } from '@/components/layout';

export const metadata: Metadata = {
  title: 'Assessment Results',
  description: 'View your comprehensive neurological risk assessment results from NeuroLens-X.',
};

export default function ResultsPage() {
  return (
    <Layout>
      <div className="py-8">
        <ResultsDashboard />
      </div>
    </Layout>
  );
}
