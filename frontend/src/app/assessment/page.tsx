import { AssessmentFlow } from '@/components/assessment/AssessmentFlow';
import { Layout } from '@/components/layout';

import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Neurological Assessment',
  description: 'Complete your multi-modal neurological risk assessment with NeuroLens-X.',
};

export default function AssessmentPage() {
  return (
    <Layout showHeader={false} showFooter={false} containerized={false}>
      <AssessmentFlow />
    </Layout>
  );
}
