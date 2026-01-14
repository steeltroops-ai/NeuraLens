import { AssessmentFlow } from '@/components/assessment/AssessmentFlow';
import { Layout } from '@/components/layout';

import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Medical Assessment',
  description: 'Complete your multi-modal medical risk assessment with MediLens.',
};

export default function AssessmentPage() {
  return (
    <Layout showHeader={false} showFooter={false} containerized={false}>
      <AssessmentFlow />
    </Layout>
  );
}
