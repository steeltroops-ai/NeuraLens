import { Metadata } from 'next';
import AboutPageClient from '@/components/pages/AboutPageClient';
import { Layout } from '@/components/layout';

export const metadata: Metadata = {
  title: 'Architecture & Engineering | MediLens',
  description: 'Technical deep dive into the MediLens diagnostic platform, featuring system architecture, ML pipelines, and development roadmap.',
};

export default function AboutPage() {
  return (
    <Layout containerized={false}>
      <AboutPageClient />
    </Layout>
  );
}
