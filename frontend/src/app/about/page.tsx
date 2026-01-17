import { Metadata } from 'next';
import AboutPageClient from '@/components/pages/AboutPageClient';
import { Layout } from '@/components/layout';

export const metadata: Metadata = {
  title: 'Technical Architecture & Engineering | MediLens AI Platform',
  description: 'Comprehensive technical overview of MediLens: 11 AI diagnostic modules, modern microservices architecture, ML pipelines, and enterprise-grade security. Built with Next.js 15, PyTorch, and cloud-native infrastructure.',
};

export default function AboutPage() {
  return (
    <Layout containerized={false}>
      <AboutPageClient />
    </Layout>
  );
}
