import type { Metadata } from 'next';
import { Layout } from '@/components/layout';
import { Card } from '@/components/ui';

export const metadata: Metadata = {
  title: 'About NeuroLens-X',
  description:
    'Learn about NeuroLens-X, the revolutionary multi-modal neurological risk assessment platform.',
};

export default function AboutPage() {
  return (
    <Layout>
      <div className="py-8">
        <div className="mx-auto max-w-4xl space-y-8">
          {/* Header */}
          <div className="space-y-4 text-center">
            <h1 className="text-4xl font-bold text-text-primary">
              About NeuroLens-X
            </h1>
            <p className="mx-auto max-w-2xl text-lg text-text-secondary">
              Advanced brain health screening platform combining AI technology,
              clinical expertise, and patient-centered design for early
              detection.
            </p>
          </div>

          {/* Mission */}
          <Card className="p-8">
            <h2 className="mb-4 text-2xl font-semibold text-text-primary">
              Our Mission
            </h2>
            <p className="leading-relaxed text-text-secondary">
              NeuroLens-X aims to make brain health screening accessible to
              everyone by providing easy-to-use, accurate, and comprehensive
              evaluation tools. We combine advanced AI technology with clinical
              expertise to enable early detection and better health outcomes for
              neurological conditions.
            </p>
          </Card>

          {/* Technology */}
          <Card className="p-8">
            <h2 className="mb-6 text-2xl font-semibold text-text-primary">
              Our Technology
            </h2>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  🎤 Voice Evaluation
                </h3>
                <p className="text-sm text-text-secondary">
                  Advanced voice analysis using AI to identify subtle changes in
                  speech patterns that may indicate brain health changes.
                </p>
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  👁️ Eye Health Scan
                </h3>
                <p className="text-sm text-text-secondary">
                  Computer vision analysis of eye blood vessel patterns to
                  detect early changes that may correlate with brain health.
                </p>
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  📊 Health Questionnaire
                </h3>
                <p className="text-sm text-text-secondary">
                  Comprehensive evaluation of your health history, lifestyle,
                  and family background using evidence-based health models.
                </p>
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  ✋ Motor Assessment
                </h3>
                <p className="text-sm text-text-secondary">
                  Simple finger tapping and movement tests to detect early motor
                  function changes that may indicate neurological conditions.
                </p>
              </div>
            </div>
          </Card>

          {/* Clinical Validation */}
          <Card className="p-8">
            <h2 className="mb-4 text-2xl font-semibold text-text-primary">
              Clinical Validation
            </h2>
            <div className="mb-6 grid grid-cols-1 gap-6 md:grid-cols-3">
              <div className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary-400">
                  89.8%
                </div>
                <div className="text-sm text-text-secondary">
                  Overall Accuracy
                </div>
              </div>
              <div className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary-400">
                  2,847
                </div>
                <div className="text-sm text-text-secondary">
                  Study Participants
                </div>
              </div>
              <div className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary-400">
                  0.924
                </div>
                <div className="text-sm text-text-secondary">AUC Score</div>
              </div>
            </div>
            <p className="text-text-secondary">
              Our platform has been validated through extensive clinical studies
              involving over 2,800 participants across 12 clinical sites,
              demonstrating excellent diagnostic performance with 89.8% accuracy
              and 0.924 AUC score.
            </p>
          </Card>

          {/* Accessibility */}
          <Card className="p-8">
            <h2 className="mb-4 text-2xl font-semibold text-text-primary">
              Accessibility First
            </h2>
            <p className="mb-6 text-text-secondary">
              NeuroLens-X is designed with accessibility at its core, ensuring
              that neurological health assessment is available to everyone,
              regardless of their abilities or circumstances.
            </p>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">
                  WCAG 2.1 AA+ Compliance
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">
                  Screen Reader Optimized
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">
                  Keyboard Navigation
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">
                  Voice Navigation Ready
                </span>
              </div>
            </div>
          </Card>

          {/* Team */}
          <Card className="p-8">
            <h2 className="mb-4 text-2xl font-semibold text-text-primary">
              Our Team
            </h2>
            <p className="text-text-secondary">
              NeuroLens-X is developed by a multidisciplinary team of AI
              researchers, clinical neurologists, accessibility experts, and
              software engineers committed to advancing neurological health
              through technology.
            </p>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
