import type { Metadata } from 'next';
import { Layout } from '@/components/layout';
import { Card } from '@/components/ui';

export const metadata: Metadata = {
  title: 'About NeuroLens-X',
  description: 'Learn about NeuroLens-X, the revolutionary multi-modal neurological risk assessment platform.',
};

export default function AboutPage() {
  return (
    <Layout>
      <div className="py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold text-text-primary">
              About NeuroLens-X
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto">
              Revolutionary multi-modal neurological risk assessment platform combining AI, 
              clinical expertise, and accessibility-first design.
            </p>
          </div>

          {/* Mission */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-4">
              Our Mission
            </h2>
            <p className="text-text-secondary leading-relaxed">
              NeuroLens-X aims to democratize neurological health assessment by providing 
              accessible, accurate, and comprehensive risk evaluation tools. We combine 
              cutting-edge AI technology with clinical expertise to enable early detection 
              and intervention for neurological conditions.
            </p>
          </Card>

          {/* Technology */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-6">
              Our Technology
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  🎤 Speech Analysis
                </h3>
                <p className="text-sm text-text-secondary">
                  Advanced voice biomarker detection using machine learning to identify 
                  subtle changes in speech patterns that may indicate neurological conditions.
                </p>
              </div>
              
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  👁️ Retinal Imaging
                </h3>
                <p className="text-sm text-text-secondary">
                  Computer vision analysis of retinal vascular patterns to detect early 
                  pathological changes that correlate with brain health.
                </p>
              </div>
              
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  📊 Risk Assessment
                </h3>
                <p className="text-sm text-text-secondary">
                  Comprehensive evaluation of demographic, medical, lifestyle, and genetic 
                  factors using evidence-based risk models.
                </p>
              </div>
              
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-text-primary">
                  🧠 AI Fusion
                </h3>
                <p className="text-sm text-text-secondary">
                  Multi-modal ensemble learning that combines all assessment types into 
                  a unified Neuro-Risk Index with uncertainty quantification.
                </p>
              </div>
            </div>
          </Card>

          {/* Clinical Validation */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-4">
              Clinical Validation
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary-400 mb-2">89.8%</div>
                <div className="text-sm text-text-secondary">Overall Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary-400 mb-2">2,847</div>
                <div className="text-sm text-text-secondary">Study Participants</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary-400 mb-2">0.924</div>
                <div className="text-sm text-text-secondary">AUC Score</div>
              </div>
            </div>
            <p className="text-text-secondary">
              Our platform has been validated through extensive clinical studies involving 
              over 2,800 participants across 12 clinical sites, demonstrating excellent 
              diagnostic performance with 89.8% accuracy and 0.924 AUC score.
            </p>
          </Card>

          {/* Accessibility */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-4">
              Accessibility First
            </h2>
            <p className="text-text-secondary mb-6">
              NeuroLens-X is designed with accessibility at its core, ensuring that 
              neurological health assessment is available to everyone, regardless of 
              their abilities or circumstances.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">WCAG 2.1 AA+ Compliance</span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">Screen Reader Optimized</span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">Keyboard Navigation</span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-success">✓</span>
                <span className="text-sm text-text-secondary">Voice Navigation Ready</span>
              </div>
            </div>
          </Card>

          {/* Team */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-4">
              Our Team
            </h2>
            <p className="text-text-secondary">
              NeuroLens-X is developed by a multidisciplinary team of AI researchers, 
              clinical neurologists, accessibility experts, and software engineers 
              committed to advancing neurological health through technology.
            </p>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
