'use client';

import { Button } from '@/components/ui';
import { Layout } from '@/components/layout';

export default function HomePage() {
  return (
    <Layout containerized={false}>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-surface-background via-surface-primary to-surface-secondary">
        <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 via-transparent to-primary-500/5" />

        <div className="container mx-auto px-4 py-20 lg:py-32">
          <div className="mx-auto max-w-4xl space-y-8 text-center">
            {/* Main Heading */}
            <div className="space-y-4">
              <h1 className="text-gradient-primary text-5xl font-black leading-tight lg:text-7xl">
                NeuroLens-X
              </h1>
              <p className="mx-auto max-w-3xl text-xl font-light text-text-secondary lg:text-2xl">
                Early Neurological Risk Detection
              </p>
              <div className="mx-auto h-1 w-24 rounded-full bg-gradient-to-r from-primary-500 to-primary-600" />
            </div>

            {/* Value Proposition */}
            <div className="space-y-6">
              <p className="mx-auto max-w-2xl text-lg leading-relaxed text-text-primary lg:text-xl">
                Detect neurological decline{' '}
                <span className="font-semibold text-primary-400">
                  5-10 years
                </span>{' '}
                before symptoms appear through AI-powered multi-modal assessment
              </p>

              <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
                <Button
                  size="lg"
                  className="w-full px-8 py-4 text-lg font-semibold sm:w-auto"
                  onClick={() => {
                    if (typeof window !== 'undefined') {
                      window.location.href = '/assessment';
                    }
                  }}
                >
                  Start Assessment
                  <svg
                    className="ml-2 h-5 w-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 7l5 5m0 0l-5 5m5-5H6"
                    />
                  </svg>
                </Button>

                <Button
                  variant="secondary"
                  size="lg"
                  className="w-full px-8 py-4 text-lg sm:w-auto"
                  onClick={() => {
                    if (typeof window !== 'undefined') {
                      window.location.href = '/about';
                    }
                  }}
                >
                  Learn More
                </Button>
              </div>
            </div>

            {/* Trust Indicators */}
            <div className="space-y-4 pt-8">
              <p className="text-sm font-medium uppercase tracking-wider text-text-muted">
                Clinical Validation
              </p>
              <div className="flex flex-wrap justify-center gap-6 text-sm text-text-secondary">
                <div className="flex items-center gap-2">
                  <svg
                    className="h-4 w-4 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  85% Sensitivity
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="h-4 w-4 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  90% Specificity
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="h-4 w-4 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  WCAG 2.1 AA
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="h-4 w-4 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  HIPAA Compliant
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div
            className="absolute inset-0 bg-gradient-to-r from-primary-500 to-transparent"
            style={{
              backgroundImage: `radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
                                  radial-gradient(circle at 75% 75%, rgba(59, 130, 246, 0.1) 0%, transparent 50%)`,
            }}
          />
        </div>
      </section>

      {/* Features Overview */}
      <section className="bg-surface-primary py-20">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-6xl">
            {/* Section Header */}
            <div className="mb-16 space-y-4 text-center">
              <h2 className="text-3xl font-bold text-text-primary lg:text-4xl">
                Multi-Modal Assessment
              </h2>
              <p className="mx-auto max-w-2xl text-lg text-text-secondary">
                Comprehensive neurological risk evaluation through four advanced
                assessment modalities
              </p>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
              {/* Speech Analysis */}
              <div className="group">
                <div className="clinical-card h-full p-6 transition-all duration-300 hover:border-primary-500/50">
                  <div className="space-y-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-500/10 transition-colors group-hover:bg-primary-500/20">
                      <svg
                        className="h-6 w-6 text-primary-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                        />
                      </svg>
                    </div>
                    <div>
                      <h3 className="mb-2 text-xl font-semibold text-text-primary">
                        Speech Analysis
                      </h3>
                      <p className="text-sm leading-relaxed text-text-secondary">
                        Voice biomarker detection through advanced speech
                        pattern analysis and micro-tremor identification
                      </p>
                    </div>
                    <div className="text-xs font-medium text-primary-400">
                      ~2 minutes
                    </div>
                  </div>
                </div>
              </div>

              {/* Retinal Imaging */}
              <div className="group">
                <div className="clinical-card h-full p-6 transition-all duration-300 hover:border-primary-500/50">
                  <div className="space-y-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-500/10 transition-colors group-hover:bg-primary-500/20">
                      <svg
                        className="h-6 w-6 text-primary-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                        />
                      </svg>
                    </div>
                    <div>
                      <h3 className="mb-2 text-xl font-semibold text-text-primary">
                        Retinal Imaging
                      </h3>
                      <p className="text-sm leading-relaxed text-text-secondary">
                        Vascular pattern analysis for early pathological changes
                        in retinal blood vessels
                      </p>
                    </div>
                    <div className="text-xs font-medium text-primary-400">
                      ~1 minute
                    </div>
                  </div>
                </div>
              </div>

              {/* Risk Assessment */}
              <div className="group">
                <div className="clinical-card h-full p-6 transition-all duration-300 hover:border-primary-500/50">
                  <div className="space-y-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-500/10 transition-colors group-hover:bg-primary-500/20">
                      <svg
                        className="h-6 w-6 text-primary-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                        />
                      </svg>
                    </div>
                    <div>
                      <h3 className="mb-2 text-xl font-semibold text-text-primary">
                        Risk Assessment
                      </h3>
                      <p className="text-sm leading-relaxed text-text-secondary">
                        Comprehensive health and lifestyle questionnaire with
                        personalized risk calculation
                      </p>
                    </div>
                    <div className="text-xs font-medium text-primary-400">
                      ~3 minutes
                    </div>
                  </div>
                </div>
              </div>

              {/* NRI Fusion */}
              <div className="group">
                <div className="clinical-card h-full p-6 transition-all duration-300 hover:border-primary-500/50">
                  <div className="space-y-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-500/10 transition-colors group-hover:bg-primary-500/20">
                      <svg
                        className="h-6 w-6 text-primary-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                        />
                      </svg>
                    </div>
                    <div>
                      <h3 className="mb-2 text-xl font-semibold text-text-primary">
                        NRI Calculation
                      </h3>
                      <p className="text-sm leading-relaxed text-text-secondary">
                        Unified Neuro-Risk Index combining all modalities with
                        uncertainty quantification
                      </p>
                    </div>
                    <div className="text-xs font-medium text-primary-400">
                      ~30 seconds
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Total Time */}
            <div className="mt-12 text-center">
              <div className="inline-flex items-center gap-2 rounded-full bg-primary-500/10 px-4 py-2">
                <svg
                  className="h-4 w-4 text-primary-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span className="text-sm font-medium text-primary-400">
                  Total Assessment Time: ~6.5 minutes
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="bg-gradient-to-r from-primary-500/10 via-primary-600/5 to-primary-500/10 py-20">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-4xl space-y-8 text-center">
            <h2 className="text-3xl font-bold text-text-primary lg:text-4xl">
              Ready to Assess Your Neurological Health?
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-text-secondary">
              Take the first step towards early detection and proactive
              neurological health management.
            </p>
            <div className="flex flex-col justify-center gap-4 sm:flex-row">
              <Button
                size="xl"
                className="px-12 py-4 text-lg font-semibold"
                onClick={() => {
                  if (typeof window !== 'undefined') {
                    window.location.href = '/assessment';
                  }
                }}
              >
                Start Your Assessment
                <svg
                  className="ml-2 h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7l5 5m0 0l-5 5m5-5H6"
                  />
                </svg>
              </Button>
            </div>

            {/* Privacy Notice */}
            <p className="mx-auto max-w-xl text-sm text-text-muted">
              <svg
                className="mr-1 inline h-4 w-4"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                  clipRule="evenodd"
                />
              </svg>
              Your data is processed locally in your browser for maximum privacy
              and security.
            </p>
          </div>
        </div>
      </section>
    </Layout>
  );
}
