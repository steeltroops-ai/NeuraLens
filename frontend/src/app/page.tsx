'use client';

import { Button } from '@/components/ui';
import { Layout } from '@/components/layout';

export default function HomePage() {
  return (
    <Layout containerized={false}>
      {/* Apple-Inspired Hero Section */}
      <section className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-gray-50 via-white to-gray-100">
        {/* Floating Brain Icon with 3D Effect */}
        <div className="absolute right-20 top-20 animate-pulse-slow opacity-10">
          <div className="h-32 w-32 rotate-12 transform rounded-full bg-gradient-to-br from-medical-500 to-neural-500 shadow-2xl"></div>
        </div>

        <div className="absolute bottom-20 left-20 animate-bounce-gentle opacity-5">
          <div className="h-24 w-24 -rotate-12 transform rounded-full bg-gradient-to-br from-success-500 to-medical-500 shadow-xl"></div>
        </div>

        <div className="container mx-auto px-6 py-20 lg:py-32">
          <div className="mx-auto max-w-4xl space-y-12 text-center">
            {/* Main Heading - Apple Style */}
            <div className="animate-fade-in space-y-6">
              <h1 className="text-6xl font-black leading-tight tracking-tight text-gradient-medical lg:text-8xl">
                NeuroLens-X
              </h1>
              <p className="mx-auto max-w-3xl text-2xl font-light text-text-secondary lg:text-3xl">
                Quick & Easy Brain Health Screening
              </p>
              <p className="mx-auto max-w-2xl text-lg text-text-tertiary">
                Trusted by healthcare professionals • Privacy protected •
                Clinically validated
              </p>
            </div>

            {/* Value Proposition - Clean & Minimal */}
            <div className="animate-slide-up space-y-8">
              <p className="mx-auto max-w-2xl text-xl leading-relaxed text-text-primary">
                Advanced technology helps identify potential brain health
                changes{' '}
                <span className="bg-clip-text font-semibold text-transparent text-gradient-medical">
                  early for better outcomes
                </span>
              </p>

              {/* Apple-Style CTA Buttons */}
              <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
                <Button
                  variant="primary"
                  size="lg"
                  className="w-full min-w-[200px] shadow-medical hover:shadow-medical-hover sm:w-auto"
                  onClick={() => {
                    if (typeof window !== 'undefined') {
                      window.location.href = '/assessment';
                    }
                  }}
                >
                  Start Health Check
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
                  className="w-full min-w-[200px] sm:w-auto"
                  onClick={() => {
                    if (typeof window !== 'undefined') {
                      window.location.href = '/validation';
                    }
                  }}
                >
                  View Validation
                </Button>
              </div>
            </div>

            {/* Apple-Style Trust Indicators */}
            <div className="animate-scale-in space-y-6 pt-12">
              <p className="text-sm font-medium uppercase tracking-wider text-text-tertiary">
                Validated by 14 Clinical Experts
              </p>

              {/* Trust Badges - Apple Card Style */}
              <div className="flex flex-wrap justify-center gap-4">
                <div className="group flex items-center gap-3 rounded-apple border border-gray-200/50 bg-white/80 px-4 py-3 shadow-apple backdrop-blur-sm transition-all duration-200 hover:-translate-y-1 hover:scale-105 hover:shadow-apple-hover">
                  <div className="h-2 w-2 rounded-full bg-success-500 transition-colors duration-200 group-hover:bg-success-600"></div>
                  <span className="text-sm font-medium text-text-primary transition-colors duration-200 group-hover:text-success-600">
                    85.2% Sensitivity
                  </span>
                </div>
                <div className="group flex items-center gap-3 rounded-apple border border-gray-200/50 bg-white/80 px-4 py-3 shadow-apple backdrop-blur-sm transition-all duration-200 hover:-translate-y-1 hover:scale-105 hover:shadow-apple-hover">
                  <div className="h-2 w-2 rounded-full bg-success-500 transition-colors duration-200 group-hover:bg-success-600"></div>
                  <span className="text-sm font-medium text-text-primary transition-colors duration-200 group-hover:text-success-600">
                    89.7% Specificity
                  </span>
                </div>
                <div className="group flex items-center gap-3 rounded-apple border border-gray-200/50 bg-white/80 px-4 py-3 shadow-apple backdrop-blur-sm transition-all duration-200 hover:-translate-y-1 hover:scale-105 hover:shadow-apple-hover">
                  <div className="h-2 w-2 rounded-full bg-medical-500 transition-colors duration-200 group-hover:bg-medical-600"></div>
                  <span className="text-sm font-medium text-text-primary transition-colors duration-200 group-hover:text-medical-600">
                    WCAG 2.1 AA+
                  </span>
                </div>
                <div className="group flex items-center gap-3 rounded-apple border border-gray-200/50 bg-white/80 px-4 py-3 shadow-apple backdrop-blur-sm transition-all duration-200 hover:-translate-y-1 hover:scale-105 hover:shadow-apple-hover">
                  <div className="h-2 w-2 rounded-full bg-medical-500 transition-colors duration-200 group-hover:bg-medical-600"></div>
                  <span className="text-sm font-medium text-text-primary transition-colors duration-200 group-hover:text-medical-600">
                    HIPAA Compliant
                  </span>
                </div>
              </div>

              {/* University Logos Placeholder */}
              <div className="flex items-center justify-center gap-8 opacity-60">
                <div className="text-xs font-medium text-text-tertiary">
                  Johns Hopkins
                </div>
                <div className="h-1 w-1 rounded-full bg-text-tertiary"></div>
                <div className="text-xs font-medium text-text-tertiary">
                  Mayo Clinic
                </div>
                <div className="h-1 w-1 rounded-full bg-text-tertiary"></div>
                <div className="text-xs font-medium text-text-tertiary">
                  Stanford Medicine
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

      {/* Apple-Style Features Overview */}
      <section className="bg-gray-50 py-24">
        <div className="container mx-auto px-6">
          <div className="mx-auto max-w-7xl">
            {/* Section Header - Apple Style */}
            <div className="mb-20 animate-fade-in space-y-6 text-center">
              <h2 className="text-4xl font-bold tracking-tight text-text-primary lg:text-5xl">
                Four Simple Health Checks
              </h2>
              <p className="mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary">
                Comprehensive brain health evaluation in under 90 seconds
              </p>
            </div>

            {/* Apple-Style Features Grid */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
              {/* Speech Analysis - Apple Card Style */}
              <div className="group animate-slide-up">
                <div className="card-apple h-full p-8 transition-all duration-300 hover:-translate-y-2 hover:shadow-apple-hover">
                  <div className="space-y-6">
                    <div className="flex h-16 w-16 items-center justify-center rounded-apple-lg bg-medical-50 transition-all duration-300 group-hover:scale-110 group-hover:bg-medical-100">
                      <svg
                        className="h-8 w-8 text-medical-500"
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
                    <div className="space-y-3">
                      <h3 className="text-xl font-semibold text-text-primary">
                        Voice Evaluation
                      </h3>
                      <p className="text-base leading-relaxed text-text-secondary">
                        Simple voice recording to detect early signs of
                        neurological changes
                      </p>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="rounded-full bg-medical-50 px-3 py-1 text-sm font-medium text-medical-500">
                        30 seconds
                      </span>
                      <div className="h-2 w-2 rounded-full bg-success-500"></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Retinal Imaging - Apple Card Style */}
              <div
                className="group animate-slide-up"
                style={{ animationDelay: '0.1s' }}
              >
                <div className="card-apple h-full p-8 transition-all duration-300 hover:-translate-y-2 hover:shadow-apple-hover">
                  <div className="space-y-6">
                    <div className="flex h-16 w-16 items-center justify-center rounded-apple-lg bg-neural-50 transition-all duration-300 group-hover:scale-110 group-hover:bg-neural-100">
                      <svg
                        className="h-8 w-8 text-neural-500"
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
                    <div className="space-y-3">
                      <h3 className="text-xl font-semibold text-text-primary">
                        Eye Health Scan
                      </h3>
                      <p className="text-base leading-relaxed text-text-secondary">
                        Quick eye photo analysis to check blood vessel health
                      </p>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="rounded-full bg-neural-50 px-3 py-1 text-sm font-medium text-neural-500">
                        15 seconds
                      </span>
                      <div className="h-2 w-2 rounded-full bg-success-500"></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Risk Assessment - Apple Card Style */}
              <div
                className="group animate-slide-up"
                style={{ animationDelay: '0.2s' }}
              >
                <div className="card-apple h-full p-8 transition-all duration-300 hover:-translate-y-2 hover:shadow-apple-hover">
                  <div className="space-y-6">
                    <div className="flex h-16 w-16 items-center justify-center rounded-apple-lg bg-warning-50 transition-all duration-300 group-hover:scale-110 group-hover:bg-warning-100">
                      <svg
                        className="h-8 w-8 text-warning-500"
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
                    <div className="space-y-3">
                      <h3 className="text-xl font-semibold text-text-primary">
                        Health Questionnaire
                      </h3>
                      <p className="text-base leading-relaxed text-text-secondary">
                        Simple questions about your health and lifestyle
                      </p>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="rounded-full bg-warning-50 px-3 py-1 text-sm font-medium text-warning-500">
                        30 seconds
                      </span>
                      <div className="h-2 w-2 rounded-full bg-success-500"></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Motor Assessment - Apple Card Style */}
              <div
                className="group animate-slide-up"
                style={{ animationDelay: '0.3s' }}
              >
                <div className="card-apple h-full p-8 transition-all duration-300 hover:-translate-y-2 hover:shadow-apple-hover">
                  <div className="space-y-6">
                    <div className="flex h-16 w-16 items-center justify-center rounded-apple-lg bg-success-50 transition-all duration-300 group-hover:scale-110 group-hover:bg-success-100">
                      <svg
                        className="h-8 w-8 text-success-500"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11"
                        />
                      </svg>
                    </div>
                    <div className="space-y-3">
                      <h3 className="text-xl font-semibold text-text-primary">
                        Movement Check
                      </h3>
                      <p className="text-base leading-relaxed text-text-secondary">
                        Simple finger tapping test to assess motor function
                      </p>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="rounded-full bg-success-50 px-3 py-1 text-sm font-medium text-success-500">
                        15 seconds
                      </span>
                      <div className="h-2 w-2 rounded-full bg-success-500"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Apple-Style Total Time Badge */}
            <div className="mt-16 animate-scale-in text-center">
              <div className="inline-flex items-center gap-3 rounded-apple-xl border border-gray-200/50 bg-white/80 px-6 py-4 shadow-apple backdrop-blur-sm">
                <div className="flex h-10 w-10 items-center justify-center rounded-apple bg-medical-50">
                  <svg
                    className="h-5 w-5 text-medical-500"
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
                </div>
                <div className="text-left">
                  <div className="text-lg font-semibold text-text-primary">
                    Total Screening Time
                  </div>
                  <div className="text-sm text-text-secondary">
                    Under 90 seconds
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Apple-Style Call to Action */}
      <section className="bg-white py-24">
        <div className="container mx-auto px-6">
          <div className="mx-auto max-w-4xl space-y-12 text-center">
            {/* Headline */}
            <div className="animate-fade-in space-y-6">
              <h2 className="text-4xl font-bold tracking-tight text-text-primary lg:text-5xl">
                Ready to Get Started?
              </h2>
              <p className="mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary">
                Take the first step towards early detection and better brain
                health outcomes.
              </p>
            </div>

            {/* Apple-Style CTA Button */}
            <div className="animate-slide-up">
              <Button
                variant="primary"
                size="xl"
                className="min-w-[280px] shadow-medical hover:shadow-medical-hover"
                onClick={() => {
                  if (typeof window !== 'undefined') {
                    window.location.href = '/assessment';
                  }
                }}
              >
                Start Your Health Check
                <svg
                  className="ml-3 h-6 w-6"
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

            {/* Apple-Style Privacy Notice */}
            <div className="animate-scale-in pt-8">
              <div className="inline-flex items-center gap-3 rounded-apple-lg bg-gray-50 px-6 py-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-apple bg-medical-50">
                  <svg
                    className="h-4 w-4 text-medical-500"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="text-left">
                  <div className="text-sm font-medium text-text-primary">
                    Privacy Protected
                  </div>
                  <div className="text-xs text-text-secondary">
                    Data processed locally in your browser
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
