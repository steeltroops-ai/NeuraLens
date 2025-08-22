'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Brain,
  ArrowRight,
  Activity,
  Shield,
  Clock,
  Mic,
  Eye,
  Hand,
  Zap,
} from 'lucide-react';
import { Layout } from '@/components/layout';
import { Button } from '@/components/ui';

export default function HomePage() {
  const router = useRouter();

  const handleStartAssessment = () => {
    router.push('/assessment');
  };

  const handleViewDashboard = () => {
    router.push('/dashboard');
  };

  return (
    <Layout showHeader={true} showFooter={true} containerized={false}>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
        {/* Hero Section */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -right-32 -top-40 h-80 w-80 rounded-full bg-gradient-to-br from-blue-400/20 to-purple-400/20 blur-3xl"></div>
            <div className="absolute -bottom-40 -left-32 h-80 w-80 rounded-full bg-gradient-to-br from-purple-400/20 to-pink-400/20 blur-3xl"></div>
          </div>

          <div className="relative mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8 lg:py-32">
            <div className="text-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="mb-16 space-y-8"
              >
                <h1 className="text-5xl font-bold leading-tight text-slate-900 sm:text-6xl lg:text-7xl">
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    NeuroLens-X
                  </span>
                </h1>
                <p className="text-2xl font-medium text-slate-700 sm:text-3xl">
                  Early Detection, Better Outcomes.
                </p>
                <p className="mx-auto max-w-4xl text-lg leading-relaxed text-slate-600 sm:text-xl">
                  Multi-modal AI platform detecting neurological disorders
                  through advanced analysis.
                  <br />
                  <span className="font-semibold text-blue-600">
                    50M affected globally
                  </span>{' '}
                  •
                  <span className="font-semibold text-green-600">
                    {' '}
                    40% cost savings
                  </span>{' '}
                  with early detection.
                </p>
              </motion.div>

              {/* Key Features */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="mb-16 flex flex-wrap justify-center gap-6"
              >
                <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                  <Clock className="h-5 w-5 text-blue-600" />
                  <span className="font-medium text-slate-700">
                    Sub-100ms Processing
                  </span>
                </div>
                <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                  <Shield className="h-5 w-5 text-green-600" />
                  <span className="font-medium text-slate-700">
                    90%+ Clinical Accuracy
                  </span>
                </div>
                <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                  <Activity className="h-5 w-5 text-purple-600" />
                  <span className="font-medium text-slate-700">
                    4-Modal AI Analysis
                  </span>
                </div>
                <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                  <Zap className="h-5 w-5 text-orange-600" />
                  <span className="font-medium text-slate-700">
                    HIPAA Compliant
                  </span>
                </div>
              </motion.div>

              {/* CTA Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                className="mb-20 flex flex-col items-center justify-center gap-6 sm:flex-row"
              >
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleStartAssessment}
                  className="w-full bg-blue-600 font-semibold text-white transition-all duration-200 hover:scale-105 hover:bg-blue-700 sm:w-auto"
                  rightIcon={<ArrowRight className="h-5 w-5" />}
                >
                  Start Test
                </Button>

                <Button
                  variant="secondary"
                  size="lg"
                  onClick={handleViewDashboard}
                  className="w-full bg-teal-600 font-semibold text-white transition-all duration-200 hover:scale-105 hover:bg-teal-700 sm:w-auto"
                >
                  Dashboard
                </Button>
              </motion.div>

              {/* Assessment Types - Clean Grid */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className="grid grid-cols-2 gap-6 lg:grid-cols-4"
              >
                <AssessmentCard
                  icon={<Mic className="h-8 w-8 text-white" />}
                  title="Speech Analysis"
                  description="AI-powered voice biomarker detection for Parkinson's and dementia"
                  processingTime="11.7ms"
                  color="from-blue-500 to-blue-600"
                />
                <AssessmentCard
                  icon={<Eye className="h-8 w-8 text-white" />}
                  title="Retinal Imaging"
                  description="Fundus analysis for vascular changes and neurological indicators"
                  processingTime="145ms"
                  color="from-green-500 to-green-600"
                />
                <AssessmentCard
                  icon={<Hand className="h-8 w-8 text-white" />}
                  title="Motor Assessment"
                  description="Tremor detection and movement pattern analysis"
                  processingTime="42ms"
                  color="from-purple-500 to-purple-600"
                />
                <AssessmentCard
                  icon={<Brain className="h-8 w-8 text-white" />}
                  title="Cognitive Testing"
                  description="Memory, attention, and executive function evaluation"
                  processingTime="38ms"
                  color="from-indigo-500 to-indigo-600"
                />
              </motion.div>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}

// Assessment Card Component
function AssessmentCard({
  icon,
  title,
  description,
  processingTime,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  processingTime: string;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white/60 p-6 backdrop-blur-sm transition-all duration-300 hover:border-slate-300 hover:shadow-lg">
      <div
        className={`mx-auto mb-4 w-fit rounded-lg bg-gradient-to-r p-4 ${color}`}
      >
        {icon}
      </div>
      <h3 className="mb-2 text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mb-4 text-slate-600">{description}</p>
      <div className="text-sm font-medium text-green-600">
        {processingTime} processing
      </div>
    </div>
  );
}
