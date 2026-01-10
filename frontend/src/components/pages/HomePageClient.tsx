'use client';

import { motion } from 'framer-motion';
import {
  Brain,
  Eye,
  Mic,
  Hand,
  Activity,
  Shield,
  Clock,
  Zap,
  ArrowRight,
  CheckCircle,
  Star,
  Users,
  Award,
  TrendingUp,
  Mail,
  Phone,
  MapPin,
} from 'lucide-react';
import { useState, useEffect } from 'react';
import Link from 'next/link';

// Animation variants for smooth Apple-like transitions
const fadeInUp = {
  initial: { opacity: 0, y: 60 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.8, ease: [0.6, -0.05, 0.01, 0.99] },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const scaleOnHover = {
  whileHover: { scale: 1.05 },
  whileTap: { scale: 0.95 },
  transition: { type: 'spring' as const, stiffness: 400, damping: 17 },
};

// Hero Section Component
function HeroSection() {
  const [currentFeature, setCurrentFeature] = useState(0);
  const features = ['Speech Analysis', 'Retinal Imaging', 'Motor Assessment', 'Cognitive Testing'];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature(prev => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [features.length]);

  return (
    <section className='relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50'>
      {/* Background geometric elements */}
      <div className='absolute inset-0 overflow-hidden'>
        <div className='absolute left-1/4 top-1/4 h-64 w-64 rounded-full bg-blue-200/20 blur-3xl'></div>
        <div className='absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-indigo-200/20 blur-3xl'></div>
      </div>

      <div className='relative z-10 mx-auto max-w-7xl px-4 text-center sm:px-6 lg:px-8'>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className='space-y-8'
        >
          {/* Main headline */}
          <h1 className='text-5xl font-bold leading-tight text-slate-900 md:text-7xl'>
            The Future of
            <br />
            <span className='bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent'>
              Neurological Care
            </span>
          </h1>

          {/* Dynamic feature showcase */}
          <div className='flex h-16 items-center justify-center'>
            <motion.p
              key={currentFeature}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className='text-xl font-medium text-slate-600 md:text-2xl'
            >
              Advanced {features[currentFeature]} • AI-Powered Detection
            </motion.p>
          </div>

          {/* Subtitle */}
          <p className='mx-auto max-w-3xl text-lg leading-relaxed text-slate-600 md:text-xl'>
            Revolutionary multi-modal AI platform for early detection of neurological conditions.
            Transforming healthcare through precision, speed, and accessibility.
          </p>

          {/* CTA Buttons */}
          <div className='flex flex-col items-center justify-center gap-4 pt-8 sm:flex-row'>
            <motion.div {...scaleOnHover}>
              <Link
                href='/assessment'
                className='inline-flex items-center rounded-2xl bg-blue-600 px-8 py-4 font-semibold text-white shadow-lg transition-colors hover:bg-blue-700 hover:shadow-xl'
              >
                Start Assessment
                <ArrowRight className='ml-2 h-5 w-5' />
              </Link>
            </motion.div>

            <motion.div {...scaleOnHover}>
              <Link
                href='/dashboard'
                className='inline-flex items-center rounded-2xl border border-slate-200 bg-white px-8 py-4 font-semibold text-slate-700 shadow-lg transition-colors hover:bg-slate-50 hover:shadow-xl'
              >
                View Dashboard
                <Activity className='ml-2 h-5 w-5' />
              </Link>
            </motion.div>
          </div>

          {/* Trust indicators */}
          <div className='flex flex-wrap items-center justify-center gap-8 pt-12 text-slate-500'>
            <div className='flex items-center space-x-2'>
              <Shield className='h-5 w-5' />
              <span className='text-sm font-medium'>HIPAA Compliant</span>
            </div>
            <div className='flex items-center space-x-2'>
              <Award className='h-5 w-5' />
              <span className='text-sm font-medium'>FDA Cleared</span>
            </div>
            <div className='flex items-center space-x-2'>
              <Users className='h-5 w-5' />
              <span className='text-sm font-medium'>10,000+ Patients</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Assessment Features Section
function AssessmentFeatures() {
  const assessments = [
    {
      icon: Mic,
      title: 'Speech Analysis',
      description:
        'Advanced voice pattern recognition to detect early signs of neurological conditions through speech fluency, tremor, and articulation analysis.',
      features: ['Real-time processing', 'Voice biomarkers', 'Tremor detection', 'Fluency scoring'],
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Eye,
      title: 'Retinal Imaging',
      description:
        'Computer vision analysis of fundus images to identify vascular changes and neurological indicators in retinal patterns.',
      features: [
        'Vessel analysis',
        'A/V ratio calculation',
        'Cup-disc assessment',
        'Density mapping',
      ],
      color: 'from-emerald-500 to-teal-500',
    },
    {
      icon: Hand,
      title: 'Motor Assessment',
      description:
        'Interactive finger tapping and coordination tests to evaluate motor function, rhythm consistency, and movement patterns.',
      features: [
        'Rhythm analysis',
        'Coordination testing',
        'Bradykinesia detection',
        'Movement tracking',
      ],
      color: 'from-purple-500 to-indigo-500',
    },
    {
      icon: Brain,
      title: 'Cognitive Testing',
      description:
        'Comprehensive cognitive evaluation through memory, attention, and executive function assessments.',
      features: ['Memory testing', 'Attention span', 'Executive function', 'Processing speed'],
      color: 'from-orange-500 to-red-500',
    },
    {
      icon: Activity,
      title: 'Risk Assessment',
      description:
        'Multi-modal fusion algorithm combining all assessment data to provide comprehensive neurological risk scoring.',
      features: ['Data fusion', 'Risk scoring', 'Trend analysis', 'Predictive modeling'],
      color: 'from-pink-500 to-rose-500',
    },
    {
      icon: TrendingUp,
      title: 'Progress Tracking',
      description:
        'Longitudinal monitoring and analysis of assessment results to track changes and treatment effectiveness over time.',
      features: [
        'Timeline tracking',
        'Progress reports',
        'Trend visualization',
        'Clinical insights',
      ],
      color: 'from-violet-500 to-purple-500',
    },
  ];

  return (
    <section className='bg-white py-24'>
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className='mb-16 text-center'
        >
          <h2 className='mb-6 text-4xl font-bold text-slate-900 md:text-5xl'>
            Comprehensive Assessment Suite
          </h2>
          <p className='mx-auto max-w-3xl text-xl text-slate-600'>
            Six specialized modules working together to provide the most complete neurological
            assessment available.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial='initial'
          whileInView='animate'
          viewport={{ once: true }}
          className='grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3'
        >
          {assessments.map((assessment, index) => (
            <motion.div
              key={assessment.title}
              variants={fadeInUp}
              whileHover={{ y: -8 }}
              className='rounded-3xl border border-slate-100 bg-white p-8 shadow-lg transition-all duration-300 hover:shadow-2xl'
            >
              <div
                className={`h-16 w-16 rounded-2xl bg-gradient-to-r ${assessment.color} mb-6 flex items-center justify-center`}
              >
                <assessment.icon className='h-8 w-8 text-white' />
              </div>

              <h3 className='mb-4 text-2xl font-bold text-slate-900'>{assessment.title}</h3>
              <p className='mb-6 leading-relaxed text-slate-600'>{assessment.description}</p>

              <ul className='space-y-2'>
                {assessment.features.map((feature, idx) => (
                  <li key={idx} className='flex items-center text-sm text-slate-600'>
                    <CheckCircle className='mr-2 h-4 w-4 flex-shrink-0 text-green-500' />
                    {feature}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

// Performance Stats Section
function PerformanceStats() {
  const stats = [
    { icon: Clock, value: '< 200ms', label: 'Response Time' },
    { icon: Shield, value: '99.9%', label: 'Accuracy Rate' },
    { icon: Users, value: '10,000+', label: 'Patients Served' },
    { icon: Zap, value: '24/7', label: 'Availability' },
  ];

  return (
    <section className='bg-slate-50 py-24'>
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className='mb-16 text-center'
        >
          <h2 className='mb-6 text-4xl font-bold text-slate-900 md:text-5xl'>
            Trusted by Healthcare Professionals
          </h2>
          <p className='mx-auto max-w-3xl text-xl text-slate-600'>
            Our platform delivers consistent, reliable results that healthcare providers depend on
            for critical decisions.
          </p>
        </motion.div>

        <div className='grid grid-cols-2 gap-8 md:grid-cols-4'>
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className='text-center'
            >
              <div className='mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-100'>
                <stat.icon className='h-8 w-8 text-blue-600' />
              </div>
              <div className='mb-2 text-3xl font-bold text-slate-900 md:text-4xl'>{stat.value}</div>
              <div className='font-medium text-slate-600'>{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Professional Footer
function Footer() {
  return (
    <footer className='bg-slate-900 py-16 text-white'>
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <div className='grid grid-cols-1 gap-8 md:grid-cols-4'>
          {/* Company Info */}
          <div className='col-span-1 md:col-span-2'>
            <div className='mb-6 flex items-center space-x-3'>
              <div className='flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600'>
                <Brain className='h-6 w-6 text-white' />
              </div>
              <span className='text-2xl font-bold'>NeuraLens</span>
            </div>
            <p className='mb-6 max-w-md text-slate-300'>
              Advancing neurological healthcare through AI-powered assessment technology. Early
              detection for better outcomes.
            </p>
            <div className='flex space-x-4'>
              <div className='flex items-center space-x-2 text-slate-300'>
                <Mail className='h-4 w-4' />
                <span className='text-sm'>contact@neuralens.ai</span>
              </div>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className='mb-4 text-lg font-semibold'>Platform</h3>
            <ul className='space-y-2'>
              <li>
                <Link
                  href='/assessment'
                  className='text-slate-300 transition-colors hover:text-white'
                >
                  Start Assessment
                </Link>
              </li>
              <li>
                <Link
                  href='/dashboard'
                  className='text-slate-300 transition-colors hover:text-white'
                >
                  Dashboard
                </Link>
              </li>
              <li>
                <Link href='/docs' className='text-slate-300 transition-colors hover:text-white'>
                  Documentation
                </Link>
              </li>
              <li>
                <Link href='/api' className='text-slate-300 transition-colors hover:text-white'>
                  API Reference
                </Link>
              </li>
            </ul>
          </div>

          {/* Support */}
          <div>
            <h3 className='mb-4 text-lg font-semibold'>Support</h3>
            <ul className='space-y-2'>
              <li>
                <Link href='/help' className='text-slate-300 transition-colors hover:text-white'>
                  Help Center
                </Link>
              </li>
              <li>
                <Link href='/contact' className='text-slate-300 transition-colors hover:text-white'>
                  Contact Us
                </Link>
              </li>
              <li>
                <Link href='/privacy' className='text-slate-300 transition-colors hover:text-white'>
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href='/terms' className='text-slate-300 transition-colors hover:text-white'>
                  Terms of Service
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className='mt-12 flex flex-col items-center justify-between border-t border-slate-800 pt-8 md:flex-row'>
          <p className='text-sm text-slate-400'>© 2024 NeuraLens. All rights reserved.</p>
          <div className='mt-4 flex items-center space-x-6 md:mt-0'>
            <span className='text-sm text-slate-400'>Built with precision and care</span>
            <div className='flex items-center space-x-2'>
              <Shield className='h-4 w-4 text-slate-400' />
              <span className='text-sm text-slate-400'>HIPAA Compliant</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

// Main HomePageClient Component
export function HomePageClient() {
  return (
    <div className='min-h-screen'>
      <HeroSection />
      <AssessmentFeatures />
      <PerformanceStats />
      <Footer />
    </div>
  );
}
