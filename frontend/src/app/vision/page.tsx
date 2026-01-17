'use client';

import { motion } from 'framer-motion';
import {
  Target,
  Lightbulb,
  Users,
  Globe,
  Heart,
  Shield,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  Eye,
  Brain,
  Zap,
} from 'lucide-react';
import Link from 'next/link';
import { Layout } from '@/components/layout';

// Animation variants following UI/UX guidelines
const fadeInUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.08,
    },
  },
};

// Hero Section - Vision Statement
function HeroSection() {
  return (
    <section className="relative -mt-14 pt-14 overflow-hidden bg-black selection:bg-white/20">
      {/* Spotlight Effect */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] sm:w-[800px] h-[300px] sm:h-[400px] bg-violet-500/[0.04] blur-[100px] sm:blur-[120px] rounded-full pointer-events-none" />

      {/* Grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:48px_48px] sm:bg-[size:64px_64px]" />

      <div className="relative z-10 mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-16 sm:py-20 lg:py-28">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center gap-2 rounded-full bg-violet-900/20 border border-violet-800/50 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-violet-300 mb-6"
          >
            <Target className="h-3 w-3" />
            <span>Our Vision</span>
          </motion.div>

          {/* Headline */}
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-semibold leading-[1.1] tracking-tight text-white mb-6 px-2">
            Democratizing
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-pink-400">
              AI-Powered Healthcare
            </span>
          </h1>

          {/* Description */}
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] lg:text-[16px] leading-relaxed text-zinc-400 mb-8 px-4">
            We envision a future where advanced AI diagnostics are accessible to every healthcare provider,
            regardless of resources or location. MediLens bridges the gap between cutting-edge technology
            and everyday clinical practice.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 px-4">
            <Link
              href="/dashboard"
              className="group inline-flex items-center justify-center gap-2 rounded-full bg-white px-6 sm:px-8 py-3 text-[13px] sm:text-[14px] font-medium text-black transition-all hover:bg-zinc-200 w-full sm:w-auto"
            >
              Get Started
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/about"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-zinc-800 bg-white/5 px-6 sm:px-8 py-3 text-[13px] sm:text-[14px] font-medium text-white transition-all hover:bg-white/10 w-full sm:w-auto"
            >
              Learn More
            </Link>
          </div>
        </motion.div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-black to-transparent" />
    </section>
  );
}

// Mission Section
function MissionSection() {
  const missions = [
    {
      icon: Heart,
      title: 'Patient-First Approach',
      description: 'Every feature we build is designed with patients in mind, ensuring accurate, timely, and compassionate care delivery.',
      color: 'text-red-400',
      bg: 'bg-red-900/20',
      border: 'group-hover:border-red-500/30',
    },
    {
      icon: Shield,
      title: 'Clinical Accuracy',
      description: 'Our AI models achieve 95%+ accuracy through rigorous validation against gold-standard medical datasets.',
      color: 'text-blue-400',
      bg: 'bg-blue-900/20',
      border: 'group-hover:border-blue-500/30',
    },
    {
      icon: Globe,
      title: 'Global Accessibility',
      description: 'Breaking barriers to advanced diagnostics, making AI healthcare tools available across borders and resource constraints.',
      color: 'text-emerald-400',
      bg: 'bg-emerald-900/20',
      border: 'group-hover:border-emerald-500/30',
    },
  ];

  return (
    <section className="bg-zinc-950 py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-red-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">Our Mission</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-4 tracking-tight">
            Transforming Healthcare Delivery
          </h2>
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            We are committed to empowering healthcare providers with AI tools that enhance
            diagnostic accuracy, reduce turnaround time, and improve patient outcomes.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8"
        >
          {missions.map((mission, index) => (
            <motion.div
              key={mission.title}
              variants={fadeInUp}
              className={`group text-center p-6 sm:p-8 rounded-2xl bg-zinc-900/30 border border-zinc-800 hover:bg-zinc-900/50 transition-all duration-300 ${mission.border}`}
            >
              <div className={`mx-auto mb-5 sm:mb-6 flex h-14 w-14 items-center justify-center rounded-2xl ${mission.bg} border border-white/5`}>
                <mission.icon className={`h-6 w-6 ${mission.color}`} strokeWidth={1.5} />
              </div>
              <h3 className="text-[15px] sm:text-[16px] font-medium text-white mb-3">{mission.title}</h3>
              <p className="text-[13px] sm:text-[14px] text-zinc-400 leading-relaxed">{mission.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

// Core Values Section
function ValuesSection() {
  const values = [
    {
      icon: Lightbulb,
      title: 'Innovation',
      description: 'Continuously pushing the boundaries of medical AI technology.',
      color: 'text-amber-400',
      bg: 'bg-amber-900/20',
    },
    {
      icon: Users,
      title: 'Collaboration',
      description: 'Working alongside clinicians and researchers to build better solutions.',
      color: 'text-violet-400',
      bg: 'bg-violet-900/20',
    },
    {
      icon: Shield,
      title: 'Trust',
      description: 'Maintaining the highest standards of data privacy and security.',
      color: 'text-cyan-400',
      bg: 'bg-cyan-900/20',
    },
    {
      icon: Sparkles,
      title: 'Excellence',
      description: 'Pursuing perfection in every diagnostic output we deliver.',
      color: 'text-pink-400',
      bg: 'bg-pink-900/20',
    },
  ];

  return (
    <section className="bg-black py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-amber-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">Core Values</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-4 tracking-tight">
            What Drives Us Forward
          </h2>
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            Our core values guide every decision, from product development to patient interaction.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          {values.map((value, index) => (
            <motion.div
              key={value.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="text-center p-5 sm:p-6 rounded-xl bg-zinc-900/20 border border-zinc-800 hover:border-zinc-700 transition-all duration-300"
            >
              <div className={`mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl ${value.bg} border border-white/5`}>
                <value.icon className={`h-5 w-5 ${value.color}`} strokeWidth={1.5} />
              </div>
              <h3 className="text-[14px] sm:text-[15px] font-medium text-white mb-2">{value.title}</h3>
              <p className="text-[12px] sm:text-[13px] text-zinc-500 leading-relaxed">{value.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Impact Goals Section
function ImpactSection() {
  const goals = [
    { value: '1M+', label: 'Patients Reached', description: 'Target by 2027' },
    { value: '50+', label: 'Countries', description: 'Global deployment' },
    { value: '100+', label: 'Hospitals', description: 'Partnership network' },
    { value: '99%', label: 'Satisfaction', description: 'Provider rating' },
  ];

  return (
    <section className="bg-zinc-950 py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">Impact Goals</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-4 tracking-tight">
            Measurable Healthcare Impact
          </h2>
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            We set ambitious goals to ensure MediLens makes a meaningful difference in global healthcare.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 sm:gap-8">
          {goals.map((goal, index) => (
            <motion.div
              key={goal.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="text-center p-5 sm:p-6 rounded-xl bg-zinc-900/30 border border-zinc-800"
            >
              <div className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-white mb-2 tracking-tight">{goal.value}</div>
              <div className="text-[13px] sm:text-[14px] font-medium text-zinc-300 mb-1">{goal.label}</div>
              <div className="text-[11px] sm:text-[12px] text-zinc-500">{goal.description}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Future Roadmap Section
function RoadmapSection() {
  const milestones = [
    {
      phase: '2024',
      title: 'Foundation',
      items: ['Launch core platform with 4 modules', 'HIPAA compliance certification', 'Initial clinical partnerships'],
      status: 'completed',
      icon: CheckCircle2,
      color: 'text-emerald-400',
      bg: 'bg-emerald-900/20',
    },
    {
      phase: '2025',
      title: 'Expansion',
      items: ['Expand to 11 diagnostic modules', 'Multi-language support', 'Mobile application launch'],
      status: 'in-progress',
      icon: TrendingUp,
      color: 'text-blue-400',
      bg: 'bg-blue-900/20',
    },
    {
      phase: '2026',
      title: 'Scale',
      items: ['Enterprise EMR integrations', 'Real-time collaboration features', 'Advanced analytics dashboard'],
      status: 'upcoming',
      icon: Zap,
      color: 'text-violet-400',
      bg: 'bg-violet-900/20',
    },
    {
      phase: '2027',
      title: 'Global',
      items: ['Global deployment across 50+ countries', 'AI-powered treatment recommendations', 'Research partnership program'],
      status: 'upcoming',
      icon: Globe,
      color: 'text-pink-400',
      bg: 'bg-pink-900/20',
    },
  ];

  return (
    <section className="bg-black py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-violet-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">Roadmap</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-4 tracking-tight">
            Building the Future of Healthcare
          </h2>
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            Our strategic roadmap outlines the path to becoming the global standard in AI-powered diagnostics.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          {milestones.map((milestone, index) => (
            <motion.div
              key={milestone.phase}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`p-5 sm:p-6 rounded-xl border transition-all duration-300 ${
                milestone.status === 'completed'
                  ? 'bg-zinc-900/50 border-emerald-800/30'
                  : milestone.status === 'in-progress'
                  ? 'bg-zinc-900/50 border-blue-800/30'
                  : 'bg-zinc-900/20 border-zinc-800'
              }`}
            >
              <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${milestone.bg} mb-4`}>
                <milestone.icon className={`h-5 w-5 ${milestone.color}`} strokeWidth={1.5} />
              </div>
              <div className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider mb-1">{milestone.phase}</div>
              <h3 className="text-[15px] sm:text-[16px] font-medium text-white mb-3">{milestone.title}</h3>
              <ul className="space-y-2">
                {milestone.items.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-[12px] sm:text-[13px] text-zinc-400">
                    <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                      milestone.status === 'completed' ? 'bg-emerald-500' : 'bg-zinc-600'
                    }`} />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// CTA Section
function CTASection() {
  return (
    <section className="bg-zinc-950 py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="inline-flex items-center justify-center gap-3 mb-6">
            <Eye className="h-8 w-8 text-violet-400" />
            <Brain className="h-8 w-8 text-pink-400" />
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-4 tracking-tight">
            Join Us in Shaping Healthcare&apos;s Future
          </h2>
          <p className="text-[14px] sm:text-[15px] lg:text-[16px] text-zinc-400 mb-8 max-w-2xl mx-auto leading-relaxed">
            Whether you&apos;re a healthcare provider, researcher, or organization, we invite you to explore
            what MediLens can do for your practice.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4">
            <Link
              href="/dashboard"
              className="group inline-flex h-11 sm:h-12 items-center justify-center gap-2 rounded-full bg-white px-6 sm:px-8 text-[13px] sm:text-[14px] font-medium text-black transition-colors hover:bg-zinc-200 w-full sm:w-auto"
            >
              Start Using MediLens
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/about"
              className="inline-flex h-11 sm:h-12 items-center justify-center rounded-full border border-zinc-700 bg-transparent px-6 sm:px-8 text-[13px] sm:text-[14px] font-medium text-white transition-colors hover:bg-zinc-900 hover:border-zinc-500 w-full sm:w-auto"
            >
              View Technical Details
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Main Vision Page Component
export default function VisionPage() {
  return (
    <Layout containerized={false}>
      <HeroSection />
      <MissionSection />
      <ValuesSection />
      <ImpactSection />
      <RoadmapSection />
      <CTASection />
    </Layout>
  );
}
