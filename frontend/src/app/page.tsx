'use client';

import React from 'react';
import { motion } from 'framer-motion';
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
  Menu,
  X
} from 'lucide-react';

export default function HomePage() {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Navigation Header */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold text-slate-900">NeuroLens</span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <a href="/validation" className="text-slate-600 hover:text-slate-900 transition-colors">Validation</a>
              <a href="/dashboard" className="text-slate-600 hover:text-slate-900 transition-colors">Dashboard</a>
              <button 
                onClick={() => window.location.href = '/assessment'}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
              >
                Start Assessment
              </button>
            </div>

            {/* Mobile Menu Button */}
            <div className="md:hidden">
              <button 
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-slate-600 hover:text-slate-900"
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden">
              <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white border-t border-slate-200">
                <a href="/validation" className="block px-3 py-2 text-slate-600 hover:text-slate-900">Validation</a>
                <a href="/dashboard" className="block px-3 py-2 text-slate-600 hover:text-slate-900">Dashboard</a>
                <button 
                  onClick={() => window.location.href = '/assessment'}
                  className="block w-full text-left px-3 py-2 bg-blue-600 text-white rounded-lg font-medium"
                >
                  Start Assessment
                </button>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-32 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-40 -left-32 w-80 h-80 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full blur-3xl"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="space-y-8 mb-16"
            >
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-slate-900 leading-tight">
                Advanced{' '}
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Neurological
                </span>
                <br />
                Assessment Platform
              </h1>
              <p className="text-xl sm:text-2xl text-slate-600 max-w-4xl mx-auto leading-relaxed">
                Real-time AI-powered screening for neurological conditions with 
                <span className="font-semibold text-blue-600"> 95%+ accuracy</span> and 
                <span className="font-semibold text-purple-600"> sub-100ms processing</span>
              </p>
            </motion.div>

            {/* Key Features */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex flex-wrap justify-center gap-6 mb-16"
            >
              <div className="flex items-center space-x-2 bg-white/60 backdrop-blur-sm px-6 py-3 rounded-full border border-slate-200">
                <Clock className="h-5 w-5 text-blue-600" />
                <span className="text-slate-700 font-medium">Real-time Processing</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/60 backdrop-blur-sm px-6 py-3 rounded-full border border-slate-200">
                <Shield className="h-5 w-5 text-green-600" />
                <span className="text-slate-700 font-medium">95%+ Accuracy</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/60 backdrop-blur-sm px-6 py-3 rounded-full border border-slate-200">
                <Activity className="h-5 w-5 text-purple-600" />
                <span className="text-slate-700 font-medium">Multi-Modal Analysis</span>
              </div>
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-20"
            >
              <button
                onClick={() => window.location.href = '/assessment'}
                className="group w-full sm:w-auto bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-10 py-4 rounded-xl font-semibold text-lg transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center justify-center space-x-3"
              >
                <span>Start Assessment</span>
                <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </button>
              
              <button
                onClick={() => window.location.href = '/dashboard'}
                className="w-full sm:w-auto bg-white hover:bg-slate-50 text-slate-900 px-10 py-4 rounded-xl font-semibold text-lg transition-all duration-300 border border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md"
              >
                View Dashboard
              </button>
            </motion.div>

            {/* Assessment Types - Clean Grid */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="grid grid-cols-2 lg:grid-cols-4 gap-6"
            >
              <AssessmentCard
                icon={<Mic className="h-8 w-8 text-white" />}
                title="Speech Analysis"
                description="Voice pattern analysis"
                processingTime="11.7ms"
                color="from-blue-500 to-blue-600"
              />
              <AssessmentCard
                icon={<Eye className="h-8 w-8 text-white" />}
                title="Retinal Imaging"
                description="Fundus analysis"
                processingTime="145ms"
                color="from-green-500 to-green-600"
              />
              <AssessmentCard
                icon={<Hand className="h-8 w-8 text-white" />}
                title="Motor Function"
                description="Movement patterns"
                processingTime="42ms"
                color="from-purple-500 to-purple-600"
              />
              <AssessmentCard
                icon={<Brain className="h-8 w-8 text-white" />}
                title="Cognitive Tests"
                description="Memory & attention"
                processingTime="38ms"
                color="from-indigo-500 to-indigo-600"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <span className="text-2xl font-bold">NeuroLens</span>
          </div>
          <p className="text-slate-400 mb-6 text-lg">
            Advanced neurological assessment platform powered by real-time AI analysis.
          </p>
          <div className="flex items-center justify-center space-x-6 text-slate-400">
            <span>© 2024 NeuroLens</span>
            <span>•</span>
            <span>All rights reserved</span>
          </div>
        </div>
      </footer>
    </div>
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
    <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-slate-200 hover:border-slate-300 transition-all duration-300 hover:shadow-lg">
      <div className={`p-4 bg-gradient-to-r ${color} rounded-lg w-fit mx-auto mb-4`}>
        {icon}
      </div>
      <h3 className="font-semibold text-slate-900 mb-2 text-lg">{title}</h3>
      <p className="text-slate-600 mb-4">{description}</p>
      <div className="text-sm text-green-600 font-medium">{processingTime} processing</div>
    </div>
  );
}
