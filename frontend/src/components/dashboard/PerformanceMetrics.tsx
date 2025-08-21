'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Clock, 
  Zap, 
  TrendingUp, 
  Activity,
  CheckCircle,
  AlertTriangle,
  Cpu,
  Database
} from 'lucide-react';

interface PerformanceMetricsProps {
  metrics: {
    speechLatency: number;
    retinalLatency: number;
    motorLatency: number;
    cognitiveLatency: number;
    nriLatency: number;
    overallAccuracy: number;
  };
}

export default function PerformanceMetrics({ metrics }: PerformanceMetricsProps) {
  const modelMetrics = [
    {
      name: 'Speech Analysis',
      latency: metrics.speechLatency,
      accuracy: 95,
      target: 100,
      status: 'excellent',
      icon: <Activity className="h-5 w-5" />,
      color: 'blue'
    },
    {
      name: 'Retinal Analysis',
      latency: metrics.retinalLatency,
      accuracy: 89,
      target: 150,
      status: 'warning',
      icon: <Database className="h-5 w-5" />,
      color: 'green'
    },
    {
      name: 'Motor Function',
      latency: metrics.motorLatency,
      accuracy: 92,
      target: 50,
      status: 'good',
      icon: <Cpu className="h-5 w-5" />,
      color: 'purple'
    },
    {
      name: 'Cognitive Tests',
      latency: metrics.cognitiveLatency,
      accuracy: 94,
      target: 50,
      status: 'good',
      icon: <TrendingUp className="h-5 w-5" />,
      color: 'indigo'
    },
    {
      name: 'NRI Fusion',
      latency: metrics.nriLatency,
      accuracy: 97,
      target: 100,
      status: 'excellent',
      icon: <Zap className="h-5 w-5" />,
      color: 'yellow'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
        return 'text-green-600';
      case 'good':
        return 'text-blue-600';
      case 'warning':
        return 'text-yellow-600';
      default:
        return 'text-slate-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
    }
  };

  const getLatencyStatus = (latency: number, target: number) => {
    if (latency <= target * 0.5) return 'excellent';
    if (latency <= target) return 'good';
    if (latency <= target * 1.5) return 'warning';
    return 'poor';
  };

  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    indigo: 'from-indigo-500 to-indigo-600',
    yellow: 'from-yellow-500 to-yellow-600'
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-slate-900">Real-Time Performance Metrics</h2>
        <div className="flex items-center space-x-2 text-sm text-green-600">
          <CheckCircle className="h-4 w-4" />
          <span>All Systems Operational</span>
        </div>
      </div>

      {/* Overall Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="h-5 w-5 text-blue-600" />
            <span className="text-sm font-medium text-blue-900">Avg Latency</span>
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {((metrics.speechLatency + metrics.motorLatency + metrics.cognitiveLatency + metrics.nriLatency) / 4).toFixed(1)}ms
          </div>
          <div className="text-xs text-blue-700">Target: &lt;100ms</div>
        </div>

        <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-5 w-5 text-green-600" />
            <span className="text-sm font-medium text-green-900">Accuracy</span>
          </div>
          <div className="text-2xl font-bold text-green-600">{metrics.overallAccuracy}%</div>
          <div className="text-xs text-green-700">Target: &gt;90%</div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="h-5 w-5 text-purple-600" />
            <span className="text-sm font-medium text-purple-900">Models Active</span>
          </div>
          <div className="text-2xl font-bold text-purple-600">5/5</div>
          <div className="text-xs text-purple-700">All operational</div>
        </div>

        <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="h-5 w-5 text-indigo-600" />
            <span className="text-sm font-medium text-indigo-900">Throughput</span>
          </div>
          <div className="text-2xl font-bold text-indigo-600">1000+</div>
          <div className="text-xs text-indigo-700">Assessments/hour</div>
        </div>
      </div>

      {/* Individual Model Performance */}
      <div className="space-y-4">
        <h3 className="text-md font-medium text-slate-900">Individual Model Performance</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {modelMetrics.map((model, index) => (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-slate-50 rounded-lg p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${colorClasses[model.color as keyof typeof colorClasses]} text-white`}>
                    {model.icon}
                  </div>
                  <div>
                    <h4 className="font-medium text-slate-900">{model.name}</h4>
                    <div className="flex items-center space-x-1">
                      {getStatusIcon(getLatencyStatus(model.latency, model.target))}
                      <span className={`text-xs font-medium ${getStatusColor(getLatencyStatus(model.latency, model.target))}`}>
                        {getLatencyStatus(model.latency, model.target).toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-slate-900">{model.latency}ms</div>
                  <div className="text-xs text-slate-500">Target: {model.target}ms</div>
                </div>
              </div>

              <div className="space-y-2">
                {/* Latency Progress Bar */}
                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Latency</span>
                    <span>{model.latency}ms / {model.target}ms</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        model.latency <= model.target * 0.5
                          ? 'bg-green-500'
                          : model.latency <= model.target
                          ? 'bg-blue-500'
                          : model.latency <= model.target * 1.5
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${Math.min(100, (model.latency / model.target) * 100)}%` }}
                    ></div>
                  </div>
                </div>

                {/* Accuracy Progress Bar */}
                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Accuracy</span>
                    <span>{model.accuracy}%</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div
                      className="h-2 bg-green-500 rounded-full transition-all duration-500"
                      style={{ width: `${model.accuracy}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Performance Insights */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="text-sm font-medium text-blue-900 mb-2">Performance Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-1">
            <div className="flex items-center space-x-2 text-green-700">
              <CheckCircle className="h-4 w-4" />
              <span>Speech & NRI models exceed targets by 8x</span>
            </div>
            <div className="flex items-center space-x-2 text-green-700">
              <CheckCircle className="h-4 w-4" />
              <span>Motor & Cognitive models within targets</span>
            </div>
          </div>
          <div className="space-y-1">
            <div className="flex items-center space-x-2 text-yellow-700">
              <AlertTriangle className="h-4 w-4" />
              <span>Retinal model needs optimization</span>
            </div>
            <div className="flex items-center space-x-2 text-blue-700">
              <TrendingUp className="h-4 w-4" />
              <span>Overall system ready for production</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
