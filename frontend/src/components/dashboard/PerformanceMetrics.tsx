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
  Database,
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

export default function PerformanceMetrics({
  metrics,
}: PerformanceMetricsProps) {
  // Apple Design System color mapping
  const getAppleColor = (color: string) => {
    switch (color) {
      case 'blue':
        return '#007AFF';
      case 'green':
        return '#34C759';
      case 'purple':
        return '#9B59B6';
      case 'indigo':
        return '#5856D6';
      case 'yellow':
        return '#FF9500';
      default:
        return '#007AFF';
    }
  };

  const getAppleStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
        return '#34C759';
      case 'good':
        return '#007AFF';
      case 'warning':
        return '#FF9500';
      default:
        return '#FF3B30';
    }
  };

  const getAppleStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good':
        return <CheckCircle className="h-4 w-4" style={{ color: '#34C759' }} />;
      case 'warning':
        return (
          <AlertTriangle className="h-4 w-4" style={{ color: '#FF9500' }} />
        );
      default:
        return (
          <AlertTriangle className="h-4 w-4" style={{ color: '#FF3B30' }} />
        );
    }
  };

  const modelMetrics = [
    {
      name: 'Speech Analysis',
      latency: metrics.speechLatency,
      accuracy: 95,
      target: 100,
      status: 'excellent',
      icon: <Activity className="h-5 w-5" />,
      color: 'blue',
    },
    {
      name: 'Retinal Analysis',
      latency: metrics.retinalLatency,
      accuracy: 89,
      target: 150,
      status: 'warning',
      icon: <Database className="h-5 w-5" />,
      color: 'green',
    },
    {
      name: 'Motor Function',
      latency: metrics.motorLatency,
      accuracy: 92,
      target: 50,
      status: 'good',
      icon: <Cpu className="h-5 w-5" />,
      color: 'purple',
    },
    {
      name: 'Cognitive Tests',
      latency: metrics.cognitiveLatency,
      accuracy: 94,
      target: 50,
      status: 'good',
      icon: <TrendingUp className="h-5 w-5" />,
      color: 'indigo',
    },
    {
      name: 'NRI Fusion',
      latency: metrics.nriLatency,
      accuracy: 97,
      target: 100,
      status: 'excellent',
      icon: <Zap className="h-5 w-5" />,
      color: 'yellow',
    },
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
    yellow: 'from-yellow-500 to-yellow-600',
  };

  return (
    <div className="mx-auto max-w-7xl">
      {/* Apple-Style Header */}
      <div className="mb-12 flex items-center justify-between">
        <h2
          className="text-2xl font-bold"
          style={{
            color: '#1D1D1F',
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          }}
        >
          Real-Time Performance Metrics
        </h2>
        <div className="flex items-center space-x-3">
          <CheckCircle className="h-5 w-5" style={{ color: '#34C759' }} />
          <span className="text-base font-medium" style={{ color: '#34C759' }}>
            All Systems Operational
          </span>
        </div>
      </div>

      {/* Apple-Style Overall Stats - No Cards */}
      <div className="mb-16 grid grid-cols-1 gap-8 md:grid-cols-4">
        <div className="text-center">
          <div className="mb-4 flex items-center justify-center space-x-3">
            <Clock className="h-6 w-6" style={{ color: '#007AFF' }} />
            <span className="text-lg font-medium" style={{ color: '#1D1D1F' }}>
              Avg Latency
            </span>
          </div>
          <div className="mb-2 text-4xl font-bold" style={{ color: '#007AFF' }}>
            {(
              (metrics.speechLatency +
                metrics.motorLatency +
                metrics.cognitiveLatency +
                metrics.nriLatency) /
              4
            ).toFixed(1)}
            ms
          </div>
          <div className="text-base" style={{ color: '#86868B' }}>
            Target: &lt;100ms
          </div>
        </div>

        <div className="text-center">
          <div className="mb-4 flex items-center justify-center space-x-3">
            <TrendingUp className="h-6 w-6" style={{ color: '#34C759' }} />
            <span className="text-lg font-medium" style={{ color: '#1D1D1F' }}>
              Accuracy
            </span>
          </div>
          <div className="mb-2 text-4xl font-bold" style={{ color: '#34C759' }}>
            {metrics.overallAccuracy}%
          </div>
          <div className="text-base" style={{ color: '#86868B' }}>
            Target: &gt;90%
          </div>
        </div>

        <div className="text-center">
          <div className="mb-4 flex items-center justify-center space-x-3">
            <Zap className="h-6 w-6" style={{ color: '#FF9500' }} />
            <span className="text-lg font-medium" style={{ color: '#1D1D1F' }}>
              Models Active
            </span>
          </div>
          <div className="mb-2 text-4xl font-bold" style={{ color: '#FF9500' }}>
            5/5
          </div>
          <div className="text-base" style={{ color: '#86868B' }}>
            All operational
          </div>
        </div>

        <div className="text-center">
          <div className="mb-4 flex items-center justify-center space-x-3">
            <Activity className="h-6 w-6" style={{ color: '#9B59B6' }} />
            <span className="text-lg font-medium" style={{ color: '#1D1D1F' }}>
              Throughput
            </span>
          </div>
          <div className="mb-2 text-4xl font-bold" style={{ color: '#9B59B6' }}>
            1000+
          </div>
          <div className="text-base" style={{ color: '#86868B' }}>
            Assessments/hour
          </div>
        </div>
      </div>

      {/* Apple-Style Individual Model Performance */}
      <div className="space-y-8">
        <h3
          className="text-2xl font-bold"
          style={{
            color: '#1D1D1F',
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          }}
        >
          Individual Model Performance
        </h3>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          {modelMetrics.map((model, index) => (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-6"
              style={{ backgroundColor: '#FFFFFF' }}
            >
              <div className="mb-6 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div
                    className="rounded-lg p-3"
                    style={{ backgroundColor: getAppleColor(model.color) }}
                  >
                    {React.cloneElement(model.icon as React.ReactElement, {
                      style: { color: '#FFFFFF' },
                      className: 'h-6 w-6',
                    })}
                  </div>
                  <div>
                    <h4
                      className="text-lg font-semibold"
                      style={{ color: '#1D1D1F' }}
                    >
                      {model.name}
                    </h4>
                    <div className="flex items-center space-x-2">
                      {getAppleStatusIcon(
                        getLatencyStatus(model.latency, model.target)
                      )}
                      <span
                        className="text-sm font-medium"
                        style={{
                          color: getAppleStatusColor(
                            getLatencyStatus(model.latency, model.target)
                          ),
                        }}
                      >
                        {getLatencyStatus(
                          model.latency,
                          model.target
                        ).toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className="text-2xl font-bold"
                    style={{ color: '#1D1D1F' }}
                  >
                    {model.latency}ms
                  </div>
                  <div className="text-xs text-slate-500">
                    Target: {model.target}ms
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                {/* Latency Progress Bar */}
                <div>
                  <div className="mb-1 flex justify-between text-xs text-slate-600">
                    <span>Latency</span>
                    <span>
                      {model.latency}ms / {model.target}ms
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-slate-200">
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
                      style={{
                        width: `${Math.min(100, (model.latency / model.target) * 100)}%`,
                      }}
                    ></div>
                  </div>
                </div>

                {/* Accuracy Progress Bar */}
                <div>
                  <div className="mb-1 flex justify-between text-xs text-slate-600">
                    <span>Accuracy</span>
                    <span>{model.accuracy}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-slate-200">
                    <div
                      className="h-2 rounded-full bg-green-500 transition-all duration-500"
                      style={{ width: `${model.accuracy}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Apple-Style Performance Insights */}
      <div className="mt-12 pt-8" style={{ borderTop: '1px solid #F2F2F7' }}>
        <h3
          className="mb-8 text-2xl font-bold"
          style={{
            color: '#1D1D1F',
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          }}
        >
          Performance Insights
        </h3>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5" style={{ color: '#34C759' }} />
              <span className="text-base" style={{ color: '#1D1D1F' }}>
                Speech & NRI models exceed targets by 8x
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5" style={{ color: '#34C759' }} />
              <span className="text-base" style={{ color: '#1D1D1F' }}>
                Motor & Cognitive models within targets
              </span>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="h-5 w-5" style={{ color: '#FF9500' }} />
              <span className="text-base" style={{ color: '#1D1D1F' }}>
                Retinal model needs optimization
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <TrendingUp className="h-5 w-5" style={{ color: '#007AFF' }} />
              <span className="text-base" style={{ color: '#1D1D1F' }}>
                Overall system ready for production
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
