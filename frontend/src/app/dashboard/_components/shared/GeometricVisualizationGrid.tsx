'use client';

import React, { memo } from 'react';
import { motion } from 'framer-motion';
import { Mic, Eye, Hand, Brain, Zap, Activity, Maximize2 } from 'lucide-react';

// Import existing geometric visualization components
import {
  SpeechWaveform,
  RetinalEye,
  BrainNeural,
  HandKinematics,
  NRIFusion,
  MultiModalNetwork,
} from '@/components/visuals/AnimatedGeometry';

interface VisualizationCard {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  component: React.ComponentType;
  color: string;
  status: 'active' | 'inactive' | 'processing';
}

interface GeometricVisualizationGridProps {
  visualizations?: VisualizationCard[];
  onVisualizationClick?: (id: string) => void;
}

// Memoized visualization components for performance
const MemoizedSpeechWaveform = memo(SpeechWaveform);
const MemoizedRetinalEye = memo(RetinalEye);
const MemoizedBrainNeural = memo(BrainNeural);
const MemoizedHandKinematics = memo(HandKinematics);
const MemoizedNRIFusion = memo(NRIFusion);
const MemoizedMultiModalNetwork = memo(MultiModalNetwork);

export default function GeometricVisualizationGrid({
  visualizations,
  onVisualizationClick,
}: GeometricVisualizationGridProps) {
  const defaultVisualizations: VisualizationCard[] = [
    {
      id: 'speech',
      title: 'Speech Analysis',
      description: 'Real-time voice pattern visualization',
      icon: <Mic className='h-5 w-5' />,
      component: MemoizedSpeechWaveform,
      color: '#007AFF',
      status: 'active',
    },
    {
      id: 'retinal',
      title: 'Retinal Analysis',
      description: 'Eye structure and vessel patterns',
      icon: <Eye className='h-5 w-5' />,
      component: MemoizedRetinalEye,
      color: '#34C759',
      status: 'active',
    },
    {
      id: 'motor',
      title: 'Hand Kinematics',
      description: 'Motor function and coordination',
      icon: <Hand className='h-5 w-5' />,
      component: MemoizedHandKinematics,
      color: '#FF9500',
      status: 'processing',
    },
    {
      id: 'cognitive',
      title: 'Brain Neural Activity',
      description: 'Cognitive processing visualization',
      icon: <Brain className='h-5 w-5' />,
      component: MemoizedBrainNeural,
      color: '#AF52DE',
      status: 'active',
    },
    {
      id: 'nri-fusion',
      title: 'NRI Fusion',
      description: 'Multi-modal data integration',
      icon: <Zap className='h-5 w-5' />,
      component: MemoizedNRIFusion,
      color: '#FFD60A',
      status: 'active',
    },
    {
      id: 'multimodal',
      title: 'Multi-Modal Network',
      description: 'Integrated assessment network',
      icon: <Activity className='h-5 w-5' />,
      component: MemoizedMultiModalNetwork,
      color: '#FF2D92',
      status: 'inactive',
    },
  ];

  const visualizationList = visualizations || defaultVisualizations;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return '#34C759';
      case 'processing':
        return '#FF9500';
      case 'inactive':
        return '#86868B';
      default:
        return '#86868B';
    }
  };

  const getStatusIndicator = (status: string) => {
    switch (status) {
      case 'active':
        return (
          <div
            className='h-2 w-2 animate-pulse rounded-full'
            style={{ backgroundColor: '#34C759' }}
          />
        );
      case 'processing':
        return (
          <div
            className='h-2 w-2 animate-spin rounded-full'
            style={{ backgroundColor: '#FF9500' }}
          />
        );
      case 'inactive':
        return <div className='h-2 w-2 rounded-full' style={{ backgroundColor: '#86868B' }} />;
      default:
        return null;
    }
  };

  const handleVisualizationClick = (id: string) => {
    if (onVisualizationClick) {
      onVisualizationClick(id);
    } else {
      console.log(`Visualization clicked: ${id}`);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className='rounded-2xl border p-6 backdrop-blur-xl'
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.6)',
        borderColor: 'rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header */}
      <div className='mb-6 flex items-center justify-between'>
        <div>
          <h3
            className='text-lg font-semibold'
            style={{
              color: '#1D1D1F',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            }}
          >
            Geometric Visualizations
          </h3>
          <p className='mt-1 text-sm' style={{ color: '#86868B' }}>
            Real-time anatomical and neurological analysis
          </p>
        </div>
        <button
          className='text-sm font-medium transition-opacity hover:opacity-70'
          style={{ color: '#007AFF' }}
        >
          Full Screen
        </button>
      </div>

      {/* Visualization Grid */}
      <div className='grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3'>
        {visualizationList.map((viz, index) => {
          const VisualizationComponent = viz.component;

          return (
            <motion.div
              key={viz.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className='group relative cursor-pointer overflow-hidden rounded-xl border transition-all duration-300 hover:shadow-lg'
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                borderColor: 'rgba(0, 0, 0, 0.1)',
                minHeight: '200px',
              }}
              onClick={() => handleVisualizationClick(viz.id)}
            >
              {/* Visualization Canvas */}
              <div className='relative h-48 overflow-hidden'>
                <VisualizationComponent />

                {/* Overlay Controls */}
                <div className='absolute inset-0 bg-black/0 transition-colors duration-300 group-hover:bg-black/10'>
                  <div className='absolute right-3 top-3 opacity-0 transition-opacity duration-300 group-hover:opacity-100'>
                    <button
                      className='rounded-full border p-2 backdrop-blur-xl'
                      style={{
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        borderColor: 'rgba(0, 0, 0, 0.1)',
                      }}
                      onClick={e => {
                        e.stopPropagation();
                        handleVisualizationClick(`${viz.id}-fullscreen`);
                      }}
                    >
                      <Maximize2 className='h-4 w-4' style={{ color: '#1D1D1F' }} />
                    </button>
                  </div>
                </div>
              </div>

              {/* Info Panel */}
              <div className='p-4'>
                <div className='mb-2 flex items-start justify-between'>
                  <div className='flex items-center space-x-3'>
                    <div
                      className='flex h-8 w-8 items-center justify-center rounded-full'
                      style={{ backgroundColor: `${viz.color}20` }}
                    >
                      {React.cloneElement(viz.icon as React.ReactElement, {
                        style: { color: viz.color },
                      })}
                    </div>
                    <div>
                      <h4 className='text-sm font-medium' style={{ color: '#1D1D1F' }}>
                        {viz.title}
                      </h4>
                    </div>
                  </div>
                  <div className='flex items-center space-x-2'>
                    {getStatusIndicator(viz.status)}
                    <span
                      className='text-xs font-medium capitalize'
                      style={{ color: getStatusColor(viz.status) }}
                    >
                      {viz.status}
                    </span>
                  </div>
                </div>
                <p className='line-clamp-2 text-xs' style={{ color: '#86868B' }}>
                  {viz.description}
                </p>
              </div>

              {/* Hover Effect */}
              <motion.div
                className='pointer-events-none absolute inset-0 rounded-xl opacity-0'
                style={{ backgroundColor: `${viz.color}05` }}
                whileHover={{ opacity: 1 }}
                transition={{ duration: 0.2 }}
              />
            </motion.div>
          );
        })}
      </div>

      {/* Footer */}
      <div
        className='mt-6 flex items-center justify-between border-t pt-4'
        style={{ borderColor: 'rgba(0, 0, 0, 0.1)' }}
      >
        <div className='text-sm' style={{ color: '#86868B' }}>
          All visualizations running at 60fps with hardware acceleration
        </div>
        <button
          className='text-sm font-medium transition-opacity hover:opacity-70'
          style={{ color: '#007AFF' }}
          onClick={() => handleVisualizationClick('settings')}
        >
          Visualization Settings
        </button>
      </div>
    </motion.div>
  );
}
