'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface AnimatedSectionProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  duration?: number;
  direction?: 'up' | 'down' | 'left' | 'right' | 'scale';
}

export function AnimatedSection({
  children,
  className = '',
  delay = 0,
  duration = 0.8,
  direction = 'up',
}: AnimatedSectionProps) {
  const getInitialState = () => {
    switch (direction) {
      case 'up':
        return { opacity: 0, y: 30 };
      case 'down':
        return { opacity: 0, y: -30 };
      case 'left':
        return { opacity: 0, x: -30 };
      case 'right':
        return { opacity: 0, x: 30 };
      case 'scale':
        return { opacity: 0, scale: 0.9 };
      default:
        return { opacity: 0, y: 30 };
    }
  };

  const getAnimateState = () => {
    switch (direction) {
      case 'up':
      case 'down':
        return { opacity: 1, y: 0 };
      case 'left':
      case 'right':
        return { opacity: 1, x: 0 };
      case 'scale':
        return { opacity: 1, scale: 1 };
      default:
        return { opacity: 1, y: 0 };
    }
  };

  return (
    <motion.div
      initial={getInitialState()}
      animate={getAnimateState()}
      transition={{
        duration,
        delay,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// Specialized animated components for common use cases
export function AnimatedHero({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <AnimatedSection direction='up' duration={0.8} className={className}>
      {children}
    </AnimatedSection>
  );
}

export function AnimatedCard({
  children,
  className = '',
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <AnimatedSection direction='scale' duration={0.6} delay={delay} className={className}>
      {children}
    </AnimatedSection>
  );
}

export function AnimatedList({
  children,
  className = '',
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <AnimatedSection direction='up' duration={0.6} delay={delay} className={className}>
      {children}
    </AnimatedSection>
  );
}

// Complex animated components for About page
export function AnimatedGeometricDesign({ className = '' }: { className?: string }) {
  return (
    <AnimatedSection direction='scale' duration={0.8} delay={0.2} className={className}>
      <div className='relative mx-auto h-32 w-32'>
        <div className='absolute inset-0 animate-pulse rounded-full bg-gradient-to-r from-blue-500 to-purple-600 opacity-20'></div>
        <div
          className='absolute inset-2 animate-pulse rounded-full bg-gradient-to-r from-purple-500 to-pink-500 opacity-30'
          style={{ animationDelay: '0.5s' }}
        ></div>
        <div
          className='absolute inset-4 animate-pulse rounded-full bg-gradient-to-r from-pink-500 to-red-500 opacity-40'
          style={{ animationDelay: '1s' }}
        ></div>
        <div
          className='absolute inset-6 animate-pulse rounded-full bg-gradient-to-r from-red-500 to-orange-500 opacity-50'
          style={{ animationDelay: '1.5s' }}
        ></div>
        <div
          className='absolute inset-8 animate-pulse rounded-full bg-gradient-to-r from-orange-500 to-yellow-500 opacity-60'
          style={{ animationDelay: '2s' }}
        ></div>
      </div>
    </AnimatedSection>
  );
}

export default AnimatedSection;
