'use client';

import React, { forwardRef } from 'react';

import { cn } from '@/utils/cn';

import type { CardProps } from '@/types/design-system';

/**
 * Clinical-grade Card component with multiple variants and accessibility support
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      variant = 'default',
      padding = 'component-padding-lg',
      hover = false,
      onClick,
      children,
      className,
      testId,
      ...props
    },
    ref,
  ) => {
    // Apple-style base card classes
    const baseClasses = [
      'card-apple',
      'relative',
      'transition-all',
      'duration-300',
      'ease-out-quint',
    ];

    // Apple-style padding classes
    const paddingClasses = {
      'component-padding-sm': 'p-4',
      'component-padding-md': 'p-6',
      'component-padding-lg': 'p-8',
      'component-padding-xl': 'p-10',
    };

    // Apple-inspired variant classes
    const variantClasses = {
      default: ['bg-white', 'border', 'border-gray-200', 'shadow-apple'],
      glass: [
        'glass-card',
        'bg-surface-glass',
        'backdrop-blur-xl',
        'border',
        'border-white/10',
        'shadow-glass',
      ],
      clinical: ['bg-surface-primary', 'border-2', 'border-primary-500', 'shadow-clinical'],
      results: [
        'bg-gradient-to-br',
        'from-surface-primary',
        'to-surface-secondary',
        'border',
        'border-primary-500/30',
        'shadow-xl',
      ],
    };

    // Hover effects
    const hoverClasses = hover
      ? [
          'hover:border-neutral-700',
          'hover:shadow-xl',
          'hover:-translate-y-1',
          'hover:scale-[1.01]',
          'cursor-pointer',
        ]
      : [];

    // Interactive classes for clickable cards
    const interactiveClasses = onClick
      ? [
          'cursor-pointer',
          'focus:outline-none',
          'focus:ring-2',
          'focus:ring-primary-500',
          'focus:ring-offset-2',
          'focus:ring-offset-surface-background',
          'active:scale-[0.99]',
        ]
      : [];

    // Combine all classes
    const cardClasses = cn(
      ...baseClasses,
      paddingClasses[padding],
      ...variantClasses[variant],
      ...hoverClasses,
      ...interactiveClasses,
      className,
    );

    // Handle keyboard interaction
    const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (onClick && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        onClick();
      }
    };

    return (
      <div
        ref={ref}
        className={cardClasses}
        onClick={onClick}
        onKeyDown={handleKeyDown}
        tabIndex={onClick ? 0 : undefined}
        role={onClick ? 'button' : undefined}
        data-testid={testId}
        {...props}
      >
        {children}
      </div>
    );
  },
);

Card.displayName = 'Card';

/**
 * Card Header component
 */
interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
  actions?: React.ReactNode;
}

export const CardHeader: React.FC<CardHeaderProps> = ({ children, className, actions }) => {
  return (
    <div className={cn('mb-4 flex items-center justify-between', className)}>
      <div className='flex-1'>{children}</div>
      {actions && <div className='ml-4 flex items-center gap-2'>{actions}</div>}
    </div>
  );
};

/**
 * Card Title component
 */
interface CardTitleProps {
  children: React.ReactNode;
  className?: string;
  level?: 1 | 2 | 3 | 4 | 5 | 6;
}

export const CardTitle: React.FC<CardTitleProps> = ({ children, className, level = 3 }) => {
  const Tag = `h${level}` as keyof JSX.IntrinsicElements;

  const levelClasses = {
    1: 'text-3xl font-bold',
    2: 'text-2xl font-bold',
    3: 'text-xl font-semibold',
    4: 'text-lg font-semibold',
    5: 'text-base font-medium',
    6: 'text-sm font-medium',
  };

  return (
    <Tag
      className={cn('text-text-primary', 'leading-tight', 'mb-2', levelClasses[level], className)}
    >
      {children}
    </Tag>
  );
};

/**
 * Card Description component
 */
interface CardDescriptionProps {
  children: React.ReactNode;
  className?: string;
}

export const CardDescription: React.FC<CardDescriptionProps> = ({ children, className }) => {
  return (
    <p className={cn('text-text-secondary', 'text-sm', 'leading-relaxed', 'mb-0', className)}>
      {children}
    </p>
  );
};

/**
 * Card Content component
 */
interface CardContentProps {
  children: React.ReactNode;
  className?: string;
}

export const CardContent: React.FC<CardContentProps> = ({ children, className }) => {
  return <div className={cn('space-y-4', className)}>{children}</div>;
};

/**
 * Card Footer component
 */
interface CardFooterProps {
  children: React.ReactNode;
  className?: string;
  justify?: 'start' | 'center' | 'end' | 'between';
}

export const CardFooter: React.FC<CardFooterProps> = ({ children, className, justify = 'end' }) => {
  const justifyClasses = {
    start: 'justify-start',
    center: 'justify-center',
    end: 'justify-end',
    between: 'justify-between',
  };

  return (
    <div
      className={cn(
        'flex',
        'items-center',
        'gap-3',
        'mt-6',
        'pt-4',
        'border-t',
        'border-neutral-800',
        justifyClasses[justify],
        className,
      )}
    >
      {children}
    </div>
  );
};

/**
 * Assessment Card - Specialized card for assessment steps
 */
interface AssessmentCardProps extends Omit<CardProps, 'variant'> {
  title: string;
  description: string;
  icon: React.ReactNode;
  status: 'pending' | 'current' | 'completed' | 'error';
  estimatedTime?: string;
  onStart?: () => void;
}

export const AssessmentCard: React.FC<AssessmentCardProps> = ({
  title,
  description,
  icon,
  status,
  estimatedTime,
  onStart,
  className,
  ...props
}) => {
  const statusClasses = {
    pending: 'opacity-60',
    current: 'border-primary-500 shadow-clinical',
    completed: 'border-success bg-success/5',
    error: 'border-error bg-error/5',
  };

  const statusIcons = {
    pending: '○',
    current: '⟳',
    completed: '✓',
    error: '✗',
  };

  return (
    <Card
      variant='default'
      hover={status === 'pending' || status === 'current'}
      {...(onStart && { onClick: onStart })}
      className={cn('transition-all duration-300', statusClasses[status], className)}
      {...props}
    >
      <CardHeader>
        <div className='flex items-center gap-4'>
          <div className='flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-primary-500/10'>
            {icon}
          </div>
          <div className='flex-1'>
            <CardTitle level={4}>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <div className='flex-shrink-0 text-2xl'>{statusIcons[status]}</div>
        </div>
      </CardHeader>

      {estimatedTime && (
        <CardFooter justify='between'>
          <span className='text-text-muted text-sm'>Estimated time: {estimatedTime}</span>
          {(status === 'pending' || status === 'current') && (
            <span className='text-sm text-primary-400'>Click to start →</span>
          )}
        </CardFooter>
      )}
    </Card>
  );
};

/**
 * Results Card - Specialized card for displaying assessment results
 */
interface ResultsCardProps extends Omit<CardProps, 'variant'> {
  title: string;
  score: number;
  category: 'low' | 'moderate' | 'high' | 'critical';
  findings: string[];
  confidence?: number;
}

export const ResultsCard: React.FC<ResultsCardProps> = ({
  title,
  score,
  category,
  findings,
  confidence,
  className,
  ...props
}) => {
  const categoryColors = {
    low: 'text-success border-success',
    moderate: 'text-warning border-warning',
    high: 'text-orange-500 border-orange-500',
    critical: 'text-error border-error',
  };

  const categoryLabels = {
    low: 'Low Risk',
    moderate: 'Moderate Risk',
    high: 'High Risk',
    critical: 'Critical Risk',
  };

  return (
    <Card
      variant='results'
      className={cn('border-l-4', categoryColors[category], className)}
      {...props}
    >
      <CardHeader>
        <CardTitle level={4}>{title}</CardTitle>
        <div className='text-right'>
          <div className='text-3xl font-bold text-text-primary'>{score}/100</div>
          <div className={cn('text-sm font-medium', categoryColors[category])}>
            {categoryLabels[category]}
          </div>
          {confidence && <div className='text-text-muted text-xs'>±{confidence}% confidence</div>}
        </div>
      </CardHeader>

      <CardContent>
        <div className='mb-4 h-2 w-full rounded-full bg-neutral-800'>
          <div
            className={cn(
              'h-2 rounded-full transition-all duration-1000',
              category === 'low' && 'bg-success',
              category === 'moderate' && 'bg-warning',
              category === 'high' && 'bg-orange-500',
              category === 'critical' && 'bg-error',
            )}
            style={{ width: `${score}%` }}
          />
        </div>

        {findings.length > 0 && (
          <div>
            <h5 className='mb-2 text-sm font-medium text-text-primary'>Key Findings:</h5>
            <ul className='space-y-1'>
              {findings.map((finding, index) => (
                <li key={index} className='flex items-start gap-2 text-sm text-text-secondary'>
                  <span className='mt-1 text-primary-400'>•</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default Card;
