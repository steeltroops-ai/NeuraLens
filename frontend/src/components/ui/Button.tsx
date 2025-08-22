'use client';

import React, { forwardRef } from 'react';
import type { ButtonProps } from '@/types/design-system';
import { cn } from '@/utils/cn';

/**
 * Clinical-grade Button component with accessibility and animation support
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      disabled = false,
      loading = false,
      leftIcon,
      rightIcon,
      children,
      className,
      testId,
      onClick,
      type = 'button',
      ...props
    },
    ref
  ) => {
    // Apple-style base button classes
    const baseClasses = [
      'btn-apple',
      'apple-focus',
      'font-semibold',
      'transition-all',
      'duration-150',
      'ease-out',
      'active:animate-button-press',
      'disabled:opacity-50',
      'disabled:cursor-not-allowed',
      'disabled:pointer-events-none',
    ];

    // Apple HIG size classes (minimum 44px touch targets)
    const sizeClasses = {
      sm: ['px-4', 'py-2', 'text-sm', 'min-h-[36px]', 'rounded-apple'],
      md: ['px-6', 'py-3', 'text-base', 'min-h-[44px]', 'rounded-apple'],
      lg: ['px-8', 'py-4', 'text-lg', 'min-h-[48px]', 'rounded-apple-lg'],
      xl: ['px-10', 'py-5', 'text-xl', 'min-h-[52px]', 'rounded-apple-xl'],
    };

    // Apple-inspired variant classes
    const variantClasses = {
      primary: [
        'btn-primary',
        'text-white',
        'shadow-medical',
        'hover:shadow-medical-hover',
        'hover:-translate-y-0.5',
        'active:translate-y-0',
        'active:shadow-apple-pressed',
      ],
      secondary: [
        'btn-secondary',
        'text-text-primary',
        'border',
        'border-gray-300',
        'hover:bg-gray-100',
        'hover:-translate-y-0.5',
        'active:translate-y-0',
      ],
      ghost: [
        'bg-transparent',
        'text-medical-500',
        'hover:bg-medical-50',
        'hover:text-medical-600',
        'border-transparent',
        'rounded-apple',
      ],
      destructive: [
        'bg-gradient-to-r',
        'from-error-500',
        'to-error-600',
        'text-white',
        'shadow-lg',
        'hover:shadow-xl',
        'hover:-translate-y-0.5',
        'active:translate-y-0',
      ],
    };

    // Combine all classes
    const buttonClasses = cn(
      ...baseClasses,
      ...sizeClasses[size],
      ...variantClasses[variant],
      loading && 'pointer-events-none',
      className
    );

    // Handle click with loading state
    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
      if (loading || disabled) {
        event.preventDefault();
        return;
      }
      onClick?.(event);
    };

    return (
      <button
        ref={ref}
        type={type}
        className={buttonClasses}
        disabled={disabled || loading}
        data-testid={testId}
        onClick={handleClick}
        aria-disabled={disabled || loading}
        aria-busy={loading}
        {...props}
      >
        {/* Loading Spinner */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-current rounded-full animate-spin border-t-transparent" />
          </div>
        )}

        {/* Button Content */}
        <div className={cn('flex items-center gap-2', loading && 'opacity-0')}>
          {leftIcon && (
            <span className="flex-shrink-0" aria-hidden="true">
              {leftIcon}
            </span>
          )}

          {children && <span className="flex-1">{children}</span>}

          {rightIcon && (
            <span className="flex-shrink-0" aria-hidden="true">
              {rightIcon}
            </span>
          )}
        </div>

        {/* Ripple Effect */}
        <div className="absolute inset-0 overflow-hidden rounded-inherit">
          <div className="absolute inset-0 transition-opacity duration-150 bg-white opacity-0 hover:opacity-10" />
        </div>
      </button>
    );
  }
);

Button.displayName = 'Button';

/**
 * Icon Button variant for compact actions
 */
export const IconButton = forwardRef<
  HTMLButtonElement,
  ButtonProps & { icon: React.ReactNode }
>(({ icon, size = 'md', className, ...props }, ref) => {
  const iconSizes = {
    sm: 'w-8 h-8',
    md: 'w-10 h-10',
    lg: 'w-12 h-12',
    xl: 'w-14 h-14',
  };

  return (
    <Button
      ref={ref}
      size={size}
      className={cn(
        'rounded-full',
        'aspect-square',
        'p-0',
        iconSizes[size],
        className
      )}
      {...props}
    >
      <span className="flex items-center justify-center">{icon}</span>
    </Button>
  );
});

IconButton.displayName = 'IconButton';

/**
 * Button Group for related actions
 */
interface ButtonGroupProps {
  children: React.ReactNode;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
  spacing?: 'none' | 'sm' | 'md' | 'lg';
}

export const ButtonGroup: React.FC<ButtonGroupProps> = ({
  children,
  className,
  orientation = 'horizontal',
  spacing = 'sm',
}) => {
  const orientationClasses = {
    horizontal: 'flex-row',
    vertical: 'flex-col',
  };

  const spacingClasses = {
    none: 'gap-0',
    sm: 'gap-2',
    md: 'gap-4',
    lg: 'gap-6',
  };

  return (
    <div
      className={cn(
        'flex',
        orientationClasses[orientation],
        spacingClasses[spacing],
        className
      )}
      role="group"
    >
      {children}
    </div>
  );
};

/**
 * Loading Button with built-in loading state management
 */
interface LoadingButtonProps extends Omit<ButtonProps, 'loading'> {
  onAsyncClick?: () => Promise<void>;
}

export const LoadingButton: React.FC<LoadingButtonProps> = ({
  onAsyncClick,
  onClick,
  children,
  ...props
}) => {
  const [loading, setLoading] = React.useState(false);

  const handleClick = async (event: React.MouseEvent<HTMLButtonElement>) => {
    if (onAsyncClick) {
      setLoading(true);
      try {
        await onAsyncClick();
      } catch (error) {
        console.error('Async button action failed:', error);
      } finally {
        setLoading(false);
      }
    } else {
      onClick?.(event);
    }
  };

  return (
    <Button {...props} loading={loading} onClick={handleClick}>
      {children}
    </Button>
  );
};

/**
 * Copy Button with built-in copy functionality
 */
interface CopyButtonProps extends Omit<ButtonProps, 'onClick'> {
  textToCopy: string;
  successMessage?: string;
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  textToCopy,
  successMessage = 'Copied!',
  children,
  ...props
}) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);

      // Reset after 2 seconds
      setTimeout(() => setCopied(false), 2000);

      // Announce to screen readers
      const announcement = document.createElement('div');
      announcement.setAttribute('aria-live', 'polite');
      announcement.className = 'sr-only';
      announcement.textContent = successMessage;
      document.body.appendChild(announcement);

      setTimeout(() => {
        document.body.removeChild(announcement);
      }, 1000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  return (
    <Button
      {...props}
      onClick={handleCopy}
      aria-label={copied ? successMessage : `Copy ${textToCopy}`}
    >
      {copied ? (
        <>
          <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
          {successMessage}
        </>
      ) : (
        <>
          <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
            <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
          </svg>
          {children}
        </>
      )}
    </Button>
  );
};

export default Button;
