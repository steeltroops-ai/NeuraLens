'use client';

import React, { forwardRef, useState } from 'react';

import { cn } from '@/utils/cn';

import type { InputProps } from '@/types/design-system';

/**
 * MediLens Design System Input Component
 * 
 * Clinical-grade input with validation and accessibility support.
 * Follows MediLens Design System specifications:
 * - h-12, rounded-xl, bg-surface-secondary
 * - Focus states with medilens-blue-500 border and ring
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      type = 'text',
      placeholder,
      value,
      defaultValue,
      disabled = false,
      required = false,
      error,
      helperText,
      leftIcon,
      rightIcon,
      min,
      max,
      step,
      className,
      testId,
      onChange,
      onBlur,
      onFocus,
      ...props
    },
    ref,
  ) => {
    const [isFocused, setIsFocused] = useState(false);
    const [hasValue, setHasValue] = useState(Boolean(value || defaultValue));

    // Generate unique IDs for accessibility
    const inputId = React.useId();
    const errorId = error ? `${inputId}-error` : undefined;
    const helperId = helperText ? `${inputId}-helper` : undefined;

    // MediLens Design System base input classes
    const baseClasses = [
      'w-full',
      'h-12',           // MediLens: h-12 height
      'px-4',
      'bg-surface-secondary',  // MediLens: bg-surface-secondary
      'text-text-primary',
      'text-[17px]',    // MediLens body text size
      'rounded-xl',     // MediLens: rounded-xl
      'border',
      'border-transparent',
      'transition-all',
      'duration-200',
      'ease-out-quint',
      'placeholder:text-text-quaternary',  // MediLens placeholder color
      // MediLens focus states
      'focus:outline-none',
      'focus:bg-white',
      'focus:border-medilens-blue-500',
      'focus:ring-4',
      'focus:ring-medilens-blue-500/20',
      // Disabled states
      'disabled:opacity-50',
      'disabled:cursor-not-allowed',
      'disabled:bg-surface-tertiary',
    ];

    // State-dependent classes
    const stateClasses = error
      ? ['border-error-500', 'focus:ring-error-500/20', 'focus:border-error-500', 'bg-error-50']
      : [];

    // Icon spacing classes
    const iconClasses = {
      left: leftIcon ? 'pl-12' : '',
      right: rightIcon ? 'pr-12' : '',
    };

    // Combine all classes
    const inputClasses = cn(
      ...baseClasses,
      ...stateClasses,
      iconClasses.left,
      iconClasses.right,
      className,
    );

    // Handle input changes
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      setHasValue(Boolean(event.target.value));
      onChange?.(event);
    };

    const handleFocus = (event: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(true);
      onFocus?.(event);
    };

    const handleBlur = (event: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(false);
      onBlur?.(event);
    };

    return (
      <div className='relative'>
        {/* Left Icon */}
        {leftIcon && (
          <div className='text-text-tertiary pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 transform' aria-hidden='true'>
            {leftIcon}
          </div>
        )}

        {/* Input Element */}
        <input
          ref={ref}
          id={inputId}
          type={type}
          placeholder={placeholder}
          value={value}
          defaultValue={defaultValue}
          disabled={disabled}
          required={required}
          min={min}
          max={max}
          step={step}
          className={inputClasses}
          data-testid={testId}
          onChange={handleChange}
          onBlur={handleBlur}
          onFocus={handleFocus}
          aria-invalid={Boolean(error)}
          aria-required={required}
          aria-describedby={cn(errorId, helperId).trim() || undefined}
          {...props}
        />

        {/* Right Icon */}
        {rightIcon && (
          <div className='text-text-tertiary pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 transform' aria-hidden='true'>
            {rightIcon}
          </div>
        )}

        {/* Floating Label (if no placeholder) */}
        {!placeholder && (
          <label
            htmlFor={inputId}
            className={cn(
              'pointer-events-none absolute left-4 transition-all duration-200',
              'text-text-tertiary',
              isFocused || hasValue
                ? 'top-2 text-xs text-medilens-blue-500'
                : 'top-1/2 -translate-y-1/2 text-base',
            )}
          >
            {props['aria-label'] || 'Input'}
            {required && <span className='text-error-500 ml-1'>*</span>}
          </label>
        )}

        {/* Error Message */}
        {error && (
          <div
            id={errorId}
            className='text-error-500 mt-2 flex items-center gap-2 text-sm'
            role='alert'
            aria-live='polite'
          >
            <svg className='h-4 w-4 flex-shrink-0' fill='currentColor' viewBox='0 0 20 20'>
              <path
                fillRule='evenodd'
                d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z'
                clipRule='evenodd'
              />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Helper Text */}
        {helperText && !error && (
          <div id={helperId} className='text-text-tertiary mt-2 text-sm'>
            {helperText}
          </div>
        )}
      </div>
    );
  },
);

Input.displayName = 'Input';

/**
 * MediLens Design System Textarea Component
 * 
 * Textarea with similar styling and functionality to Input.
 */
interface TextareaProps extends Omit<InputProps, 'type' | 'leftIcon' | 'rightIcon'> {
  rows?: number;
  resize?: 'none' | 'vertical' | 'horizontal' | 'both';
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      placeholder,
      value,
      defaultValue,
      disabled = false,
      required = false,
      error,
      helperText,
      rows = 4,
      resize = 'vertical',
      className,
      testId,
      onChange,
      onBlur,
      onFocus,
      ...props
    },
    ref,
  ) => {
    // Generate unique IDs for accessibility
    const textareaId = React.useId();
    const errorId = error ? `${textareaId}-error` : undefined;
    const helperId = helperText ? `${textareaId}-helper` : undefined;

    // MediLens Design System base textarea classes
    const baseClasses = [
      'w-full',
      'px-4',
      'py-3',
      'bg-surface-secondary',
      'text-text-primary',
      'text-[17px]',
      'rounded-xl',
      'border',
      'border-transparent',
      'transition-all',
      'duration-200',
      'ease-out-quint',
      'placeholder:text-text-quaternary',
      // MediLens focus states
      'focus:outline-none',
      'focus:bg-white',
      'focus:border-medilens-blue-500',
      'focus:ring-4',
      'focus:ring-medilens-blue-500/20',
      // Disabled states
      'disabled:opacity-50',
      'disabled:cursor-not-allowed',
      'disabled:bg-surface-tertiary',
    ];

    // State-dependent classes
    const stateClasses = error
      ? ['border-error-500', 'focus:ring-error-500/20', 'focus:border-error-500', 'bg-error-50']
      : [];

    // Resize classes
    const resizeClasses = {
      none: 'resize-none',
      vertical: 'resize-y',
      horizontal: 'resize-x',
      both: 'resize',
    };

    // Combine all classes
    const textareaClasses = cn(...baseClasses, ...stateClasses, resizeClasses[resize], className);

    const handleFocus = (event: React.FocusEvent<HTMLTextAreaElement>) => {
      onFocus?.(event as any);
    };

    const handleBlur = (event: React.FocusEvent<HTMLTextAreaElement>) => {
      onBlur?.(event as any);
    };

    return (
      <div className='relative'>
        {/* Textarea Element */}
        <textarea
          ref={ref}
          id={textareaId}
          placeholder={placeholder}
          value={value}
          defaultValue={defaultValue}
          disabled={disabled}
          required={required}
          rows={rows}
          className={textareaClasses}
          data-testid={testId}
          onChange={onChange as any}
          onBlur={handleBlur}
          onFocus={handleFocus}
          aria-invalid={Boolean(error)}
          aria-required={required}
          aria-describedby={cn(errorId, helperId).trim() || undefined}
          {...props}
        />

        {/* Error Message */}
        {error && (
          <div
            id={errorId}
            className='text-error-500 mt-2 flex items-center gap-2 text-sm'
            role='alert'
            aria-live='polite'
          >
            <svg className='h-4 w-4 flex-shrink-0' fill='currentColor' viewBox='0 0 20 20'>
              <path
                fillRule='evenodd'
                d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z'
                clipRule='evenodd'
              />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Helper Text */}
        {helperText && !error && (
          <div id={helperId} className='text-text-tertiary mt-2 text-sm'>
            {helperText}
          </div>
        )}
      </div>
    );
  },
);

Textarea.displayName = 'Textarea';

/**
 * Search Input with built-in search functionality
 */
interface SearchInputProps extends Omit<InputProps, 'type' | 'leftIcon'> {
  onSearch?: (query: string) => void;
  onClear?: () => void;
  showClearButton?: boolean;
}

export const SearchInput: React.FC<SearchInputProps> = ({
  onSearch,
  onClear,
  showClearButton = true,
  placeholder = 'Search...',
  value,
  onChange,
  className,
  ...props
}) => {
  const [searchValue, setSearchValue] = useState(value || '');

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    setSearchValue(newValue);
    onChange?.(event);

    // Debounced search
    const timeoutId = setTimeout(() => {
      onSearch?.(newValue);
    }, 300);

    return () => clearTimeout(timeoutId);
  };

  const handleClear = () => {
    setSearchValue('');
    onClear?.();

    // Create synthetic event for onChange
    const syntheticEvent = {
      target: { value: '' },
    } as React.ChangeEvent<HTMLInputElement>;
    onChange?.(syntheticEvent);
  };

  const searchIcon = (
    <svg className='h-5 w-5' fill='currentColor' viewBox='0 0 20 20'>
      <path
        fillRule='evenodd'
        d='M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z'
        clipRule='evenodd'
      />
    </svg>
  );

  const clearIcon =
    showClearButton && searchValue ? (
      <button
        type='button'
        onClick={handleClear}
        className='text-text-muted transition-colors hover:text-text-primary'
        aria-label='Clear search'
      >
        <svg className='h-5 w-5' fill='currentColor' viewBox='0 0 20 20'>
          <path
            fillRule='evenodd'
            d='M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z'
            clipRule='evenodd'
          />
        </svg>
      </button>
    ) : null;

  return (
    <Input
      type='search'
      placeholder={placeholder}
      value={searchValue}
      onChange={handleChange}
      leftIcon={searchIcon}
      rightIcon={clearIcon}
      {...(className && { className })}
      {...props}
    />
  );
};

export default Input;
