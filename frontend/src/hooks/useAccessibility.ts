/**
 * Accessibility React Hooks
 * Custom hooks for WCAG 2.1 AA compliance
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  FocusManager,
  ScreenReader,
  KeyboardNavigation,
  AriaUtils,
  MotionUtils,
} from '@/lib/accessibility/utils';

// Focus management hook
export function useFocusManagement() {
  const saveFocus = useCallback((newFocus?: HTMLElement) => {
    FocusManager.saveFocus(newFocus);
  }, []);

  const restoreFocus = useCallback(() => {
    FocusManager.restoreFocus();
  }, []);

  return { saveFocus, restoreFocus };
}

// Focus trap hook for modals and dialogs
export function useFocusTrap(isActive: boolean = false) {
  const containerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (isActive && containerRef.current) {
      FocusManager.trapFocus(containerRef.current);
      return () => {
        FocusManager.releaseFocusTrap();
      };
    }

    return undefined;
  }, [isActive]);

  return containerRef;
}

// Screen reader announcements hook
export function useScreenReader() {
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    ScreenReader.announce(message, priority);
  }, []);

  const announceProgress = useCallback((step: string, progress: number, total: number) => {
    ScreenReader.announceProgress(step, progress, total);
  }, []);

  const announceResults = useCallback((riskCategory: string, nriScore: number) => {
    ScreenReader.announceResults(riskCategory, nriScore);
  }, []);

  const announceError = useCallback((error: string) => {
    ScreenReader.announceError(error);
  }, []);

  return { announce, announceProgress, announceResults, announceError };
}

// Keyboard navigation hook
export function useKeyboardNavigation<T extends HTMLElement>(items: T[], initialIndex: number = 0) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      KeyboardNavigation.handleArrowNavigation(event, items, currentIndex, setCurrentIndex);
    },
    [items, currentIndex],
  );

  const handleEscape = useCallback((onEscape: () => void) => {
    return (event: KeyboardEvent) => {
      KeyboardNavigation.handleEscape(event, onEscape);
    };
  }, []);

  const handleActivation = useCallback((onActivate: () => void) => {
    return (event: KeyboardEvent) => {
      KeyboardNavigation.handleActivation(event, onActivate);
    };
  }, []);

  return {
    currentIndex,
    setCurrentIndex,
    handleKeyDown,
    handleEscape,
    handleActivation,
  };
}

// ARIA utilities hook
export function useAria() {
  const generateId = useCallback((prefix?: string) => {
    return AriaUtils.generateId(prefix);
  }, []);

  const setDescribedBy = useCallback((element: HTMLElement, descriptionId: string) => {
    AriaUtils.setDescribedBy(element, descriptionId);
  }, []);

  const removeDescribedBy = useCallback((element: HTMLElement, descriptionId: string) => {
    AriaUtils.removeDescribedBy(element, descriptionId);
  }, []);

  const setLabelledBy = useCallback((element: HTMLElement, labelId: string) => {
    AriaUtils.setLabelledBy(element, labelId);
  }, []);

  return { generateId, setDescribedBy, removeDescribedBy, setLabelledBy };
}

// Reduced motion hook
export function useReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const getSafeAnimationDuration = useCallback((normalDuration: number) => {
    return MotionUtils.getSafeAnimationDuration(normalDuration);
  }, []);

  const applySafeAnimation = useCallback((element: HTMLElement, animation: string) => {
    MotionUtils.applySafeAnimation(element, animation);
  }, []);

  return { prefersReducedMotion, getSafeAnimationDuration, applySafeAnimation };
}

// Live region hook for dynamic content updates
export function useLiveRegion(priority: 'polite' | 'assertive' = 'polite') {
  const liveRegionRef = useRef<HTMLDivElement>(null);
  const [currentMessage, setCurrentMessage] = useState<string>('');

  const announce = useCallback(
    (message: string) => {
      setCurrentMessage(message);
      if (liveRegionRef.current) {
        AriaUtils.updateLiveRegion(liveRegionRef.current, message, priority);
      }
    },
    [priority],
  );

  const LiveRegion = useCallback(
    ({ className = '' }: { className?: string }) => {
      return React.createElement(
        'div',
        {
          ref: liveRegionRef,
          'aria-live': priority,
          'aria-atomic': 'true',
          className: `sr-only ${className}`,
          style: {
            position: 'absolute',
            width: '1px',
            height: '1px',
            padding: '0',
            margin: '-1px',
            overflow: 'hidden',
            clip: 'rect(0, 0, 0, 0)',
            whiteSpace: 'nowrap',
            border: '0',
          },
        },
        currentMessage,
      );
    },
    [currentMessage, priority],
  );

  return { announce, LiveRegion };
}

// Skip link hook
export function useSkipLink() {
  const skipLinkRef = useRef<HTMLAnchorElement>(null);
  const targetRef = useRef<HTMLElement>(null);

  const handleSkip = useCallback((event: React.MouseEvent | React.KeyboardEvent) => {
    event.preventDefault();
    if (targetRef.current) {
      targetRef.current.focus();
      targetRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  const SkipLink = useCallback(
    ({
      href,
      children = 'Skip to main content',
      className = '',
    }: {
      href: string;
      children?: React.ReactNode;
      className?: string;
    }) => {
      return React.createElement(
        'a',
        {
          ref: skipLinkRef,
          href,
          onClick: handleSkip,
          onKeyDown: (e: React.KeyboardEvent) => {
            if (e.key === 'Enter' || e.key === ' ') {
              handleSkip(e as any);
            }
          },
          className: `sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-lg focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${className}`,
        },
        children,
      );
    },
    [handleSkip],
  );

  return { SkipLink, targetRef };
}

// Form accessibility hook
export function useFormAccessibility() {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  const getFieldProps = useCallback(
    (fieldName: string, label: string) => {
      const fieldId = AriaUtils.generateId(`field-${fieldName}`);
      const errorId = AriaUtils.generateId(`error-${fieldName}`);
      const hasError = errors[fieldName] && touched[fieldName];

      return {
        id: fieldId,
        'aria-label': label,
        'aria-describedby': hasError ? errorId : undefined,
        'aria-invalid': hasError ? 'true' : 'false',
        onBlur: () => setTouched(prev => ({ ...prev, [fieldName]: true })),
        errorId,
        hasError,
        errorMessage: errors[fieldName],
      };
    },
    [errors, touched],
  );

  const setFieldError = useCallback((fieldName: string, error: string) => {
    setErrors(prev => ({ ...prev, [fieldName]: error }));
  }, []);

  const clearFieldError = useCallback((fieldName: string) => {
    setErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[fieldName];
      return newErrors;
    });
  }, []);

  const clearAllErrors = useCallback(() => {
    setErrors({});
    setTouched({});
  }, []);

  return {
    getFieldProps,
    setFieldError,
    clearFieldError,
    clearAllErrors,
    hasErrors: Object.keys(errors).length > 0,
  };
}

// Assessment accessibility hook
export function useAssessmentAccessibility() {
  const { announce, announceProgress, announceResults, announceError } = useScreenReader();
  const { prefersReducedMotion } = useReducedMotion();

  const announceStepChange = useCallback(
    (step: string, stepNumber: number, totalSteps: number) => {
      announceProgress(step, stepNumber, totalSteps);
    },
    [announceProgress],
  );

  const announceAssessmentComplete = useCallback(
    (riskCategory: string, nriScore: number) => {
      announceResults(riskCategory, nriScore);
    },
    [announceResults],
  );

  const announceAssessmentError = useCallback(
    (error: string, step?: string) => {
      const message = step ? `Error in ${step}: ${error}` : error;
      announceError(message);
    },
    [announceError],
  );

  const announceFileUpload = useCallback(
    (fileName: string, fileType: string) => {
      announce(`${fileType} file uploaded: ${fileName}`);
    },
    [announce],
  );

  const announceValidationError = useCallback(
    (errors: string[]) => {
      const message = `Validation failed: ${errors.join(', ')}`;
      announceError(message);
    },
    [announceError],
  );

  return {
    announceStepChange,
    announceAssessmentComplete,
    announceAssessmentError,
    announceFileUpload,
    announceValidationError,
    prefersReducedMotion,
  };
}

// Color contrast hook
export function useColorContrast() {
  const checkContrast = useCallback(
    (foreground: string, background: string, isLargeText: boolean = false) => {
      // This would integrate with the ColorContrast utility
      // For now, return a mock implementation
      return {
        ratio: 4.5,
        meetsAA: true,
        meetsAAA: false,
      };
    },
    [],
  );

  return { checkContrast };
}
