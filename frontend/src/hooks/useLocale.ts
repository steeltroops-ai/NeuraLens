/**
 * useLocale Hook
 * 
 * React hook for managing locale/language preferences.
 * Supports persistence and automatic detection.
 * 
 * Requirements: 14.1-14.6
 * 
 * @module hooks/useLocale
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  SupportedLocale,
  RetinalTranslations,
  getTranslations,
  getDefaultLocale,
  setLocale as saveLocale,
  t as translate,
} from '@/lib/i18n/retinal-translations';

// ============================================================================
// Types
// ============================================================================

interface UseLocaleReturn {
  /** Current locale */
  locale: SupportedLocale;
  /** Set new locale */
  setLocale: (locale: SupportedLocale) => void;
  /** All translations for current locale */
  translations: RetinalTranslations;
  /** Get translation by key path */
  t: (keyPath: string) => string;
  /** Available locales */
  availableLocales: SupportedLocale[];
  /** Locale display names */
  localeNames: Record<SupportedLocale, string>;
  /** Is locale loading */
  isLoading: boolean;
}

// ============================================================================
// Hook
// ============================================================================

export function useLocale(): UseLocaleReturn {
  const [locale, setLocaleState] = useState<SupportedLocale>('en');
  const [isLoading, setIsLoading] = useState(true);

  // Available locales with display names
  const availableLocales: SupportedLocale[] = ['en', 'es', 'zh'];
  const localeNames: Record<SupportedLocale, string> = {
    en: 'English',
    es: 'Español',
    zh: '中文',
  };

  // Initialize locale on mount
  useEffect(() => {
    const defaultLocale = getDefaultLocale();
    setLocaleState(defaultLocale);
    setIsLoading(false);
  }, []);

  // Set and persist locale
  const setLocale = useCallback((newLocale: SupportedLocale) => {
    setLocaleState(newLocale);
    saveLocale(newLocale);
    
    // Update document lang attribute
    if (typeof document !== 'undefined') {
      document.documentElement.lang = newLocale;
    }
  }, []);

  // Get translations for current locale
  const translations = getTranslations(locale);

  // Translation function
  const t = useCallback((keyPath: string): string => {
    return translate(locale, keyPath);
  }, [locale]);

  return {
    locale,
    setLocale,
    translations,
    t,
    availableLocales,
    localeNames,
    isLoading,
  };
}

export default useLocale;
