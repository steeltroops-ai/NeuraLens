'use client';

import { useState, useCallback } from 'react';

interface LogoutState {
  isLoggingOut: boolean;
  error: string | null;
}

interface UseLogoutReturn {
  isLoggingOut: boolean;
  error: string | null;
  logout: () => Promise<void>;
  clearError: () => void;
}

/**
 * Enhanced logout hook with comprehensive session clearing and error handling
 * Provides loading states, error management, and consistent redirect behavior
 */
export function useLogout(): UseLogoutReturn {
  const [state, setState] = useState<LogoutState>({
    isLoggingOut: false,
    error: null,
  });

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  const logout = useCallback(async () => {
    try {
      // Set loading state
      setState(prev => ({ ...prev, isLoggingOut: true, error: null }));

      // Clear authentication data from localStorage
      const authKeys = [
        'auth-token',
        'user-data',
        'refresh-token',
        'session-id',
        'neuralens-auth',
        'access-token',
      ];

      authKeys.forEach(key => {
        try {
          localStorage.removeItem(key);
        } catch (error) {
          console.warn(`Failed to remove localStorage key: ${key}`, error);
        }
      });

      // Clear session storage
      try {
        sessionStorage.clear();
      } catch (error) {
        console.warn('Failed to clear sessionStorage:', error);
      }

      // Clear any cookies (if using cookie-based auth)
      try {
        document.cookie.split(';').forEach(c => {
          const eqPos = c.indexOf('=');
          const name = eqPos > -1 ? c.substr(0, eqPos) : c;
          document.cookie = name + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
        });
      } catch (error) {
        console.warn('Failed to clear cookies:', error);
      }

      // Call logout API if available
      try {
        const response = await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });

        if (!response.ok) {
          console.warn('Logout API call failed, but continuing with client-side logout');
        }
      } catch (apiError) {
        console.warn('Logout API not available or failed:', apiError);
        // Continue with logout even if API fails
      }

      // Small delay to ensure all cleanup is complete
      await new Promise(resolve => setTimeout(resolve, 100));

      // Force redirect to home page
      window.location.href = '/';
    } catch (error) {
      console.error('Logout error:', error);

      setState(prev => ({
        ...prev,
        isLoggingOut: false,
        error: error instanceof Error ? error.message : 'Logout failed. Please try again.',
      }));

      // Even on error, attempt to redirect after a short delay
      setTimeout(() => {
        window.location.href = '/';
      }, 2000);
    }
  }, []);

  return {
    isLoggingOut: state.isLoggingOut,
    error: state.error,
    logout,
    clearError,
  };
}

/**
 * Utility function to check if user is authenticated
 * Can be used to determine if logout functionality should be available
 */
export function useAuthStatus() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;

    try {
      // Check for any authentication tokens
      const authToken =
        localStorage.getItem('auth-token') ||
        localStorage.getItem('access-token') ||
        localStorage.getItem('neuralens-auth');

      return !!authToken;
    } catch (error) {
      console.warn('Failed to check auth status:', error);
      return false;
    }
  });

  const checkAuthStatus = useCallback(() => {
    try {
      const authToken =
        localStorage.getItem('auth-token') ||
        localStorage.getItem('access-token') ||
        localStorage.getItem('neuralens-auth');

      const newAuthStatus = !!authToken;
      setIsAuthenticated(newAuthStatus);
      return newAuthStatus;
    } catch (error) {
      console.warn('Failed to check auth status:', error);
      setIsAuthenticated(false);
      return false;
    }
  }, []);

  return {
    isAuthenticated,
    checkAuthStatus,
  };
}

/**
 * Enhanced logout function with additional safety measures
 * Can be used as a standalone function without the hook
 */
export async function performLogout(): Promise<void> {
  try {
    // Clear all possible authentication data
    const storageKeys = [
      'auth-token',
      'user-data',
      'refresh-token',
      'session-id',
      'neuralens-auth',
      'access-token',
      'user-session',
      'jwt-token',
    ];

    // Clear localStorage
    storageKeys.forEach(key => {
      try {
        localStorage.removeItem(key);
      } catch (error) {
        console.warn(`Failed to remove localStorage key: ${key}`, error);
      }
    });

    // Clear sessionStorage
    try {
      sessionStorage.clear();
    } catch (error) {
      console.warn('Failed to clear sessionStorage:', error);
    }

    // Clear IndexedDB if used for auth
    try {
      if ('indexedDB' in window) {
        // Clear any auth-related IndexedDB data
        // This is a basic implementation - adjust based on your IndexedDB usage
        const deleteDB = indexedDB.deleteDatabase('neuralens-auth');
        deleteDB.onerror = () => console.warn('Failed to clear IndexedDB');
      }
    } catch (error) {
      console.warn('Failed to clear IndexedDB:', error);
    }

    // Call logout API
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('Logout API call failed:', error);
    }

    // Force page reload and redirect
    window.location.replace('/');
  } catch (error) {
    console.error('Critical logout error:', error);
    // Force redirect even on critical error
    window.location.replace('/');
  }
}
