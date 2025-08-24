'use client';

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2 } from 'lucide-react';

/**
 * Global loading state component that shows during navigation
 * Prevents white screen flashes during route transitions
 */
export function GlobalLoadingState() {
  const pathname = usePathname();
  const [isLoading, setIsLoading] = useState(false);
  const [loadingTimeout, setLoadingTimeout] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Clear any existing timeout
    if (loadingTimeout) {
      clearTimeout(loadingTimeout);
    }

    // Show loading state immediately on route change
    setIsLoading(true);

    // Hide loading state after a short delay to allow page to render
    const timeout = setTimeout(() => {
      setIsLoading(false);
    }, 300); // 300ms should be enough for most page transitions

    setLoadingTimeout(timeout);

    return () => {
      if (timeout) {
        clearTimeout(timeout);
      }
    };
  }, [pathname]);

  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-white/80 backdrop-blur-sm"
          style={{ pointerEvents: 'none' }}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ duration: 0.2, delay: 0.1 }}
            className="flex flex-col items-center gap-3"
          >
            <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
            <p className="text-sm font-medium text-gray-600">Loading...</p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/**
 * Hook to manually control global loading state
 */
export function useGlobalLoading() {
  const [isLoading, setIsLoading] = useState(false);

  const showLoading = () => setIsLoading(true);
  const hideLoading = () => setIsLoading(false);

  return {
    isLoading,
    showLoading,
    hideLoading,
  };
}

/**
 * Loading overlay for specific components
 */
interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  message?: string;
  className?: string;
}

export function LoadingOverlay({ 
  isLoading, 
  children, 
  message = 'Loading...', 
  className = '' 
}: LoadingOverlayProps) {
  return (
    <div className={`relative ${className}`}>
      {children}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-white/90 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ duration: 0.2, delay: 0.1 }}
              className="flex flex-col items-center gap-3 rounded-lg bg-white p-6 shadow-lg"
            >
              <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              <p className="text-sm font-medium text-gray-600">{message}</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/**
 * Page transition loading component
 */
export function PageTransitionLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col items-center gap-4"
      >
        <div className="relative">
          <div className="h-12 w-12 rounded-full border-4 border-gray-200"></div>
          <div className="absolute top-0 h-12 w-12 animate-spin rounded-full border-4 border-transparent border-t-blue-600"></div>
        </div>
        <p className="text-lg font-medium text-gray-700">Loading page...</p>
        <p className="text-sm text-gray-500">Please wait while we prepare your content</p>
      </motion.div>
    </div>
  );
}

/**
 * Skeleton loading for content areas
 */
interface SkeletonLoadingProps {
  lines?: number;
  className?: string;
}

export function SkeletonLoading({ lines = 3, className = '' }: SkeletonLoadingProps) {
  return (
    <div className={`animate-pulse space-y-3 ${className}`}>
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={index}
          className={`h-4 rounded bg-gray-200 ${
            index === lines - 1 ? 'w-3/4' : 'w-full'
          }`}
        />
      ))}
    </div>
  );
}

export default GlobalLoadingState;
