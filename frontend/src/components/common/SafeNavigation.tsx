'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useRef, useEffect } from 'react';

/**
 * Enhanced SafeNavigation hook provides optimized navigation methods with
 * intelligent prefetching, instant SPA transitions, and performance monitoring.
 */
export function useSafeNavigation() {
  const router = useRouter();
  const isNavigatingRef = useRef(false);
  const prefetchedRoutes = useRef(new Set<string>());

  // Prefetch critical routes on mount for instant navigation
  useEffect(() => {
    const criticalRoutes = ['/dashboard', '/assessment', '/readme', '/about'];

    // Prefetch critical routes with a small delay to avoid blocking initial render
    const prefetchTimer = setTimeout(() => {
      criticalRoutes.forEach(route => {
        if (!prefetchedRoutes.current.has(route)) {
          router.prefetch(route);
          prefetchedRoutes.current.add(route);
        }
      });
    }, 100);

    return () => clearTimeout(prefetchTimer);
  }, [router]);

  const safeNavigate = useCallback(
    (url: string, options?: { replace?: boolean; scroll?: boolean }) => {
      if (isNavigatingRef.current) return;

      isNavigatingRef.current = true;

      // Prefetch the route if not already prefetched
      if (!prefetchedRoutes.current.has(url)) {
        router.prefetch(url);
        prefetchedRoutes.current.add(url);
      }

      // Use optimized navigation method
      if (options?.replace) {
        router.replace(url, { scroll: options?.scroll ?? true });
      } else {
        router.push(url, { scroll: options?.scroll ?? true });
      }

      // Reset navigation flag after a brief delay
      setTimeout(() => {
        isNavigatingRef.current = false;
      }, 50);
    },
    [router],
  );

  const safePrefetch = useCallback(
    (url: string, priority: 'high' | 'low' = 'low') => {
      if (!prefetchedRoutes.current.has(url)) {
        router.prefetch(url);
        prefetchedRoutes.current.add(url);
      }
    },
    [router],
  );

  const preloadRoute = useCallback(
    (url: string) => {
      // Immediate prefetch for hover/focus events
      safePrefetch(url, 'high');
    },
    [safePrefetch],
  );

  return {
    navigate: safeNavigate,
    prefetch: safePrefetch,
    preload: preloadRoute,
    push: safeNavigate,
    replace: (url: string) => safeNavigate(url, { replace: true }),
    isNavigating: isNavigatingRef.current,
  };
}

/**
 * SafeNavigationProvider component that wraps the app to provide
 * safe navigation context and cleanup handlers.
 */
export function SafeNavigationProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export default useSafeNavigation;
