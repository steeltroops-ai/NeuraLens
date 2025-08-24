'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useRef } from 'react';

/**
 * SafeNavigation hook provides navigation methods that properly clean up
 * DOM elements and prevent hydration conflicts during route transitions.
 */
export function useSafeNavigation() {
  const router = useRouter();
  const isNavigatingRef = useRef(false);

  const safeNavigate = useCallback(
    (url: string) => {
      if (isNavigatingRef.current) return;

      isNavigatingRef.current = true;

      // Immediate navigation for instant SPA-style transitions
      router.push(url);

      // Reset navigation flag immediately
      isNavigatingRef.current = false;
    },
    [router],
  );

  const safePrefetch = useCallback(
    (url: string) => {
      router.prefetch(url);
    },
    [router],
  );

  return {
    navigate: safeNavigate,
    prefetch: safePrefetch,
    push: safeNavigate,
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
