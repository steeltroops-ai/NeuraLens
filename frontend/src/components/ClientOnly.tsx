'use client';

import { useState, useEffect, ReactNode, useRef } from 'react';
import { usePathname } from 'next/navigation';

interface ClientOnlyProps {
  children: ReactNode;
  fallback?: ReactNode;
  resetOnNavigation?: boolean;
}

/**
 * Enhanced ClientOnly component prevents hydration mismatches and navigation conflicts
 * by only rendering children on the client side after hydration is complete.
 * Includes navigation-aware cleanup for canvas/WebGL components.
 */
export function ClientOnly({
  children,
  fallback = null,
  resetOnNavigation = false,
}: ClientOnlyProps) {
  const [hasMounted, setHasMounted] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const pathname = usePathname();
  const previousPathname = useRef(pathname);
  const mountTimeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    // Clear any existing timeout
    if (mountTimeoutRef.current) {
      clearTimeout(mountTimeoutRef.current);
    }

    // Delay mounting to ensure DOM is stable
    mountTimeoutRef.current = setTimeout(() => {
      setHasMounted(true);
      // Additional delay for complex components
      setTimeout(() => setIsReady(true), 100);
    }, 50);

    return () => {
      if (mountTimeoutRef.current) {
        clearTimeout(mountTimeoutRef.current);
      }
    };
  }, []);

  // Handle navigation changes
  useEffect(() => {
    if (resetOnNavigation && previousPathname.current !== pathname) {
      setIsReady(false);
      setHasMounted(false);

      // Re-mount after navigation
      const timeout = setTimeout(() => {
        setHasMounted(true);
        setTimeout(() => setIsReady(true), 100);
      }, 100);

      previousPathname.current = pathname;

      return () => clearTimeout(timeout);
    }

    return undefined;
  }, [pathname, resetOnNavigation]);

  if (!hasMounted || !isReady) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}

export default ClientOnly;
