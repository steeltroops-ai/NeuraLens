import { useEffect, useRef } from 'react';

/**
 * Custom hook to handle clicks outside of a component
 * Uses passive event listeners to improve performance
 */
export function useOnClickOutside<T extends HTMLElement = HTMLElement>(
  handler: (event: Event) => void
) {
  const ref = useRef<T>(null);

  useEffect(() => {
    const listener = (event: Event) => {
      const element = ref.current;
      if (!element || element.contains(event.target as Node)) {
        return;
      }
      handler(event);
    };

    // Use passive event listeners for better performance
    const options: AddEventListenerOptions = {
      passive: true,
      capture: true,
    };

    document.addEventListener('mousedown', listener, options);
    document.addEventListener('touchstart', listener, options);

    return () => {
      document.removeEventListener('mousedown', listener, options);
      document.removeEventListener('touchstart', listener, options);
    };
  }, [handler]);

  return ref;
}
