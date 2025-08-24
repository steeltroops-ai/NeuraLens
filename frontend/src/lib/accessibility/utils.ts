/**
 * Accessibility Utilities
 * WCAG 2.1 AA compliance utilities and helpers
 */

// Focus management utilities
export class FocusManager {
  private static focusStack: HTMLElement[] = [];
  private static trapStack: HTMLElement[] = [];

  /**
   * Save current focus and set new focus
   */
  static saveFocus(newFocus?: HTMLElement): void {
    const currentFocus = document.activeElement as HTMLElement;
    if (currentFocus) {
      this.focusStack.push(currentFocus);
    }

    if (newFocus) {
      newFocus.focus();
    }
  }

  /**
   * Restore previously saved focus
   */
  static restoreFocus(): void {
    const previousFocus = this.focusStack.pop();
    if (previousFocus && document.contains(previousFocus)) {
      previousFocus.focus();
    }
  }

  /**
   * Trap focus within an element
   */
  static trapFocus(element: HTMLElement): void {
    this.trapStack.push(element);

    const focusableElements = this.getFocusableElements(element);
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (!firstElement || !lastElement) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    element.addEventListener('keydown', handleKeyDown);
    firstElement.focus();

    // Store cleanup function
    (element as any)._focusTrapCleanup = () => {
      element.removeEventListener('keydown', handleKeyDown);
    };
  }

  /**
   * Release focus trap
   */
  static releaseFocusTrap(): void {
    const element = this.trapStack.pop();
    if (element && (element as any)._focusTrapCleanup) {
      (element as any)._focusTrapCleanup();
      delete (element as any)._focusTrapCleanup;
    }
  }

  /**
   * Get all focusable elements within a container
   */
  static getFocusableElements(container: HTMLElement): HTMLElement[] {
    const focusableSelectors = [
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'a[href]',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]',
    ].join(', ');

    return Array.from(container.querySelectorAll(focusableSelectors)).filter(el => {
      const element = el as HTMLElement;
      return element.offsetWidth > 0 && element.offsetHeight > 0;
    }) as HTMLElement[];
  }
}

// Color contrast utilities
export class ColorContrast {
  /**
   * Calculate relative luminance of a color
   */
  static getLuminance(r: number, g: number, b: number): number {
    const values = [r, g, b].map(c => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });

    const [rs = 0, gs = 0, bs = 0] = values;
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
  }

  /**
   * Calculate contrast ratio between two colors
   */
  static getContrastRatio(
    color1: [number, number, number],
    color2: [number, number, number],
  ): number {
    const lum1 = this.getLuminance(...color1);
    const lum2 = this.getLuminance(...color2);

    const brightest = Math.max(lum1, lum2);
    const darkest = Math.min(lum1, lum2);

    return (brightest + 0.05) / (darkest + 0.05);
  }

  /**
   * Check if contrast ratio meets WCAG AA standards
   */
  static meetsWCAGAA(contrastRatio: number, isLargeText: boolean = false): boolean {
    return contrastRatio >= (isLargeText ? 3 : 4.5);
  }

  /**
   * Check if contrast ratio meets WCAG AAA standards
   */
  static meetsWCAGAAA(contrastRatio: number, isLargeText: boolean = false): boolean {
    return contrastRatio >= (isLargeText ? 4.5 : 7);
  }

  /**
   * Parse hex color to RGB
   */
  static hexToRgb(hex: string): [number, number, number] | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result && result[1] && result[2] && result[3]
      ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
      : null;
  }
}

// Screen reader utilities
export class ScreenReader {
  private static announceElement: HTMLElement | null = null;

  /**
   * Initialize screen reader utilities
   */
  static init(): void {
    if (this.announceElement) return;

    this.announceElement = document.createElement('div');
    this.announceElement.setAttribute('aria-live', 'polite');
    this.announceElement.setAttribute('aria-atomic', 'true');
    this.announceElement.className = 'sr-only';
    this.announceElement.style.cssText = `
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    `;

    document.body.appendChild(this.announceElement);
  }

  /**
   * Announce message to screen readers
   */
  static announce(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    this.init();

    if (this.announceElement) {
      this.announceElement.setAttribute('aria-live', priority);
      this.announceElement.textContent = message;

      // Clear after announcement
      setTimeout(() => {
        if (this.announceElement) {
          this.announceElement.textContent = '';
        }
      }, 1000);
    }
  }

  /**
   * Announce assessment progress
   */
  static announceProgress(step: string, progress: number, total: number): void {
    const message = `Assessment progress: ${step}, step ${progress} of ${total}`;
    this.announce(message, 'polite');
  }

  /**
   * Announce assessment results
   */
  static announceResults(riskCategory: string, nriScore: number): void {
    const message = `Assessment complete. Risk category: ${riskCategory}. NRI score: ${Math.round(nriScore * 100)} percent.`;
    this.announce(message, 'polite');
  }

  /**
   * Announce error
   */
  static announceError(error: string): void {
    const message = `Error: ${error}`;
    this.announce(message, 'assertive');
  }
}

// Keyboard navigation utilities
export class KeyboardNavigation {
  /**
   * Handle arrow key navigation in a list
   */
  static handleArrowNavigation(
    event: KeyboardEvent,
    items: HTMLElement[],
    currentIndex: number,
    onIndexChange: (index: number) => void,
  ): void {
    let newIndex = currentIndex;

    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        event.preventDefault();
        newIndex = (currentIndex + 1) % items.length;
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        event.preventDefault();
        newIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1;
        break;
      case 'Home':
        event.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        event.preventDefault();
        newIndex = items.length - 1;
        break;
      default:
        return;
    }

    onIndexChange(newIndex);
    items[newIndex]?.focus();
  }

  /**
   * Handle escape key to close modals/dropdowns
   */
  static handleEscape(event: KeyboardEvent, onEscape: () => void): void {
    if (event.key === 'Escape') {
      event.preventDefault();
      onEscape();
    }
  }

  /**
   * Handle enter/space activation
   */
  static handleActivation(event: KeyboardEvent, onActivate: () => void): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onActivate();
    }
  }
}

// ARIA utilities
export class AriaUtils {
  /**
   * Generate unique ID for ARIA relationships
   */
  static generateId(prefix: string = 'aria'): string {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Set up ARIA describedby relationship
   */
  static setDescribedBy(element: HTMLElement, descriptionId: string): void {
    const existingIds = element.getAttribute('aria-describedby') || '';
    const ids = existingIds.split(' ').filter(id => id.length > 0);

    if (!ids.includes(descriptionId)) {
      ids.push(descriptionId);
      element.setAttribute('aria-describedby', ids.join(' '));
    }
  }

  /**
   * Remove ARIA describedby relationship
   */
  static removeDescribedBy(element: HTMLElement, descriptionId: string): void {
    const existingIds = element.getAttribute('aria-describedby') || '';
    const ids = existingIds.split(' ').filter(id => id !== descriptionId);

    if (ids.length > 0) {
      element.setAttribute('aria-describedby', ids.join(' '));
    } else {
      element.removeAttribute('aria-describedby');
    }
  }

  /**
   * Set up ARIA labelledby relationship
   */
  static setLabelledBy(element: HTMLElement, labelId: string): void {
    element.setAttribute('aria-labelledby', labelId);
  }

  /**
   * Update ARIA live region
   */
  static updateLiveRegion(
    element: HTMLElement,
    message: string,
    priority: 'polite' | 'assertive' = 'polite',
  ): void {
    element.setAttribute('aria-live', priority);
    element.textContent = message;
  }
}

// Motion and animation utilities
export class MotionUtils {
  /**
   * Check if user prefers reduced motion
   */
  static prefersReducedMotion(): boolean {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /**
   * Get safe animation duration (respects reduced motion preference)
   */
  static getSafeAnimationDuration(normalDuration: number): number {
    return this.prefersReducedMotion() ? 0 : normalDuration;
  }

  /**
   * Apply safe animation styles
   */
  static applySafeAnimation(element: HTMLElement, animation: string): void {
    if (!this.prefersReducedMotion()) {
      element.style.animation = animation;
    }
  }
}

// Initialize screen reader utilities on module load
if (typeof window !== 'undefined') {
  ScreenReader.init();
}
