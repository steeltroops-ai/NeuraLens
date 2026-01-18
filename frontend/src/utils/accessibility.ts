// NeuroLens-X Accessibility Utilities

/* ===== ACCESSIBILITY HELPERS ===== */

/**
 * Announces text to screen readers using aria-live regions
 */
export const announceToScreenReader = (
  message: string,
  priority: 'polite' | 'assertive' = 'polite',
  delay: number = 100,
): void => {
  // Check if we're in a browser environment
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return;
  }

  // Create temporary announcement element
  const announcement = document.createElement('div');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;

  // Add to DOM
  document.body.appendChild(announcement);

  // Remove after announcement
  setTimeout(() => {
    if (document.body.contains(announcement)) {
      document.body.removeChild(announcement);
    }
  }, delay + 1000);
};

/**
 * Manages focus for single-page application navigation
 */
export class FocusManager {
  private focusHistory: HTMLElement[] = [];

  /**
   * Sets focus to an element and adds it to history
   */
  setFocus(element: HTMLElement | string, options?: FocusOptions): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    const targetElement =
      typeof element === 'string' ? (document.querySelector(element) as HTMLElement) : element;

    if (targetElement) {
      // Store current focus in history
      const currentFocus = document.activeElement as HTMLElement;
      if (currentFocus && currentFocus !== targetElement) {
        this.focusHistory.push(currentFocus);
      }

      // Set focus with options
      targetElement.focus(options);

      // Announce focus change to screen readers
      const label =
        targetElement.getAttribute('aria-label') ||
        targetElement.getAttribute('title') ||
        targetElement.textContent?.trim() ||
        'Element focused';
      announceToScreenReader(`Focused on ${label}`, 'polite');
    }
  }

  /**
   * Returns focus to the previous element
   */
  returnFocus(): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    const previousElement = this.focusHistory.pop();
    if (previousElement && document.body.contains(previousElement)) {
      previousElement.focus();
    }
  }

  /**
   * Traps focus within a container (for modals, dialogs)
   */
  trapFocus(container: HTMLElement): () => void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return () => { }; // Return empty cleanup function
    }

    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    ) as NodeListOf<HTMLElement>;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleTabKey = (event: KeyboardEvent) => {
      if (event.key === 'Tab' && firstElement && lastElement) {
        if (event.shiftKey) {
          // Shift + Tab
          if (document.activeElement === firstElement) {
            event.preventDefault();
            lastElement.focus();
          }
        } else {
          // Tab
          if (document.activeElement === lastElement) {
            event.preventDefault();
            firstElement.focus();
          }
        }
      }

      // Escape key to close
      if (event.key === 'Escape') {
        this.returnFocus();
      }
    };

    container.addEventListener('keydown', handleTabKey);

    // Set initial focus
    if (firstElement) {
      firstElement.focus();
    }

    // Return cleanup function
    return () => {
      container.removeEventListener('keydown', handleTabKey);
    };
  }
}

/**
 * Keyboard navigation utilities
 */
export class KeyboardNavigation {
  private shortcuts: Map<string, () => void> = new Map();

  /**
   * Registers a keyboard shortcut
   */
  registerShortcut(key: string, callback: () => void, description?: string): void {
    this.shortcuts.set(key.toLowerCase(), callback);

    // Add to help documentation if description provided
    if (description) {
      this.addToShortcutHelp(key, description);
    }
  }

  /**
   * Handles keyboard events for registered shortcuts
   */
  handleKeyDown = (event: KeyboardEvent): void => {
    const key = event.key.toLowerCase();
    const modifiers = {
      ctrl: event.ctrlKey,
      alt: event.altKey,
      shift: event.shiftKey,
      meta: event.metaKey,
    };

    // Build key combination string
    let keyCombo = '';
    if (modifiers.ctrl) keyCombo += 'ctrl+';
    if (modifiers.alt) keyCombo += 'alt+';
    if (modifiers.shift) keyCombo += 'shift+';
    if (modifiers.meta) keyCombo += 'meta+';
    keyCombo += key;

    const callback = this.shortcuts.get(keyCombo);
    if (callback) {
      event.preventDefault();
      callback();
    }
  };

  /**
   * Initializes keyboard navigation
   */
  initialize(): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    document.addEventListener('keydown', this.handleKeyDown);

    // Register default shortcuts
    this.registerShortcut(
      'alt+1',
      () => {
        const mainContent = document.getElementById('main-content');
        if (mainContent) focusManager.setFocus(mainContent);
      },
      'Skip to main content',
    );

    this.registerShortcut(
      'alt+2',
      () => {
        const navigation = document.getElementById('main-navigation');
        if (navigation) focusManager.setFocus(navigation);
      },
      'Skip to navigation',
    );

    this.registerShortcut(
      'alt+h',
      () => {
        if (typeof window !== 'undefined' && window.history) {
          window.history.pushState(null, '', '/');
          window.dispatchEvent(new PopStateEvent('popstate'));
        }
      },
      'Go to home page',
    );


  }

  /**
   * Cleans up keyboard navigation
   */
  destroy(): void {
    document.removeEventListener('keydown', this.handleKeyDown);
    this.shortcuts.clear();
  }

  private addToShortcutHelp(key: string, description: string): void {
    // This would integrate with a help system to document shortcuts
    if (process.env.NODE_ENV === 'development') {
      console.debug(`Keyboard shortcut registered: ${key} - ${description}`);
    }
  }
}

/**
 * Voice navigation utilities (for future implementation)
 */
export class VoiceNavigation {
  private recognition: any = null;
  private commands: Map<string, () => void> = new Map();

  /**
   * Initializes voice recognition if available
   */
  initialize(): boolean {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') {
      return false;
    }

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-US';

      this.recognition.onresult = (event: any) => {
        const command = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
        this.executeCommand(command);
      };

      // Register default voice commands
      this.registerCommand('go home', () => {
        if (typeof window !== 'undefined') {
          window.location.href = '/';
        }
      });

      this.registerCommand('go to dashboard', () => {
        if (typeof window !== 'undefined') {
          window.location.href = '/dashboard';
        }
      });

      return true;
    }

    return false;
  }

  /**
   * Registers a voice command
   */
  registerCommand(phrase: string, callback: () => void): void {
    this.commands.set(phrase.toLowerCase(), callback);
  }

  /**
   * Executes a voice command
   */
  private executeCommand(command: string): void {
    const callback = this.commands.get(command);
    if (callback) {
      announceToScreenReader(`Executing command: ${command}`, 'assertive');
      callback();
    } else {
      announceToScreenReader('Command not recognized', 'polite');
    }
  }

  /**
   * Starts voice recognition
   */
  start(): void {
    if (this.recognition) {
      this.recognition.start();
      announceToScreenReader('Voice navigation activated', 'polite');
    }
  }

  /**
   * Stops voice recognition
   */
  stop(): void {
    if (this.recognition) {
      this.recognition.stop();
      announceToScreenReader('Voice navigation deactivated', 'polite');
    }
  }
}

/**
 * Accessibility preferences manager
 */
export class AccessibilityPreferences {
  private preferences: Record<string, any> = {};

  constructor() {
    this.loadPreferences();
    this.applyPreferences();
  }

  /**
   * Sets an accessibility preference
   */
  setPreference(key: string, value: any): void {
    this.preferences[key] = value;
    this.savePreferences();
    this.applyPreferences();
  }

  /**
   * Gets an accessibility preference
   */
  getPreference(key: string, defaultValue?: any): any {
    return this.preferences[key] ?? defaultValue;
  }

  /**
   * Loads preferences from localStorage
   */
  private loadPreferences(): void {
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return;
      }

      const stored = localStorage.getItem('neurolens-accessibility-preferences');
      if (stored) {
        this.preferences = JSON.parse(stored);
      }
    } catch (error) {
      console.warn('Failed to load accessibility preferences:', error);
    }
  }

  /**
   * Saves preferences to localStorage
   */
  private savePreferences(): void {
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return;
      }

      localStorage.setItem('neurolens-accessibility-preferences', JSON.stringify(this.preferences));
    } catch (error) {
      console.warn('Failed to save accessibility preferences:', error);
    }
  }

  /**
   * Applies preferences to the document
   */
  private applyPreferences(): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    const root = document.documentElement;

    // High contrast mode
    if (this.preferences.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Large text
    if (this.preferences.largeText) {
      root.classList.add('large-text');
    } else {
      root.classList.remove('large-text');
    }

    // Reduced motion
    if (this.preferences.reducedMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }

    // Focus indicators
    if (this.preferences.enhancedFocus) {
      root.classList.add('enhanced-focus');
    } else {
      root.classList.remove('enhanced-focus');
    }
  }
}

/**
 * ARIA live region manager
 */
export class LiveRegionManager {
  private regions: Map<string, HTMLElement> = new Map();

  /**
   * Creates a live region
   */
  createRegion(id: string, politeness: 'polite' | 'assertive' = 'polite'): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    if (this.regions.has(id)) {
      return;
    }

    const region = document.createElement('div');
    region.id = `live-region-${id}`;
    region.setAttribute('aria-live', politeness);
    region.setAttribute('aria-atomic', 'true');
    region.className = 'sr-only';

    document.body.appendChild(region);
    this.regions.set(id, region);
  }

  /**
   * Announces a message in a specific live region
   */
  announce(regionId: string, message: string): void {
    const region = this.regions.get(regionId);
    if (region) {
      region.textContent = message;
    }
  }

  /**
   * Clears a live region
   */
  clear(regionId: string): void {
    const region = this.regions.get(regionId);
    if (region) {
      region.textContent = '';
    }
  }

  /**
   * Removes a live region
   */
  removeRegion(regionId: string): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    const region = this.regions.get(regionId);
    if (region && document.body.contains(region)) {
      document.body.removeChild(region);
      this.regions.delete(regionId);
    }
  }

  clearAll(): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    this.regions.forEach((_, id) => {
      this.removeRegion(id);
    });
  }
}

// Global instances - lazy loaded to avoid SSR issues
let _focusManager: FocusManager | null = null;
let _keyboardNavigation: KeyboardNavigation | null = null;
let _voiceNavigation: VoiceNavigation | null = null;
let _accessibilityPreferences: AccessibilityPreferences | null = null;
let _liveRegionManager: LiveRegionManager | null = null;

export const focusManager = (() => {
  if (typeof window === 'undefined') return {} as FocusManager;
  if (!_focusManager) _focusManager = new FocusManager();
  return _focusManager;
})();

export const keyboardNavigation = (() => {
  if (typeof window === 'undefined') return {} as KeyboardNavigation;
  if (!_keyboardNavigation) _keyboardNavigation = new KeyboardNavigation();
  return _keyboardNavigation;
})();

export const voiceNavigation = (() => {
  if (typeof window === 'undefined') return {} as VoiceNavigation;
  if (!_voiceNavigation) _voiceNavigation = new VoiceNavigation();
  return _voiceNavigation;
})();

export const accessibilityPreferences = (() => {
  if (typeof window === 'undefined') return {} as AccessibilityPreferences;
  if (!_accessibilityPreferences) _accessibilityPreferences = new AccessibilityPreferences();
  return _accessibilityPreferences;
})();

export const liveRegionManager = (() => {
  if (typeof window === 'undefined') return {} as LiveRegionManager;
  if (!_liveRegionManager) _liveRegionManager = new LiveRegionManager();
  return _liveRegionManager;
})();

/**
 * Initializes all accessibility features
 */
export const initializeAccessibility = (): void => {
  // Check if we're in a browser environment
  if (typeof window === 'undefined') {
    return;
  }

  // Initialize keyboard navigation
  keyboardNavigation.initialize?.();

  // Create default live regions
  liveRegionManager.createRegion?.('announcements', 'polite');
  liveRegionManager.createRegion?.('alerts', 'assertive');

  // Initialize voice navigation if supported
  if (voiceNavigation.initialize?.()) {
    if (process.env.NODE_ENV === 'development') {
      console.debug('Voice navigation available');
    }
  }
};

/**
 * Cleans up accessibility features
 */
export const cleanupAccessibility = (): void => {
  // Check if we're in a browser environment
  if (typeof window === 'undefined') {
    return;
  }

  keyboardNavigation.destroy?.();
  voiceNavigation.stop?.();

  // Clear all live regions
  liveRegionManager.clearAll?.();
};
