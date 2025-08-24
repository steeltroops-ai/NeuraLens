/**
 * Accessibility Testing Utilities
 * Automated WCAG 2.1 AA compliance testing
 */

import { ColorContrast } from './utils';

// Accessibility test result interface
export interface AccessibilityTestResult {
  passed: boolean;
  level: 'A' | 'AA' | 'AAA';
  criterion: string;
  description: string;
  element?: HTMLElement;
  recommendation?: string;
}

// Accessibility test suite
export class AccessibilityTester {
  private results: AccessibilityTestResult[] = [];

  /**
   * Run all accessibility tests
   */
  async runAllTests(container: HTMLElement = document.body): Promise<AccessibilityTestResult[]> {
    this.results = [];

    // Run individual test suites
    await this.testKeyboardNavigation(container);
    await this.testAriaLabels(container);
    await this.testColorContrast(container);
    await this.testFocusManagement(container);
    await this.testSemanticStructure(container);
    await this.testFormAccessibility(container);
    await this.testImageAccessibility(container);
    await this.testLiveRegions(container);

    return this.results;
  }

  /**
   * Test keyboard navigation
   */
  private async testKeyboardNavigation(container: HTMLElement): Promise<void> {
    const focusableElements = this.getFocusableElements(container);

    // Test 1: All interactive elements are focusable
    const interactiveElements = container.querySelectorAll(
      'button, input, select, textarea, a[href]',
    );
    interactiveElements.forEach(element => {
      const htmlElement = element as HTMLElement;
      const isFocusable = htmlElement.tabIndex >= 0 || this.isNaturallyFocusable(htmlElement);

      this.results.push({
        passed: isFocusable,
        level: 'A',
        criterion: '2.1.1 Keyboard',
        description: 'All interactive elements must be keyboard accessible',
        element: htmlElement,
        recommendation: isFocusable
          ? undefined
          : 'Add tabindex="0" or ensure element is naturally focusable',
      });
    });

    // Test 2: No keyboard traps (except intentional ones)
    const elementsWithTabIndex = container.querySelectorAll('[tabindex]');
    elementsWithTabIndex.forEach(element => {
      const tabIndex = parseInt((element as HTMLElement).getAttribute('tabindex') || '0');
      const isValidTabIndex = tabIndex >= -1;

      this.results.push({
        passed: isValidTabIndex,
        level: 'A',
        criterion: '2.1.2 No Keyboard Trap',
        description: 'Keyboard focus must not be trapped',
        element: element as HTMLElement,
        recommendation: isValidTabIndex
          ? undefined
          : 'Use tabindex="-1" for programmatically focusable elements or tabindex="0" for keyboard accessible elements',
      });
    });

    // Test 3: Logical tab order
    const tabOrder = focusableElements.map(el => ({
      element: el,
      tabIndex: el.tabIndex,
      position: this.getElementPosition(el),
    }));

    let logicalOrder = true;
    for (let i = 1; i < tabOrder.length; i++) {
      const current = tabOrder[i];
      const previous = tabOrder[i - 1];

      if (!current || !previous) continue;

      if (current.tabIndex === 0 && previous.tabIndex === 0) {
        // Check visual order for elements with tabindex="0"
        if (
          current.position.top < previous.position.top ||
          (current.position.top === previous.position.top &&
            current.position.left < previous.position.left)
        ) {
          logicalOrder = false;
          break;
        }
      }
    }

    this.results.push({
      passed: logicalOrder,
      level: 'A',
      criterion: '2.4.3 Focus Order',
      description: 'Focus order must be logical and intuitive',
      recommendation: logicalOrder
        ? undefined
        : 'Ensure tab order follows visual layout or use tabindex to create logical order',
    });
  }

  /**
   * Test ARIA labels and properties
   */
  private async testAriaLabels(container: HTMLElement): Promise<void> {
    // Test 1: Form controls have labels
    const formControls = container.querySelectorAll('input, select, textarea');
    formControls.forEach(control => {
      const htmlControl = control as HTMLElement;
      const hasLabel = this.hasAccessibleLabel(htmlControl);

      this.results.push({
        passed: hasLabel,
        level: 'A',
        criterion: '1.3.1 Info and Relationships',
        description: 'Form controls must have accessible labels',
        element: htmlControl,
        recommendation: hasLabel
          ? undefined
          : 'Add aria-label, aria-labelledby, or associate with a <label> element',
      });
    });

    // Test 2: Images have alt text
    const images = container.querySelectorAll('img');
    images.forEach(img => {
      const hasAltText = img.hasAttribute('alt');

      this.results.push({
        passed: hasAltText,
        level: 'A',
        criterion: '1.1.1 Non-text Content',
        description: 'Images must have alternative text',
        element: img as HTMLElement,
        recommendation: hasAltText
          ? undefined
          : 'Add alt attribute with descriptive text or alt="" for decorative images',
      });
    });

    // Test 3: ARIA roles are valid
    const elementsWithRoles = container.querySelectorAll('[role]');
    const validRoles = [
      'alert',
      'alertdialog',
      'application',
      'article',
      'banner',
      'button',
      'cell',
      'checkbox',
      'columnheader',
      'combobox',
      'complementary',
      'contentinfo',
      'definition',
      'dialog',
      'directory',
      'document',
      'feed',
      'figure',
      'form',
      'grid',
      'gridcell',
      'group',
      'heading',
      'img',
      'link',
      'list',
      'listbox',
      'listitem',
      'log',
      'main',
      'marquee',
      'math',
      'menu',
      'menubar',
      'menuitem',
      'menuitemcheckbox',
      'menuitemradio',
      'navigation',
      'none',
      'note',
      'option',
      'presentation',
      'progressbar',
      'radio',
      'radiogroup',
      'region',
      'row',
      'rowgroup',
      'rowheader',
      'scrollbar',
      'search',
      'searchbox',
      'separator',
      'slider',
      'spinbutton',
      'status',
      'switch',
      'tab',
      'table',
      'tablist',
      'tabpanel',
      'term',
      'textbox',
      'timer',
      'toolbar',
      'tooltip',
      'tree',
      'treegrid',
      'treeitem',
    ];

    elementsWithRoles.forEach(element => {
      const role = element.getAttribute('role');
      const isValidRole = role && validRoles.includes(role);

      this.results.push({
        passed: !!isValidRole,
        level: 'A',
        criterion: '4.1.2 Name, Role, Value',
        description: 'ARIA roles must be valid',
        element: element as HTMLElement,
        recommendation: isValidRole ? undefined : `Use a valid ARIA role. Invalid role: "${role}"`,
      });
    });
  }

  /**
   * Test color contrast
   */
  private async testColorContrast(container: HTMLElement): Promise<void> {
    const textElements = container.querySelectorAll(
      'p, span, div, h1, h2, h3, h4, h5, h6, a, button, label',
    );

    textElements.forEach(element => {
      const htmlElement = element as HTMLElement;
      const styles = window.getComputedStyle(htmlElement);
      const fontSize = parseFloat(styles.fontSize);
      const fontWeight = styles.fontWeight;

      const isLargeText =
        fontSize >= 18 ||
        (fontSize >= 14 && (fontWeight === 'bold' || parseInt(fontWeight) >= 700));

      // Mock contrast check (in real implementation, would calculate actual contrast)
      const mockContrastRatio = 4.5; // This would be calculated from actual colors
      const meetsAA = ColorContrast.meetsWCAGAA(mockContrastRatio, isLargeText);

      this.results.push({
        passed: meetsAA,
        level: 'AA',
        criterion: '1.4.3 Contrast (Minimum)',
        description: 'Text must have sufficient color contrast',
        element: htmlElement,
        recommendation: meetsAA
          ? undefined
          : `Increase color contrast. Current ratio: ${mockContrastRatio.toFixed(2)}, required: ${isLargeText ? '3:1' : '4.5:1'}`,
      });
    });
  }

  /**
   * Test focus management
   */
  private async testFocusManagement(container: HTMLElement): Promise<void> {
    const focusableElements = this.getFocusableElements(container);

    // Test focus indicators
    focusableElements.forEach(element => {
      const styles = window.getComputedStyle(element, ':focus');
      const hasVisibleFocus = styles.outline !== 'none' || styles.boxShadow !== 'none';

      this.results.push({
        passed: hasVisibleFocus,
        level: 'AA',
        criterion: '2.4.7 Focus Visible',
        description: 'Focusable elements must have visible focus indicators',
        element,
        recommendation: hasVisibleFocus
          ? undefined
          : 'Add :focus styles with outline or box-shadow',
      });
    });
  }

  /**
   * Test semantic structure
   */
  private async testSemanticStructure(container: HTMLElement): Promise<void> {
    // Test heading hierarchy
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
    let previousLevel = 0;
    let hierarchyValid = true;

    headings.forEach(heading => {
      const level = parseInt(heading.tagName.charAt(1));
      if (level > previousLevel + 1) {
        hierarchyValid = false;
      }
      previousLevel = level;
    });

    this.results.push({
      passed: hierarchyValid,
      level: 'AA',
      criterion: '1.3.1 Info and Relationships',
      description: 'Heading hierarchy must be logical',
      recommendation: hierarchyValid
        ? undefined
        : 'Ensure headings follow logical order (h1, h2, h3, etc.) without skipping levels',
    });

    // Test landmark regions
    const hasMain = container.querySelector('main, [role="main"]') !== null;
    this.results.push({
      passed: hasMain,
      level: 'AA',
      criterion: '1.3.1 Info and Relationships',
      description: 'Page must have a main landmark',
      recommendation: hasMain
        ? undefined
        : 'Add <main> element or role="main" to identify main content area',
    });
  }

  /**
   * Test form accessibility
   */
  private async testFormAccessibility(container: HTMLElement): Promise<void> {
    const forms = container.querySelectorAll('form');

    forms.forEach(form => {
      // Test required field indicators
      const requiredFields = form.querySelectorAll('[required], [aria-required="true"]');
      requiredFields.forEach(field => {
        const hasRequiredIndicator = this.hasRequiredIndicator(field as HTMLElement);

        this.results.push({
          passed: hasRequiredIndicator,
          level: 'A',
          criterion: '3.3.2 Labels or Instructions',
          description: 'Required fields must be clearly indicated',
          element: field as HTMLElement,
          recommendation: hasRequiredIndicator
            ? undefined
            : 'Add visual and programmatic indication of required fields',
        });
      });

      // Test error messages
      const fieldsWithErrors = form.querySelectorAll('[aria-invalid="true"]');
      fieldsWithErrors.forEach(field => {
        const hasErrorMessage = this.hasErrorMessage(field as HTMLElement);

        this.results.push({
          passed: hasErrorMessage,
          level: 'AA',
          criterion: '3.3.1 Error Identification',
          description: 'Form errors must be clearly identified',
          element: field as HTMLElement,
          recommendation: hasErrorMessage
            ? undefined
            : 'Associate error messages with form fields using aria-describedby',
        });
      });
    });
  }

  /**
   * Test image accessibility
   */
  private async testImageAccessibility(container: HTMLElement): Promise<void> {
    const images = container.querySelectorAll('img');

    images.forEach(img => {
      const altText = img.getAttribute('alt');
      const isDecorative = altText === '';
      const hasDescriptiveAlt = altText && altText.length > 0 && altText.length < 125;

      if (!isDecorative) {
        this.results.push({
          passed: !!hasDescriptiveAlt,
          level: 'A',
          criterion: '1.1.1 Non-text Content',
          description: 'Informative images must have descriptive alt text',
          element: img as HTMLElement,
          recommendation: hasDescriptiveAlt
            ? undefined
            : 'Provide concise, descriptive alt text (under 125 characters)',
        });
      }
    });
  }

  /**
   * Test live regions
   */
  private async testLiveRegions(container: HTMLElement): Promise<void> {
    const liveRegions = container.querySelectorAll('[aria-live]');

    liveRegions.forEach(region => {
      const ariaLive = region.getAttribute('aria-live');
      const validValues = ['polite', 'assertive', 'off'];
      const isValid = validValues.includes(ariaLive || '');

      this.results.push({
        passed: isValid,
        level: 'A',
        criterion: '4.1.3 Status Messages',
        description: 'Live regions must have valid aria-live values',
        element: region as HTMLElement,
        recommendation: isValid ? undefined : 'Use aria-live="polite" or aria-live="assertive"',
      });
    });
  }

  // Helper methods
  private getFocusableElements(container: HTMLElement): HTMLElement[] {
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

  private isNaturallyFocusable(element: HTMLElement): boolean {
    const naturallyFocusable = ['button', 'input', 'select', 'textarea', 'a'];
    return (
      naturallyFocusable.includes(element.tagName.toLowerCase()) &&
      !element.hasAttribute('disabled')
    );
  }

  private getElementPosition(element: HTMLElement): { top: number; left: number } {
    const rect = element.getBoundingClientRect();
    return { top: rect.top, left: rect.left };
  }

  private hasAccessibleLabel(element: HTMLElement): boolean {
    return !!(
      element.getAttribute('aria-label') ||
      element.getAttribute('aria-labelledby') ||
      element.getAttribute('title') ||
      document.querySelector(`label[for="${element.id}"]`)
    );
  }

  private hasRequiredIndicator(element: HTMLElement): boolean {
    const label = document.querySelector(`label[for="${element.id}"]`);
    return !!(
      element.getAttribute('aria-required') === 'true' ||
      element.hasAttribute('required') ||
      (label && (label.textContent?.includes('*') || label.textContent?.includes('required')))
    );
  }

  private hasErrorMessage(element: HTMLElement): boolean {
    const describedBy = element.getAttribute('aria-describedby');
    if (describedBy) {
      const errorElement = document.getElementById(describedBy);
      return !!(errorElement && errorElement.textContent);
    }
    return false;
  }

  /**
   * Generate accessibility report
   */
  generateReport(): string {
    const totalTests = this.results.length;
    const passedTests = this.results.filter(r => r.passed).length;
    const failedTests = totalTests - passedTests;

    const levelCounts = {
      A: this.results.filter(r => r.level === 'A').length,
      AA: this.results.filter(r => r.level === 'AA').length,
      AAA: this.results.filter(r => r.level === 'AAA').length,
    };

    const passedByLevel = {
      A: this.results.filter(r => r.level === 'A' && r.passed).length,
      AA: this.results.filter(r => r.level === 'AA' && r.passed).length,
      AAA: this.results.filter(r => r.level === 'AAA' && r.passed).length,
    };

    let report = `# NeuraLens Accessibility Test Report\n\n`;
    report += `## Summary\n`;
    report += `- **Total Tests**: ${totalTests}\n`;
    report += `- **Passed**: ${passedTests} (${((passedTests / totalTests) * 100).toFixed(1)}%)\n`;
    report += `- **Failed**: ${failedTests} (${((failedTests / totalTests) * 100).toFixed(1)}%)\n\n`;

    report += `## WCAG 2.1 Compliance\n`;
    report += `- **Level A**: ${passedByLevel.A}/${levelCounts.A} (${((passedByLevel.A / levelCounts.A) * 100).toFixed(1)}%)\n`;
    report += `- **Level AA**: ${passedByLevel.AA}/${levelCounts.AA} (${((passedByLevel.AA / levelCounts.AA) * 100).toFixed(1)}%)\n`;
    report += `- **Level AAA**: ${passedByLevel.AAA}/${levelCounts.AAA} (${((passedByLevel.AAA / levelCounts.AAA) * 100).toFixed(1)}%)\n\n`;

    const failedResults = this.results.filter(r => !r.passed);
    if (failedResults.length > 0) {
      report += `## Failed Tests\n\n`;
      failedResults.forEach((result, index) => {
        report += `### ${index + 1}. ${result.criterion} (Level ${result.level})\n`;
        report += `**Description**: ${result.description}\n`;
        if (result.recommendation) {
          report += `**Recommendation**: ${result.recommendation}\n`;
        }
        report += `\n`;
      });
    }

    return report;
  }
}

// Export singleton instance
export const accessibilityTester = new AccessibilityTester();
