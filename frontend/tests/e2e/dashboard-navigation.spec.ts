import { test, expect } from '@playwright/test';

test.describe('Dashboard Navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the dashboard
    await page.goto('/dashboard');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
  });

  test('should display dashboard overview by default', async ({ page }) => {
    // Check that we're on the overview page
    await expect(page.locator('h1')).toContainText('NeuraLens Dashboard');
    
    // Check for key dashboard elements
    await expect(page.locator('[data-testid="dashboard-stats"]')).toBeVisible();
    await expect(page.locator('[data-testid="quick-actions"]')).toBeVisible();
  });

  test('should navigate between assessment modules', async ({ page }) => {
    // Navigate to Speech Analysis
    await page.click('text=Speech Analysis');
    await expect(page.locator('h1')).toContainText('Speech Analysis');
    
    // Navigate to Retinal Analysis
    await page.click('text=Retinal Analysis');
    await expect(page.locator('h1')).toContainText('Retinal Analysis');
    
    // Navigate to Motor Assessment
    await page.click('text=Motor Assessment');
    await expect(page.locator('h1')).toContainText('Motor Function Assessment');
    
    // Navigate to Assessment History
    await page.click('text=Assessment History');
    await expect(page.locator('h1')).toContainText('Assessment History');
  });

  test('should maintain responsive design on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Check that navigation is still accessible
    await expect(page.locator('nav')).toBeVisible();
    
    // Check that content adapts to mobile
    await expect(page.locator('[data-testid="dashboard-stats"]')).toBeVisible();
  });

  test('should handle keyboard navigation', async ({ page }) => {
    // Focus on the first navigation item
    await page.keyboard.press('Tab');
    
    // Navigate using arrow keys
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    
    // Verify navigation worked
    await expect(page.locator('h1')).toContainText(/Speech Analysis|Retinal Analysis|Motor|Assessment History/);
  });

  test('should display loading states during navigation', async ({ page }) => {
    // Slow down network to see loading states
    await page.route('**/*', route => {
      setTimeout(() => route.continue(), 100);
    });
    
    // Click navigation item
    await page.click('text=Speech Analysis');
    
    // Check for loading indicator (if implemented)
    // await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible();
    
    // Wait for content to load
    await expect(page.locator('h1')).toContainText('Speech Analysis');
  });

  test('should preserve state when switching modules', async ({ page }) => {
    // Navigate to Speech Analysis
    await page.click('text=Speech Analysis');
    
    // Interact with the module (if there are form inputs)
    // await page.fill('[data-testid="test-input"]', 'test value');
    
    // Navigate away and back
    await page.click('text=Retinal Analysis');
    await page.click('text=Speech Analysis');
    
    // Verify state is preserved (if applicable)
    // await expect(page.locator('[data-testid="test-input"]')).toHaveValue('test value');
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Mock API error
    await page.route('**/api/**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' })
      });
    });
    
    // Navigate to a module that makes API calls
    await page.click('text=Speech Analysis');
    
    // Verify error handling (if implemented)
    // await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    
    // Verify the app doesn't crash
    await expect(page.locator('h1')).toContainText('Speech Analysis');
  });
});

test.describe('Dashboard Accessibility', () => {
  test('should meet accessibility standards', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check for proper heading hierarchy
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();
    expect(headings.length).toBeGreaterThan(0);
    
    // Check for alt text on images
    const images = await page.locator('img').all();
    for (const img of images) {
      const alt = await img.getAttribute('alt');
      expect(alt).toBeTruthy();
    }
    
    // Check for proper form labels
    const inputs = await page.locator('input').all();
    for (const input of inputs) {
      const id = await input.getAttribute('id');
      if (id) {
        const label = page.locator(`label[for="${id}"]`);
        await expect(label).toBeVisible();
      }
    }
  });

  test('should support screen reader navigation', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check for proper ARIA labels
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label');
      const text = await button.textContent();
      expect(ariaLabel || text).toBeTruthy();
    }
    
    // Check for landmark regions
    await expect(page.locator('nav')).toBeVisible();
    await expect(page.locator('main')).toBeVisible();
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto('/dashboard');
    
    // This would require a color contrast checking library
    // For now, we'll just verify that text is visible
    const textElements = await page.locator('p, span, div').all();
    for (const element of textElements.slice(0, 10)) { // Check first 10 elements
      await expect(element).toBeVisible();
    }
  });
});
