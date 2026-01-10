import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Assessment Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
  });

  test('should complete speech assessment workflow', async ({ page }) => {
    // Navigate to Speech Analysis
    await page.click('text=Speech Analysis');
    await expect(page.locator('h1')).toContainText('Speech Analysis');

    // Check for recording interface
    await expect(page.locator('text=Voice Recording')).toBeVisible();
    
    // Mock MediaRecorder API for testing
    await page.addInitScript(() => {
      // @ts-ignore
      window.MediaRecorder = class {
        static isTypeSupported() { return true; }
        constructor() {
          this.state = 'inactive';
          this.ondataavailable = null;
          this.onstop = null;
        }
        start() {
          this.state = 'recording';
          setTimeout(() => {
            if (this.ondataavailable) {
              this.ondataavailable({ data: new Blob(['test'], { type: 'audio/wav' }) });
            }
          }, 100);
        }
        stop() {
          this.state = 'inactive';
          if (this.onstop) this.onstop();
        }
      };
      
      // @ts-ignore
      navigator.mediaDevices = {
        getUserMedia: () => Promise.resolve({
          getTracks: () => [{ stop: () => {} }]
        })
      };
    });

    // Start recording
    await page.click('text=Start Recording');
    await expect(page.locator('text=Stop Recording')).toBeVisible();

    // Stop recording
    await page.click('text=Stop Recording');
    
    // Should show analyze button
    await expect(page.locator('text=Analyze Recording')).toBeVisible();
  });

  test('should complete retinal assessment workflow', async ({ page }) => {
    // Navigate to Retinal Analysis
    await page.click('text=Retinal Analysis');
    await expect(page.locator('h1')).toContainText('Retinal Analysis');

    // Check for upload interface
    await expect(page.locator('text=Fundus Image Upload')).toBeVisible();
    
    // Create a test image file
    const testImagePath = path.join(__dirname, '../fixtures/test-retinal-image.jpg');
    
    // Mock file upload (since we can't create actual files in E2E tests)
    await page.evaluate(() => {
      // Create a mock file input change event
      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (fileInput) {
        const file = new File(['test image data'], 'test-retinal.jpg', { type: 'image/jpeg' });
        Object.defineProperty(fileInput, 'files', {
          value: [file],
          writable: false,
        });
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    // Should show analyze button after file selection
    await expect(page.locator('text=Analyze Image')).toBeVisible();
  });

  test('should complete motor assessment workflow', async ({ page }) => {
    // Navigate to Motor Assessment
    await page.click('text=Motor Assessment');
    await expect(page.locator('h1')).toContainText('Motor Function Assessment');

    // Check for finger tapping test
    await expect(page.locator('text=Finger Tapping Test')).toBeVisible();
    
    // Start tapping test
    await page.click('text=Start Tapping Test');
    
    // Should show countdown and tap button
    await expect(page.locator('[data-testid="tap-button"]')).toBeVisible();
    
    // Simulate tapping
    for (let i = 0; i < 5; i++) {
      await page.click('[data-testid="tap-button"]');
      await page.waitForTimeout(200);
    }
    
    // Wait for test to complete (or stop manually)
    await page.click('text=Stop Test');
    
    // Should show analyze button
    await expect(page.locator('text=Analyze Results')).toBeVisible();
  });

  test('should handle assessment history CRUD operations', async ({ page }) => {
    // Navigate to Assessment History
    await page.click('text=Assessment History');
    await expect(page.locator('h1')).toContainText('Assessment History');

    // Check for search and filter functionality
    await expect(page.locator('input[placeholder*="Search"]')).toBeVisible();
    await expect(page.locator('select')).toBeVisible();

    // Test search functionality
    await page.fill('input[placeholder*="Search"]', 'test');
    await page.waitForTimeout(500); // Wait for debounced search

    // Test filter functionality
    await page.selectOption('select:first-of-type', 'speech');
    await page.waitForTimeout(500);

    // Check for CRUD action buttons (if assessments exist)
    const assessmentRows = await page.locator('[data-testid="assessment-record"]').count();
    if (assessmentRows > 0) {
      await expect(page.locator('button:has-text("View Details")')).toBeVisible();
      await expect(page.locator('button:has-text("Export")')).toBeVisible();
      await expect(page.locator('button:has-text("Delete")')).toBeVisible();
    }
  });

  test('should handle file upload validation', async ({ page }) => {
    // Navigate to Retinal Analysis
    await page.click('text=Retinal Analysis');

    // Test invalid file type
    await page.evaluate(() => {
      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (fileInput) {
        const file = new File(['test'], 'test.txt', { type: 'text/plain' });
        Object.defineProperty(fileInput, 'files', {
          value: [file],
          writable: false,
        });
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    // Should show error message
    await expect(page.locator('text=Please upload a valid')).toBeVisible();
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Mock API error responses
    await page.route('**/api/v1/**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' })
      });
    });

    // Navigate to Speech Analysis
    await page.click('text=Speech Analysis');
    
    // Try to start an analysis (this should fail gracefully)
    await page.addInitScript(() => {
      // @ts-ignore
      window.MediaRecorder = class {
        static isTypeSupported() { return true; }
        constructor() {
          this.state = 'inactive';
          this.ondataavailable = null;
          this.onstop = null;
        }
        start() {
          this.state = 'recording';
          setTimeout(() => {
            if (this.ondataavailable) {
              this.ondataavailable({ data: new Blob(['test'], { type: 'audio/wav' }) });
            }
          }, 100);
        }
        stop() {
          this.state = 'inactive';
          if (this.onstop) this.onstop();
        }
      };
      
      // @ts-ignore
      navigator.mediaDevices = {
        getUserMedia: () => Promise.resolve({
          getTracks: () => [{ stop: () => {} }]
        })
      };
    });

    await page.click('text=Start Recording');
    await page.click('text=Stop Recording');
    await page.click('text=Analyze Recording');

    // Should show error message
    await expect(page.locator('text=Analysis failed')).toBeVisible();
    
    // App should still be functional
    await expect(page.locator('h1')).toContainText('Speech Analysis');
  });

  test('should maintain processing states correctly', async ({ page }) => {
    // Navigate to Motor Assessment
    await page.click('text=Motor Assessment');
    
    // Start tapping test
    await page.click('text=Start Tapping Test');
    
    // Should show processing state
    await expect(page.locator('[data-testid="tap-button"]')).toBeVisible();
    
    // Should prevent navigation during active test
    await page.click('text=Speech Analysis');
    
    // Should still be on motor assessment (if navigation is blocked during active test)
    // This depends on implementation - might need to adjust based on actual behavior
    await expect(page.locator('h1')).toContainText('Motor Function Assessment');
  });
});
