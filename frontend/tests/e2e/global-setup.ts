import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global E2E test setup...');
  
  // Launch browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for the development server to be ready
    console.log('⏳ Waiting for development server...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    
    // Verify the app is loaded
    await page.waitForSelector('body', { timeout: 30000 });
    console.log('✅ Development server is ready');
    
    // You can add authentication setup here if needed
    // For example, create test users, set up test data, etc.
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
  
  console.log('✅ Global E2E test setup completed');
}

export default globalSetup;
