import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global E2E test teardown...');
  
  try {
    // Clean up any test data, close connections, etc.
    // For example, clean up test users, reset database state, etc.
    
    console.log('✅ Global E2E test teardown completed');
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw here as it might mask test failures
  }
}

export default globalTeardown;
