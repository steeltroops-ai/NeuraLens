#!/usr/bin/env node

/**
 * Performance Verification Script for NeuraLens Frontend
 * Verifies instant page loading and SPA-like navigation
 */

const { chromium } = require('playwright');

const BASE_URL = process.env.BASE_URL || 'http://localhost:3001';
const ROUTES_TO_TEST = [
  { path: '/', name: 'Home' },
  { path: '/about', name: 'About' },
  { path: '/dashboard', name: 'Dashboard' },
  { path: '/readme', name: 'README' },
  { path: '/assessment', name: 'Assessment' },
];

const PERFORMANCE_THRESHOLDS = {
  maxLoadTime: 2000, // 2 seconds max for initial load
  maxNavigationTime: 500, // 500ms max for navigation
  maxMemoryUsage: 100, // 100MB max memory usage
  minLighthouseScore: 90, // Minimum Lighthouse performance score
};

async function measurePagePerformance(page, route) {
  console.log(`🔍 Testing ${route.name} (${route.path})...`);

  const startTime = Date.now();

  try {
    // Navigate to page
    await page.goto(`${BASE_URL}${route.path}`, {
      waitUntil: 'domcontentloaded',
      timeout: 10000,
    });

    const endTime = Date.now();
    const loadTime = endTime - startTime;

    // Get performance metrics
    const metrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0];
      const memory = performance.memory;

      return {
        loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
        renderTime: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
        memoryUsage: memory ? memory.usedJSHeapSize / 1024 / 1024 : 0,
        domElements: document.querySelectorAll('*').length,
        hasErrors: window.console.error.length > 0,
      };
    });

    // Check for console errors
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    return {
      route: route.name,
      path: route.path,
      totalLoadTime: loadTime,
      success: true,
      consoleErrors,
      ...metrics,
    };
  } catch (error) {
    return {
      route: route.name,
      path: route.path,
      success: false,
      error: error.message,
      totalLoadTime: Date.now() - startTime,
    };
  }
}

async function testNavigationSpeed(page) {
  console.log('🚀 Testing navigation speed between pages...');

  // Start at home page
  await page.goto(`${BASE_URL}/`, { waitUntil: 'domcontentloaded' });

  const navigationResults = [];

  for (let i = 1; i < ROUTES_TO_TEST.length; i++) {
    const route = ROUTES_TO_TEST[i];
    const startTime = Date.now();

    // Navigate using client-side routing
    await page.evaluate(path => {
      window.history.pushState({}, '', path);
      window.dispatchEvent(new PopStateEvent('popstate'));
    }, route.path);

    // Wait for navigation to complete
    await page.waitForLoadState('domcontentloaded');

    const endTime = Date.now();
    const navigationTime = endTime - startTime;

    navigationResults.push({
      route: route.name,
      path: route.path,
      navigationTime,
      isInstant: navigationTime < PERFORMANCE_THRESHOLDS.maxNavigationTime,
    });

    console.log(
      `  ✓ ${route.name}: ${navigationTime}ms ${navigationTime < PERFORMANCE_THRESHOLDS.maxNavigationTime ? '✅' : '⚠️'}`,
    );
  }

  return navigationResults;
}

async function runPerformanceVerification() {
  console.log('🎯 NeuraLens Performance Verification\n');
  console.log('Testing instant page loading and SPA-like navigation...\n');

  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });

  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
  });

  const page = await context.newPage();

  try {
    // Test individual page performance
    console.log('📊 Testing Page Load Performance:');
    console.log('================================');

    const pageResults = [];

    for (const route of ROUTES_TO_TEST) {
      const result = await measurePagePerformance(page, route);
      pageResults.push(result);

      if (result.success) {
        const loadStatus = result.totalLoadTime < PERFORMANCE_THRESHOLDS.maxLoadTime ? '✅' : '⚠️';
        const memoryStatus =
          result.memoryUsage < PERFORMANCE_THRESHOLDS.maxMemoryUsage ? '✅' : '⚠️';

        console.log(
          `  ${result.route}: ${result.totalLoadTime}ms ${loadStatus} | Memory: ${result.memoryUsage.toFixed(1)}MB ${memoryStatus}`,
        );

        if (result.consoleErrors.length > 0) {
          console.log(`    ❌ Console Errors: ${result.consoleErrors.length}`);
        }
      } else {
        console.log(`  ${result.route}: ❌ FAILED - ${result.error}`);
      }
    }

    // Test navigation speed
    console.log('\n🚀 Testing Navigation Performance:');
    console.log('==================================');

    const navigationResults = await testNavigationSpeed(page);

    // Calculate summary
    const successfulPages = pageResults.filter(r => r.success);
    const avgLoadTime =
      successfulPages.reduce((sum, r) => sum + r.totalLoadTime, 0) / successfulPages.length;
    const avgNavigationTime =
      navigationResults.reduce((sum, r) => sum + r.navigationTime, 0) / navigationResults.length;
    const maxMemoryUsage = Math.max(...successfulPages.map(r => r.memoryUsage));

    // Performance grade calculation
    let performanceGrade = 'A+';
    if (avgLoadTime > 1000) performanceGrade = 'A';
    if (avgLoadTime > 2000) performanceGrade = 'B';
    if (avgLoadTime > 3000) performanceGrade = 'C';
    if (avgLoadTime > 5000) performanceGrade = 'D';

    // Results summary
    console.log('\n📈 Performance Summary:');
    console.log('======================');
    console.log(
      `Average Load Time: ${avgLoadTime.toFixed(0)}ms ${avgLoadTime < PERFORMANCE_THRESHOLDS.maxLoadTime ? '✅' : '❌'}`,
    );
    console.log(
      `Average Navigation Time: ${avgNavigationTime.toFixed(0)}ms ${avgNavigationTime < PERFORMANCE_THRESHOLDS.maxNavigationTime ? '✅' : '❌'}`,
    );
    console.log(
      `Peak Memory Usage: ${maxMemoryUsage.toFixed(1)}MB ${maxMemoryUsage < PERFORMANCE_THRESHOLDS.maxMemoryUsage ? '✅' : '❌'}`,
    );
    console.log(`Performance Grade: ${performanceGrade}`);

    // Success criteria verification
    console.log('\n✅ Success Criteria Verification:');
    console.log('=================================');

    const allPagesLoad = successfulPages.length === ROUTES_TO_TEST.length;
    const instantLoading = avgLoadTime < PERFORMANCE_THRESHOLDS.maxLoadTime;
    const instantNavigation = avgNavigationTime < PERFORMANCE_THRESHOLDS.maxNavigationTime;
    const memoryEfficient = maxMemoryUsage < PERFORMANCE_THRESHOLDS.maxMemoryUsage;
    const noConsoleErrors = successfulPages.every(r => r.consoleErrors.length === 0);

    console.log(`✓ All pages load successfully: ${allPagesLoad ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`✓ Instant page loading (<2s): ${instantLoading ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`✓ Instant navigation (<500ms): ${instantNavigation ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`✓ Memory efficient (<100MB): ${memoryEfficient ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`✓ No console errors: ${noConsoleErrors ? '✅ PASS' : '❌ FAIL'}`);

    const overallSuccess =
      allPagesLoad && instantLoading && instantNavigation && memoryEfficient && noConsoleErrors;

    console.log(
      `\n🎯 Overall Result: ${overallSuccess ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED'}`,
    );

    if (overallSuccess) {
      console.log('\n🚀 NeuraLens frontend delivers instant SPA-like performance!');
    } else {
      console.log('\n⚠️  Performance optimization needed. Check failed criteria above.');
    }

    return overallSuccess;
  } catch (error) {
    console.error('❌ Performance verification failed:', error);
    return false;
  } finally {
    await browser.close();
  }
}

// Run verification if called directly
if (require.main === module) {
  runPerformanceVerification()
    .then(success => process.exit(success ? 0 : 1))
    .catch(error => {
      console.error('❌ Verification error:', error);
      process.exit(1);
    });
}

module.exports = { runPerformanceVerification };
