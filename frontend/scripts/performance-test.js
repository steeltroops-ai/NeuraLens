#!/usr/bin/env node

/**
 * Performance Testing Script for NeuraLens Frontend
 * Tests page loading performance and navigation speed
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.BASE_URL || 'http://localhost:3001';
const ROUTES_TO_TEST = ['/', '/dashboard', '/about', '/assessment', '/readme'];

async function measurePagePerformance(page, url) {
  console.log(`📊 Testing: ${url}`);

  // Start performance measurement
  const startTime = Date.now();

  // Navigate to page
  await page.goto(`${BASE_URL}${url}`, {
    waitUntil: 'networkidle',
    timeout: 30000,
  });

  // Wait for page to be fully interactive
  await page.waitForLoadState('domcontentloaded');

  const endTime = Date.now();
  const loadTime = endTime - startTime;

  // Get performance metrics
  const metrics = await page.evaluate(() => {
    const navigation = performance.getEntriesByType('navigation')[0];
    const memory = performance.memory;

    return {
      loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
      renderTime: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
      interactionTime: navigation ? navigation.domInteractive - navigation.fetchStart : 0,
      memoryUsage: memory ? memory.usedJSHeapSize / 1024 / 1024 : 0,
      ttfb: navigation ? navigation.responseStart - navigation.requestStart : 0,
      domElements: document.querySelectorAll('*').length,
      jsHeapSize: memory ? memory.totalJSHeapSize / 1024 / 1024 : 0,
    };
  });

  return {
    url,
    totalLoadTime: loadTime,
    ...metrics,
  };
}

async function testNavigationSpeed(page) {
  console.log('🚀 Testing navigation speed...');

  // Start at home page
  await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' });

  const navigationTimes = [];

  for (const route of ROUTES_TO_TEST.slice(1)) {
    const startTime = Date.now();

    // Click navigation link or navigate programmatically
    await page.evaluate(url => {
      window.location.href = url;
    }, `${BASE_URL}${route}`);

    await page.waitForLoadState('domcontentloaded');

    const endTime = Date.now();
    const navigationTime = endTime - startTime;

    navigationTimes.push({
      route,
      navigationTime,
    });

    console.log(`  ✓ ${route}: ${navigationTime}ms`);
  }

  return navigationTimes;
}

async function runPerformanceTests() {
  console.log('🎯 Starting NeuraLens Performance Tests\n');

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
    const pageMetrics = [];

    for (const route of ROUTES_TO_TEST) {
      const metrics = await measurePagePerformance(page, route);
      pageMetrics.push(metrics);
    }

    // Test navigation speed
    const navigationMetrics = await testNavigationSpeed(page);

    // Generate performance report
    const report = {
      timestamp: new Date().toISOString(),
      baseUrl: BASE_URL,
      pageMetrics,
      navigationMetrics,
      summary: {
        averageLoadTime:
          pageMetrics.reduce((sum, m) => sum + m.totalLoadTime, 0) / pageMetrics.length,
        averageRenderTime:
          pageMetrics.reduce((sum, m) => sum + m.renderTime, 0) / pageMetrics.length,
        averageNavigationTime:
          navigationMetrics.reduce((sum, m) => sum + m.navigationTime, 0) /
          navigationMetrics.length,
        totalMemoryUsage: Math.max(...pageMetrics.map(m => m.memoryUsage)),
        performanceGrade: 'A', // Will be calculated based on metrics
      },
    };

    // Calculate performance grade
    const avgLoadTime = report.summary.averageLoadTime;
    if (avgLoadTime < 1000) report.summary.performanceGrade = 'A+';
    else if (avgLoadTime < 2000) report.summary.performanceGrade = 'A';
    else if (avgLoadTime < 3000) report.summary.performanceGrade = 'B';
    else if (avgLoadTime < 5000) report.summary.performanceGrade = 'C';
    else report.summary.performanceGrade = 'D';

    // Save report
    const reportPath = path.join(__dirname, '../performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    // Display results
    console.log('\n📈 Performance Test Results:');
    console.log('================================');
    console.log(`Average Load Time: ${report.summary.averageLoadTime.toFixed(0)}ms`);
    console.log(`Average Render Time: ${report.summary.averageRenderTime.toFixed(0)}ms`);
    console.log(`Average Navigation Time: ${report.summary.averageNavigationTime.toFixed(0)}ms`);
    console.log(`Peak Memory Usage: ${report.summary.totalMemoryUsage.toFixed(1)}MB`);
    console.log(`Performance Grade: ${report.summary.performanceGrade}`);
    console.log(`\nDetailed report saved to: ${reportPath}`);

    // Performance recommendations
    console.log('\n💡 Performance Analysis:');
    if (report.summary.averageLoadTime < 1000) {
      console.log('✅ Excellent! Pages load instantly with SPA-like performance');
    } else if (report.summary.averageLoadTime < 2000) {
      console.log('✅ Good performance, pages load quickly');
    } else {
      console.log('⚠️  Performance could be improved. Consider further optimizations.');
    }

    if (report.summary.averageNavigationTime < 500) {
      console.log('✅ Navigation is instant and smooth');
    } else {
      console.log('⚠️  Navigation could be faster. Check for blocking operations.');
    }
  } catch (error) {
    console.error('❌ Performance test failed:', error);
  } finally {
    await browser.close();
  }
}

// Run tests if called directly
if (require.main === module) {
  runPerformanceTests().catch(console.error);
}

module.exports = { runPerformanceTests };
