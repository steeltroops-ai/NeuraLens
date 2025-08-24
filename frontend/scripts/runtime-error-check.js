#!/usr/bin/env node

/**
 * Runtime Error Check Script for NeuraLens Frontend
 * Systematically checks all pages for console errors, warnings, and network failures
 */

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const PAGES_TO_TEST = [
  { path: '/', name: 'Home Page' },
  { path: '/about', name: 'About Page' },
  { path: '/dashboard', name: 'Dashboard Page' },
  { path: '/readme', name: 'README Page' },
  { path: '/assessment', name: 'Assessment Page' }
];

console.log('🔍 NeuraLens Runtime Error Analysis');
console.log('===================================');
console.log(`Testing pages at: ${BASE_URL}`);
console.log(`Pages to test: ${PAGES_TO_TEST.length}`);
console.log('');

console.log('📋 MANUAL TESTING CHECKLIST:');
console.log('============================');
console.log('');

PAGES_TO_TEST.forEach((page, index) => {
  console.log(`${index + 1}. ${page.name} (${page.path})`);
  console.log(`   URL: ${BASE_URL}${page.path}`);
  console.log('   ✅ Check for:');
  console.log('      - No red error messages in console');
  console.log('      - No yellow warning messages in console');
  console.log('      - No failed network requests (404s, 500s)');
  console.log('      - No React hydration errors');
  console.log('      - Page loads completely without crashes');
  console.log('      - All interactive elements work properly');
  console.log('      - Navigation between pages works smoothly');
  console.log('');
});

console.log('🔧 HOW TO TEST:');
console.log('===============');
console.log('1. Open browser developer tools (F12)');
console.log('2. Go to Console tab');
console.log('3. Clear console (Ctrl+L or Clear button)');
console.log('4. Visit each URL above');
console.log('5. Document any errors or warnings');
console.log('6. Test navigation between pages');
console.log('7. Check Network tab for failed requests');
console.log('');

console.log('📊 SUCCESS CRITERIA:');
console.log('====================');
console.log('✅ Zero red error messages in console');
console.log('✅ Zero yellow warning messages in console');
console.log('✅ All network requests succeed (200 status)');
console.log('✅ No React hydration mismatches');
console.log('✅ Smooth navigation between all pages');
console.log('✅ All pages load completely');
console.log('✅ Interactive elements respond properly');
console.log('');

console.log('🚀 QUICK TEST URLS:');
console.log('===================');
PAGES_TO_TEST.forEach(page => {
  console.log(`${BASE_URL}${page.path}`);
});
console.log('');

console.log('💡 COMMON ISSUES TO LOOK FOR:');
console.log('==============================');
console.log('- "Cannot read property" errors');
console.log('- "Module not found" errors');
console.log('- "Hydration failed" warnings');
console.log('- "Warning: Each child should have a unique key" warnings');
console.log('- Failed API calls or missing endpoints');
console.log('- Missing images or assets (404 errors)');
console.log('- TypeScript compilation errors');
console.log('- React component lifecycle errors');
console.log('');

console.log('🎯 NEXT STEPS:');
console.log('==============');
console.log('1. Test each page manually using the checklist above');
console.log('2. Document any issues found');
console.log('3. Fix issues systematically');
console.log('4. Re-test after each fix');
console.log('5. Verify all success criteria are met');
console.log('');

console.log('✨ Ready to begin manual testing!');
console.log('Open your browser and start with the Home page:');
console.log(`${BASE_URL}/`);
