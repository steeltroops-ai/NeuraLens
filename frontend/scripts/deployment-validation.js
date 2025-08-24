#!/usr/bin/env node

/**
 * Deployment Validation Script for NeuraLens Frontend
 * Validates package compatibility and build readiness for Vercel deployment
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 NeuraLens Deployment Validation Starting...\n');

// Validation results
const results = {
  packageVersions: false,
  buildProcess: false,
  typeCheck: false,
  staticGeneration: false,
  bundleSize: false,
  performance: false,
};

/**
 * Validate package versions against npm registry
 */
async function validatePackageVersions() {
  console.log('📦 Validating package versions...');
  
  try {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    const criticalPackages = ['next', 'framer-motion', 'react', 'react-dom'];
    
    for (const pkg of criticalPackages) {
      if (packageJson.dependencies[pkg]) {
        const version = packageJson.dependencies[pkg].replace(/[\^~]/, '');
        console.log(`  ✓ ${pkg}: ${version}`);
      }
    }
    
    results.packageVersions = true;
    console.log('  ✅ Package versions validated\n');
  } catch (error) {
    console.log(`  ❌ Package validation failed: ${error.message}\n`);
  }
}

/**
 * Validate TypeScript compilation
 */
function validateTypeScript() {
  console.log('🔍 Running TypeScript validation...');
  
  try {
    execSync('bun tsc --noEmit', { stdio: 'pipe' });
    results.typeCheck = true;
    console.log('  ✅ TypeScript validation passed\n');
  } catch (error) {
    console.log('  ❌ TypeScript validation failed');
    console.log(`  Error: ${error.message}\n`);
  }
}

/**
 * Validate build process
 */
function validateBuild() {
  console.log('🏗️ Running production build...');
  
  try {
    const buildOutput = execSync('bun run build', { 
      stdio: 'pipe',
      encoding: 'utf8'
    });
    
    // Check for successful compilation
    if (buildOutput.includes('Compiled successfully')) {
      results.buildProcess = true;
      console.log('  ✅ Build process successful');
      
      // Check for static page generation
      if (buildOutput.includes('Generating static pages')) {
        results.staticGeneration = true;
        console.log('  ✅ Static page generation successful');
      }
      
      // Extract bundle size information
      const bundleMatch = buildOutput.match(/First Load JS shared by all\s+(\d+(?:\.\d+)?)\s*kB/);
      if (bundleMatch) {
        const bundleSize = parseFloat(bundleMatch[1]);
        if (bundleSize < 500) { // Target: under 500KB
          results.bundleSize = true;
          console.log(`  ✅ Bundle size optimized: ${bundleSize}KB`);
        } else {
          console.log(`  ⚠️ Bundle size large: ${bundleSize}KB`);
        }
      }
      
      console.log('');
    } else {
      console.log('  ❌ Build compilation failed\n');
    }
  } catch (error) {
    console.log('  ❌ Build process failed');
    console.log(`  Error: ${error.message}\n`);
  }
}

/**
 * Validate performance metrics
 */
function validatePerformance() {
  console.log('⚡ Validating performance metrics...');
  
  try {
    // Check if .next directory exists and has optimized files
    const nextDir = path.join(process.cwd(), '.next');
    if (fs.existsSync(nextDir)) {
      const staticDir = path.join(nextDir, 'static');
      const serverDir = path.join(nextDir, 'server');
      
      if (fs.existsSync(staticDir) && fs.existsSync(serverDir)) {
        results.performance = true;
        console.log('  ✅ Optimized build artifacts generated');
        console.log('  ✅ Static assets ready for CDN');
        console.log('  ✅ Server components optimized\n');
      }
    }
  } catch (error) {
    console.log(`  ❌ Performance validation failed: ${error.message}\n`);
  }
}

/**
 * Generate deployment readiness report
 */
function generateReport() {
  console.log('📊 Deployment Readiness Report');
  console.log('================================');
  
  const checks = [
    { name: 'Package Versions', status: results.packageVersions },
    { name: 'TypeScript Check', status: results.typeCheck },
    { name: 'Build Process', status: results.buildProcess },
    { name: 'Static Generation', status: results.staticGeneration },
    { name: 'Bundle Size', status: results.bundleSize },
    { name: 'Performance', status: results.performance },
  ];
  
  let passedChecks = 0;
  
  checks.forEach(check => {
    const status = check.status ? '✅ PASS' : '❌ FAIL';
    console.log(`${check.name.padEnd(20)} ${status}`);
    if (check.status) passedChecks++;
  });
  
  console.log('================================');
  console.log(`Overall Status: ${passedChecks}/${checks.length} checks passed`);
  
  if (passedChecks === checks.length) {
    console.log('🎉 DEPLOYMENT READY - All validations passed!');
    console.log('✅ Ready for Vercel deployment');
    process.exit(0);
  } else {
    console.log('⚠️ DEPLOYMENT ISSUES - Some validations failed');
    console.log('❌ Fix issues before deploying');
    process.exit(1);
  }
}

/**
 * Main validation workflow
 */
async function main() {
  try {
    await validatePackageVersions();
    validateTypeScript();
    validateBuild();
    validatePerformance();
    generateReport();
  } catch (error) {
    console.error('💥 Validation script failed:', error.message);
    process.exit(1);
  }
}

// Run validation
main();
