/** @type {import('next').NextConfig} */

const nextConfig = {
  // React configuration
  reactStrictMode: true,

  // Experimental features
  experimental: {
    optimizePackageImports: ['lucide-react', 'recharts'],
  },

  // TypeScript configuration
  typescript: {
    ignoreBuildErrors: false,
  },

  // ESLint configuration
  eslint: {
    ignoreDuringBuilds: true,
  },
};

module.exports = nextConfig;
