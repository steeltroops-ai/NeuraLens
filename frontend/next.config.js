/** @type {import('next').NextConfig} */

const nextConfig = {
  reactStrictMode: true,

  // Basic optimizations
  experimental: {
    optimizePackageImports: ['lucide-react', 'framer-motion'],
  },

  // Remove console logs in production
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Image optimization
  images: {
    formats: ['image/webp', 'image/avif'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },

  // TypeScript configuration
  typescript: {
    ignoreBuildErrors: false,
  },

  // Exclude test pages from build
  pageExtensions: ['tsx', 'ts', 'jsx', 'js'],

  async redirects() {
    return [
      // Redirect test pages in production
      ...(process.env.NODE_ENV === 'production'
        ? [
          {
            source: '/api-test',
            destination: '/',
            permanent: false,
          },
          {
            source: '/accessibility-test',
            destination: '/',
            permanent: false,
          },
          {
            source: '/assessment-workflow-test',
            destination: '/',
            permanent: false,
          },
          {
            source: '/comprehensive-dashboard',
            destination: '/dashboard',
            permanent: false,
          },
          {
            source: '/assessment',
            destination: '/dashboard',
            permanent: false,
          },
          {
            source: '/readme',
            destination: '/dashboard',
            permanent: false,
          },
          {
            source: '/results',
            destination: '/dashboard',
            permanent: false,
          },
          {
            source: '/help',
            destination: '/dashboard',
            permanent: false,
          },
        ]
        : []),
    ];
  },

  // Headers for performance and caching
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
      // Cache static assets aggressively
      {
        source: '/static/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      // Cache API responses with shorter TTL
      {
        source: '/api/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=300, s-maxage=300',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
