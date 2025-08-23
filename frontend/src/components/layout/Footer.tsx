'use client';

import React from 'react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-slate-100 bg-white/95 backdrop-blur-md">
      <div className="container mx-auto px-4 py-12">
        {/* Main Footer Content */}
        <div className="flex flex-col items-center justify-center space-y-6 text-center">
          {/* Brand Section */}
          <div className="flex items-center space-x-3">
            {/* Neural Node Logo */}
            <div className="relative">
              <svg
                className="h-8 w-8"
                viewBox="0 0 32 32"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                {/* Neural nodes */}
                <circle
                  cx="8"
                  cy="16"
                  r="6"
                  fill="url(#neuralGradientFooter)"
                  stroke="#1e3a8a"
                  strokeWidth="1"
                />
                <circle
                  cx="24"
                  cy="12"
                  r="4"
                  fill="url(#neuralGradientFooter)"
                  stroke="#1e3a8a"
                  strokeWidth="1"
                />

                {/* Connection line */}
                <line
                  x1="14"
                  y1="16"
                  x2="20"
                  y2="12"
                  stroke="#1e3a8a"
                  strokeWidth="2"
                  opacity="0.7"
                />

                {/* Gradient definition */}
                <defs>
                  <linearGradient
                    id="neuralGradientFooter"
                    x1="0%"
                    y1="0%"
                    x2="100%"
                    y2="100%"
                  >
                    <stop offset="0%" stopColor="#0d9488" />
                    <stop offset="100%" stopColor="#3b82f6" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span
              className="text-xl font-bold text-slate-900"
              style={{ fontFamily: 'Inter, sans-serif' }}
            >
              Neuralens
            </span>
          </div>

          {/* Tagline */}
          <p className="max-w-md text-sm text-slate-600">
            Early Detection, Better Outcomes.
          </p>

          {/* Copyright */}
          <div className="text-xs text-slate-500">
            © {currentYear} NeuroLens-X. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
};
