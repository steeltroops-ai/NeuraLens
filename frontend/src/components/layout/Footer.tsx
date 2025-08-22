'use client';

import React from 'react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="container mx-auto px-4 py-8">
        {/* Main Footer Content */}
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          {/* Brand Section */}
          <div className="flex items-center space-x-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-purple-600">
              <svg
                className="h-5 w-5 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.87-3.13-7-7-7zm0 2c2.76 0 5 2.24 5 5 0 1.64-.8 3.09-2.03 4H9.03C7.8 12.09 7 10.64 7 9c0-2.76 2.24-5 5-5zm-2 7h4v2h-4v-2z" />
              </svg>
            </div>
            <span className="text-xl font-bold text-slate-900">
              NeuroLens-X
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
