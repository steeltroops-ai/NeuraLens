'use client';

import React from 'react';
import Link from 'next/link';
import { Brain, Shield, Mail } from 'lucide-react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white border-t border-[#E5E5EA]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {/* Company Info */}
          <div className="col-span-1 sm:col-span-2">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#007AFF] to-[#0062CC]">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold text-[#000000]">MediLens</span>
            </div>
            <p className="mb-6 max-w-md text-sm text-[#3C3C43]">
              Democratizing advanced medical diagnostics through AI technology.
              A unified platform for multi-specialty AI-assisted diagnosis.
            </p>
            <div className="flex items-center gap-2 text-[#8E8E93]">
              <Mail className="h-4 w-4" />
              <span className="text-sm">contact@medilens.ai</span>
            </div>
          </div>

          {/* Platform Links */}
          <div>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-[#8E8E93]">Platform</h3>
            <ul className="space-y-3">
              <li>
                <Link href="/assessment" className="text-sm text-[#3C3C43] transition-colors hover:text-[#007AFF]">
                  Start Assessment
                </Link>
              </li>
              <li>
                <Link href="/dashboard" className="text-sm text-[#3C3C43] transition-colors hover:text-[#007AFF]">
                  Dashboard
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-sm text-[#3C3C43] transition-colors hover:text-[#007AFF]">
                  About Us
                </Link>
              </li>
            </ul>
          </div>

          {/* Support Links */}
          <div>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-[#8E8E93]">Support</h3>
            <ul className="space-y-3">
              <li>
                <Link href="/help" className="text-sm text-[#3C3C43] transition-colors hover:text-[#007AFF]">
                  Help Center
                </Link>
              </li>
              <li>
                <Link href="/privacy" className="text-sm text-[#3C3C43] transition-colors hover:text-[#007AFF]">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="text-sm text-[#3C3C43] transition-colors hover:text-[#007AFF]">
                  Terms of Service
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-[#E5E5EA] pt-8 sm:flex-row">
          <p className="text-sm text-[#8E8E93]">© {currentYear} MediLens. All rights reserved.</p>
          <div className="flex items-center gap-2 text-[#8E8E93]">
            <Shield className="h-4 w-4" />
            <span className="text-sm">HIPAA Compliant</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
