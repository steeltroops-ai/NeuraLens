'use client';

import React from 'react';
import Link from 'next/link';
import { Activity, Shield, Mail, Github, Twitter, Linkedin } from 'lucide-react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    platform: [
      { label: 'Dashboard', href: '/dashboard' },
      { label: 'Speech Analysis', href: '/dashboard/speech' },
      { label: 'Retinal Imaging', href: '/dashboard/retinal' },
      { label: 'Motor Assessment', href: '/dashboard/motor' },
      { label: 'Cognitive Testing', href: '/dashboard/cognitive' },
    ],
    company: [
      { label: 'About Us', href: '/about' },
      { label: 'Documentation', href: '/docs' },
      { label: 'API Reference', href: '/api' },
      { label: 'Changelog', href: '/changelog' },
    ],
    legal: [
      { label: 'Privacy Policy', href: '/privacy' },
      { label: 'Terms of Service', href: '/terms' },
      { label: 'HIPAA Compliance', href: '/hipaa' },
      { label: 'Security', href: '/security' },
    ],
  };

  return (
    <footer className="bg-black border-t border-zinc-900">
      {/* Main Footer */}
      <div className="mx-auto max-w-6xl px-4 sm:px-6 py-12">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4 lg:grid-cols-5">
          {/* Brand Column */}
          <div className="col-span-2 md:col-span-4 lg:col-span-2">
            <Link href="/" className="inline-flex items-center gap-2 mb-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800">
                <Activity className="h-4 w-4 text-red-500" />
              </div>
              <span className="text-[16px] font-semibold text-white">MediLens</span>
            </Link>
            <p className="text-[13px] text-zinc-500 leading-relaxed mb-5 max-w-sm">
              Democratizing advanced medical diagnostics through AI.
              A unified platform for multi-specialty AI-assisted diagnosis
              across ophthalmology, radiology, cardiology, and more.
            </p>
            <div className="flex items-center gap-3">
              <a
                href="https://twitter.com/medilens"
                target="_blank"
                rel="noopener noreferrer"
                className="flex h-8 w-8 items-center justify-center rounded-md bg-zinc-900 border border-zinc-800 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-white"
                aria-label="Twitter"
              >
                <Twitter className="h-4 w-4" />
              </a>
              <a
                href="https://github.com/medilens"
                target="_blank"
                rel="noopener noreferrer"
                className="flex h-8 w-8 items-center justify-center rounded-md bg-zinc-900 border border-zinc-800 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-white"
                aria-label="GitHub"
              >
                <Github className="h-4 w-4" />
              </a>
              <a
                href="https://linkedin.com/company/medilens"
                target="_blank"
                rel="noopener noreferrer"
                className="flex h-8 w-8 items-center justify-center rounded-md bg-zinc-900 border border-zinc-800 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-white"
                aria-label="LinkedIn"
              >
                <Linkedin className="h-4 w-4" />
              </a>
            </div>
          </div>

          {/* Platform Links */}
          <div>
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 mb-3">
              Platform
            </h3>
            <ul className="space-y-2">
              {footerLinks.platform.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-[13px] text-zinc-400 transition-colors hover:text-white"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 mb-3">
              Company
            </h3>
            <ul className="space-y-2">
              {footerLinks.company.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-[13px] text-zinc-400 transition-colors hover:text-white"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal Links */}
          <div>
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 mb-3">
              Legal
            </h3>
            <ul className="space-y-2">
              {footerLinks.legal.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-[13px] text-zinc-400 transition-colors hover:text-white"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-zinc-900">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-5">
          <div className="flex flex-col items-center justify-between gap-3 sm:flex-row">
            <p className="text-[12px] text-zinc-500">
              © {currentYear} MediLens. All rights reserved.
            </p>
            <div className="flex items-center gap-5">
              <div className="flex items-center gap-1.5 text-zinc-500">
                <Shield className="h-3.5 w-3.5 text-emerald-500" />
                <span className="text-[11px] font-medium">HIPAA Compliant</span>
              </div>
              <a
                href="mailto:contact@medilens.ai"
                className="flex items-center gap-1.5 text-zinc-500 transition-colors hover:text-zinc-300"
              >
                <Mail className="h-3.5 w-3.5" />
                <span className="text-[11px]">contact@medilens.ai</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
