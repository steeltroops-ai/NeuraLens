'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { Button, cn } from '@/components/ui';

interface NavigationItem {
  id: string;
  label: string;
  href: string;
  icon?: React.ReactNode;
  badge?: string | number;
  disabled?: boolean;
  external?: boolean;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'home',
    label: 'Home',
    href: '/',
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
        />
      </svg>
    ),
  },
  {
    id: 'dashboard',
    label: 'Dashboard',
    href: '/dashboard',
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
  },
  {
    id: 'assessment',
    label: 'Start Test',
    href: '/assessment',
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
        />
      </svg>
    ),
  },
];

export const Header: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  // Handle hydration
  useEffect(() => {
    setMounted(true);
  }, []);

  // Handle scroll effect
  useEffect(() => {
    if (!mounted) return;

    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    // Set initial scroll state
    handleScroll();

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [mounted]);

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [pathname]);

  // Handle mobile menu toggle
  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  // Handle keyboard navigation
  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      setIsMobileMenuOpen(false);
    }
  };

  return (
    <header
      className={cn(
        'fixed left-0 right-0 top-0 z-50 transition-all duration-300',
        // Always render initial state on server, apply scroll styles after hydration
        mounted && isScrolled
          ? 'border-b border-gray-200 bg-white/95 shadow-apple backdrop-blur-xl'
          : 'bg-white/80 backdrop-blur-sm'
      )}
      onKeyDown={handleKeyDown}
      suppressHydrationWarning
    >
      <nav
        className="container mx-auto px-4"
        role="navigation"
        aria-label="Main navigation"
        id="main-navigation"
      >
        <div className="flex h-16 items-center justify-between lg:h-20">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link
              href="/"
              className="flex items-center space-x-3 rounded-lg text-gray-900 transition-colors hover:text-medical-500 focus:outline-none focus:ring-2 focus:ring-medical-500 focus:ring-offset-2 focus:ring-offset-white"
              aria-label="NeuroLens-X Home"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-purple-600">
                <svg
                  className="h-5 w-5 text-white"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 2C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.87-3.13-7-7-7zm0 2c2.76 0 5 2.24 5 5 0 1.64-.8 3.09-2.03 4H9.03C7.8 12.09 7 10.64 7 9c0-2.76 2.24-5 5-5zm-2 7h4v2h-4v-2z" />
                </svg>
              </div>
              <span className="text-xl font-bold">NeuroLens-X</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex lg:items-center lg:space-x-8">
            {navigationItems.map((item) => {
              const isActive = pathname === item.href;
              const isDisabled = item.disabled;

              return (
                <Link
                  key={item.id}
                  href={isDisabled ? '#' : item.href}
                  className={cn(
                    'flex items-center space-x-2 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200',
                    'focus:outline-none focus:ring-2 focus:ring-medical-500 focus:ring-offset-2 focus:ring-offset-white',
                    isActive &&
                      'border border-medical-500/20 bg-medical-500/10 text-medical-600',
                    !isActive &&
                      !isDisabled &&
                      'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
                    isDisabled && 'cursor-not-allowed text-gray-400 opacity-50'
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  aria-disabled={isDisabled}
                  tabIndex={isDisabled ? -1 : 0}
                  {...(isDisabled && { onClick: (e) => e.preventDefault() })}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="ml-2 rounded-full bg-primary-500 px-2 py-1 text-xs text-white">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>

          {/* CTA Button */}
          <div className="hidden lg:flex lg:items-center lg:space-x-4">
            <Button
              variant="primary"
              size="md"
              onClick={() => router.push('/assessment')}
              className="font-semibold transition-all duration-200 hover:scale-105"
            >
              Start Health Check
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden">
            <Button
              variant="ghost"
              size="md"
              onClick={toggleMobileMenu}
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
              className="p-2"
            >
              {isMobileMenuOpen ? (
                <svg
                  className="h-6 w-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                <svg
                  className="h-6 w-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div
          id="mobile-menu"
          className={cn(
            'overflow-hidden transition-all duration-300 ease-in-out lg:hidden',
            isMobileMenuOpen ? 'max-h-96 pb-4 opacity-100' : 'max-h-0 opacity-0'
          )}
          aria-hidden={!isMobileMenuOpen}
        >
          <div className="space-y-2 border-t border-gray-200 pb-2 pt-4">
            {navigationItems.map((item) => {
              const isActive = pathname === item.href;
              const isDisabled = item.disabled;

              return (
                <Link
                  key={item.id}
                  href={isDisabled ? '#' : item.href}
                  className={cn(
                    'flex items-center space-x-3 rounded-lg px-4 py-3 text-base font-medium transition-all duration-200',
                    'focus:outline-none focus:ring-2 focus:ring-medical-500 focus:ring-offset-2 focus:ring-offset-white',
                    isActive &&
                      'border border-medical-500/20 bg-medical-500/10 text-medical-600',
                    !isActive &&
                      !isDisabled &&
                      'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
                    isDisabled && 'cursor-not-allowed text-gray-400 opacity-50'
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  aria-disabled={isDisabled}
                  tabIndex={isDisabled ? -1 : 0}
                  {...(isDisabled && { onClick: (e) => e.preventDefault() })}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="ml-auto rounded-full bg-primary-500 px-2 py-1 text-xs text-white">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}

            {/* Mobile CTA */}
            <div className="px-4 pt-4">
              <Button
                variant="primary"
                size="lg"
                onClick={() => router.push('/assessment')}
                className="w-full font-semibold transition-all duration-200 hover:scale-105"
              >
                Start Health Check
              </Button>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
