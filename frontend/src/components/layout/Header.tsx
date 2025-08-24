'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import React, { useState, useEffect } from 'react';

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
      <svg className='h-5 w-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
        <path
          strokeLinecap='round'
          strokeLinejoin='round'
          strokeWidth={2}
          d='M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6'
        />
      </svg>
    ),
  },
  {
    id: 'dashboard',
    label: 'Dashboard',
    href: '/dashboard',
    icon: (
      <svg className='h-5 w-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
        <path
          strokeLinecap='round'
          strokeLinejoin='round'
          strokeWidth={2}
          d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'
        />
      </svg>
    ),
  },
  {
    id: 'about',
    label: 'About',
    href: '/about',
    icon: (
      <svg className='h-5 w-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
        <path
          strokeLinecap='round'
          strokeLinejoin='round'
          strokeWidth={2}
          d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
        />
      </svg>
    ),
  },
  {
    id: 'readme',
    label: 'README',
    href: '/readme',
    icon: (
      <svg className='h-5 w-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
        <path
          strokeLinecap='round'
          strokeLinejoin='round'
          strokeWidth={2}
          d='M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'
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
          ? 'border-b border-slate-100 bg-white/95 shadow-sm backdrop-blur-xl'
          : 'bg-white/90 backdrop-blur-md',
      )}
      onKeyDown={handleKeyDown}
      suppressHydrationWarning
    >
      <nav
        className='container mx-auto px-4'
        role='navigation'
        aria-label='Main navigation'
        id='main-navigation'
      >
        <div className='flex h-16 items-center justify-between lg:h-20'>
          {/* Brand */}
          <div className='flex-shrink-0'>
            <Link
              href='/'
              className='text-xl font-semibold transition-colors duration-200 hover:no-underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
              style={{ color: '#1D1D1F' }}
              aria-label='NeuraLens Home'
            >
              NeuraLens
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className='hidden lg:flex lg:items-center lg:space-x-8'>
            {navigationItems.map(item => {
              const isActive = pathname === item.href;
              const isDisabled = item.disabled;

              return (
                <Link
                  key={item.id}
                  href={isDisabled ? '#' : item.href}
                  className={cn(
                    'text-sm font-medium transition-colors duration-200 hover:no-underline',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
                    isActive && 'text-gray-900',
                    !isActive && !isDisabled && 'text-gray-600 hover:text-gray-900',
                    isDisabled && 'cursor-not-allowed text-gray-400 opacity-50',
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  aria-disabled={isDisabled}
                  tabIndex={isDisabled ? -1 : 0}
                  {...(isDisabled && { onClick: e => e.preventDefault() })}
                >
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className='ml-2 rounded-full bg-blue-500 px-2 py-1 text-xs text-white'>
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>

          {/* CTA and Sign In Buttons */}
          <div className='hidden lg:flex lg:items-center lg:space-x-4'>
            <button
              onClick={() => router.push('/login')}
              className='rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-200 hover:no-underline focus:outline-none focus:ring-2 focus:ring-blue-500'
              style={{ color: '#8E8E93' }}
            >
              Sign In
            </button>
            <button
              onClick={() => {
                // Trigger sidebar assessment instead of navigation
                const event = new CustomEvent('openAssessmentSidebar');
                window.dispatchEvent(event);
              }}
              className='hover:scale-98 rounded-lg px-4 py-2 text-sm font-semibold text-white transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500'
              style={{
                backgroundColor: '#007AFF',
                backdropFilter: 'blur(20px)',
              }}
              onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = '#0056CC';
              }}
              onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = '#007AFF';
              }}
            >
              Start Health Check
            </button>
          </div>

          {/* Mobile Menu Button */}
          <div className='lg:hidden'>
            <Button
              variant='ghost'
              size='md'
              onClick={toggleMobileMenu}
              aria-expanded={isMobileMenuOpen}
              aria-controls='mobile-menu'
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
              className='p-2'
            >
              {isMobileMenuOpen ? (
                <svg className='h-6 w-6' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth={2}
                    d='M6 18L18 6M6 6l12 12'
                  />
                </svg>
              ) : (
                <svg className='h-6 w-6' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth={2}
                    d='M4 6h16M4 12h16M4 18h16'
                  />
                </svg>
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div
          id='mobile-menu'
          className={cn(
            'overflow-hidden transition-all duration-300 ease-in-out lg:hidden',
            isMobileMenuOpen ? 'max-h-96 pb-4 opacity-100' : 'max-h-0 opacity-0',
          )}
          aria-hidden={!isMobileMenuOpen}
        >
          <div className='space-y-2 border-t border-gray-200 pb-2 pt-4'>
            {navigationItems.map(item => {
              const isActive = pathname === item.href;
              const isDisabled = item.disabled;

              return (
                <Link
                  key={item.id}
                  href={isDisabled ? '#' : item.href}
                  className={cn(
                    'flex items-center space-x-3 rounded-lg px-4 py-3 text-base font-medium transition-all duration-200',
                    'focus:outline-none focus:ring-2 focus:ring-medical-500 focus:ring-offset-2 focus:ring-offset-white',
                    isActive && 'border border-medical-500/20 bg-medical-500/10 text-medical-600',
                    !isActive &&
                      !isDisabled &&
                      'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
                    isDisabled && 'cursor-not-allowed text-gray-400 opacity-50',
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  aria-disabled={isDisabled}
                  tabIndex={isDisabled ? -1 : 0}
                  {...(isDisabled && { onClick: e => e.preventDefault() })}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className='ml-auto rounded-full bg-primary-500 px-2 py-1 text-xs text-white'>
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}

            {/* Mobile CTA */}
            <div className='px-4 pt-4'>
              <Button
                variant='primary'
                size='lg'
                onClick={() => router.push('/assessment')}
                className='w-full font-semibold transition-all duration-200 hover:scale-105'
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
