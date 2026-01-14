'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import React, { useState, useEffect } from 'react';
import {
  SignInButton,
  SignUpButton,
  SignedIn,
  SignedOut,
  UserButton,
} from '@clerk/nextjs';

import { cn } from '@/components/ui';
import { useSafeNavigation } from '@/components/SafeNavigation';

interface NavigationItem {
  id: string;
  label: string;
  href: string;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'home',
    label: 'Home',
    href: '/',
  },
  {
    id: 'dashboard',
    label: 'Dashboard',
    href: '/dashboard',
  },
  {
    id: 'about',
    label: 'About',
    href: '/about',
  },
];

export const Header: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  const { preload } = useSafeNavigation();

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
              className='text-xl font-semibold text-[#1D1D1F] transition-colors duration-200'
              aria-label='MediLens Home'
            >
              MediLens
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className='hidden lg:flex lg:items-center lg:space-x-8'>
            {navigationItems.map(item => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.id}
                  href={item.href}
                  className={cn(
                    'text-sm font-medium transition-colors duration-200',
                    isActive ? 'text-[#000000]' : 'text-[#8E8E93] hover:text-[#000000]',
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  onMouseEnter={() => preload(item.href)}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>

          {/* CTA Button */}
          <div className='hidden lg:flex lg:items-center lg:gap-4'>
            <SignedOut>
              <SignInButton mode="modal">
                <button className='text-sm font-medium text-[#3C3C43] transition-colors duration-200 hover:text-[#000000]'>
                  Sign In
                </button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button className='rounded-full bg-[#007AFF] px-5 py-2 text-sm font-medium text-white transition-all duration-200 hover:bg-[#0062CC]'>
                  Get Started
                </button>
              </SignUpButton>
            </SignedOut>
            <SignedIn>
              <Link
                href="/dashboard"
                className='rounded-full bg-[#007AFF] px-5 py-2 text-sm font-medium text-white transition-all duration-200 hover:bg-[#0062CC]'
              >
                Go to Dashboard
              </Link>
              <UserButton
                appearance={{
                  elements: {
                    avatarBox: 'w-9 h-9',
                  },
                }}
              />
            </SignedIn>
          </div>

          {/* Mobile Menu Button */}
          <div className='lg:hidden'>
            <button
              onClick={toggleMobileMenu}
              aria-expanded={isMobileMenuOpen}
              aria-controls='mobile-menu'
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
              className='flex h-10 w-10 items-center justify-center rounded-lg text-[#3C3C43] transition-colors hover:bg-[#F2F2F7]'
            >
              {isMobileMenuOpen ? (
                <svg className='h-6 w-6' fill='none' stroke='currentColor' viewBox='0 0 24 24' aria-hidden='true'>
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth={2}
                    d='M6 18L18 6M6 6l12 12'
                  />
                </svg>
              ) : (
                <svg className='h-6 w-6' fill='none' stroke='currentColor' viewBox='0 0 24 24' aria-hidden='true'>
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth={2}
                    d='M4 6h16M4 12h16M4 18h16'
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div
          id='mobile-menu'
          className={cn(
            'overflow-hidden transition-all duration-300 ease-in-out lg:hidden',
            isMobileMenuOpen ? 'max-h-[400px] pb-4 opacity-100' : 'max-h-0 opacity-0',
          )}
          aria-hidden={!isMobileMenuOpen}
        >
          <div className='space-y-1 border-t border-[#E5E5EA] pt-4'>
            {navigationItems.map(item => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.id}
                  href={item.href}
                  className={cn(
                    'block rounded-lg px-4 py-3 text-base font-medium transition-colors',
                    isActive
                      ? 'bg-[#007AFF]/10 text-[#007AFF]'
                      : 'text-[#3C3C43] hover:bg-[#F2F2F7]',
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  onMouseEnter={() => preload(item.href)}
                >
                  {item.label}
                </Link>
              );
            })}

            {/* Mobile CTA */}
            <div className='px-4 pt-4 space-y-3'>
              <SignedOut>
                <SignInButton mode="modal">
                  <button className='block w-full rounded-full border border-[#007AFF] px-5 py-3 text-center text-base font-medium text-[#007AFF] transition-colors hover:bg-[#007AFF]/5'>
                    Sign In
                  </button>
                </SignInButton>
                <SignUpButton mode="modal">
                  <button className='block w-full rounded-full bg-[#007AFF] px-5 py-3 text-center text-base font-medium text-white transition-colors hover:bg-[#0062CC]'>
                    Get Started
                  </button>
                </SignUpButton>
              </SignedOut>
              <SignedIn>
                <Link
                  href='/dashboard'
                  className='block w-full rounded-full bg-[#007AFF] px-5 py-3 text-center text-base font-medium text-white transition-colors hover:bg-[#0062CC]'
                >
                  Go to Dashboard
                </Link>
                <div className='flex items-center justify-center pt-2'>
                  <UserButton
                    appearance={{
                      elements: {
                        avatarBox: 'w-10 h-10',
                      },
                    }}
                  />
                </div>
              </SignedIn>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
