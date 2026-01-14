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
import { Activity } from 'lucide-react';

import { cn } from '@/components/ui';
import { useSafeNavigation } from '@/components/SafeNavigation';

interface NavigationItem {
  id: string;
  label: string;
  href: string;
}

const navigationItems: NavigationItem[] = [
  { id: 'home', label: 'Home', href: '/' },
  { id: 'dashboard', label: 'Dashboard', href: '/dashboard' },
  { id: 'about', label: 'About', href: '/about' },
];

export const Header: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  const { preload } = useSafeNavigation();

  const isHomePage = pathname === '/';

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    handleScroll();
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [mounted]);

  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [pathname]);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      setIsMobileMenuOpen(false);
    }
  };

  const isTransparent = mounted && isHomePage && !isScrolled;

  return (
    <header
      className={cn(
        'fixed left-0 right-0 top-0 z-50 transition-all duration-300',
        isTransparent
          ? 'bg-transparent'
          : 'bg-black/95 backdrop-blur-md border-b border-zinc-800',
      )}
      onKeyDown={handleKeyDown}
      suppressHydrationWarning
    >
      <nav
        className="mx-auto max-w-6xl px-4 sm:px-6"
        role="navigation"
        aria-label="Main navigation"
        id="main-navigation"
      >
        <div className="flex h-14 items-center justify-between">
          {/* Brand */}
          <Link
            href="/"
            className="flex items-center gap-2.5 transition-colors duration-200"
            aria-label="MediLens Home"
          >
            <div className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800">
              <Activity className="h-4.5 w-4.5 text-red-500" strokeWidth={2} />
            </div>
            <span className={cn(
              'text-[15px] font-semibold tracking-tight transition-colors duration-200',
              isTransparent ? 'text-white' : 'text-white'
            )}>
              MediLens
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:gap-1">
            {navigationItems.map(item => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.id}
                  href={item.href}
                  className={cn(
                      'px-3 py-1.5 text-[13px] font-medium rounded-full transition-all duration-200',
                      isTransparent
                        ? isActive
                          ? 'text-white bg-white/10 backdrop-blur-sm'
                          : 'text-zinc-400 hover:text-white hover:bg-white/5'
                        : isActive
                          ? 'text-white bg-white/10'
                          : 'text-zinc-400 hover:text-white hover:bg-white/5',
                    )}
                  aria-current={isActive ? 'page' : undefined}
                  onMouseEnter={() => preload(item.href)}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>

          {/* CTA Buttons */}
          <div className="hidden md:flex md:items-center md:gap-1.5">
            <SignedOut>
              <SignInButton mode="modal">
                <button className="px-3 py-1.5 text-[13px] font-medium text-zinc-400 transition-colors duration-200 hover:text-white">
                  Sign In
                </button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button className="px-4 py-1.5 text-[13px] font-medium rounded-full bg-white text-black transition-all duration-200 hover:bg-zinc-200">
                  Get Started
                </button>
              </SignUpButton>
            </SignedOut>
            <SignedIn>
              <UserButton
                appearance={{
                  elements: {
                    avatarBox: 'w-6 h-6',
                  },
                }}
              />
            </SignedIn>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={toggleMobileMenu}
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
              className={cn(
                'flex h-8 w-8 items-center justify-center rounded transition-colors',
                isTransparent
                  ? 'text-white hover:bg-white/10'
                  : 'text-zinc-400 hover:bg-zinc-800 hover:text-white',
              )}
            >
              {isMobileMenuOpen ? (
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div
          id="mobile-menu"
          className={cn(
            'overflow-hidden transition-all duration-300 ease-in-out md:hidden',
            isMobileMenuOpen ? 'max-h-[400px] pb-4 opacity-100' : 'max-h-0 opacity-0',
          )}
          aria-hidden={!isMobileMenuOpen}
        >
          <div className={cn(
            'space-y-1 pt-3',
            isTransparent ? 'border-t border-white/10' : 'border-t border-zinc-800',
          )}>
            {navigationItems.map(item => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.id}
                  href={item.href}
                  className={cn(
                    'block rounded px-3 py-2 text-[13px] font-medium transition-colors',
                    isActive
                      ? 'bg-blue-900/50 text-blue-400'
                      : isTransparent
                        ? 'text-white/80 hover:bg-white/10 hover:text-white'
                        : 'text-zinc-400 hover:bg-zinc-800 hover:text-white',
                  )}
                  aria-current={isActive ? 'page' : undefined}
                  onMouseEnter={() => preload(item.href)}
                >
                  {item.label}
                </Link>
              );
            })}

            {/* Mobile CTA */}
            <div className="pt-3 space-y-2">
              <SignedOut>
                <SignInButton mode="modal">
                  <button className={cn(
                    'block w-full rounded border px-3 py-2 text-center text-[13px] font-medium transition-colors',
                    isTransparent
                      ? 'border-white/20 text-white hover:bg-white/10'
                      : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800',
                  )}>
                    Sign In
                  </button>
                </SignInButton>
                <SignUpButton mode="modal">
                  <button className="block w-full rounded bg-blue-600 px-3 py-2 text-center text-[13px] font-medium text-white hover:bg-blue-700">
                    Get Started
                  </button>
                </SignUpButton>
              </SignedOut>
              <SignedIn>
                <Link
                  href="/dashboard"
                  className="block w-full rounded bg-blue-600 px-3 py-2 text-center text-[13px] font-medium text-white hover:bg-blue-700"
                >
                  Go to Dashboard
                </Link>
                <div className="flex items-center justify-center pt-2">
                  <UserButton
                    appearance={{
                      elements: {
                        avatarBox: 'w-8 h-8',
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
