/**
 * NeuroLens-X Card Component
 * Glassmorphism-styled card with neural-grid patterns
 * WCAG 2.1 AAA compliant with proper focus management
 */

import React from "react";
import { cn } from "@/lib/utils";
import type { CardProps } from "@/types";

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      children,
      className,
      variant = "default",
      elevation = 1,
      interactive = false,
      ...props
    },
    ref
  ) => {
    const baseStyles = cn(
      // Premium base card styles
      "relative rounded-xl border transition-all duration-300 ease-out",
      "focus-within:outline-none focus-within:ring-2 focus-within:ring-primary-500 focus-within:ring-offset-2",
      "will-change-transform",

      // Variant styles with premium aesthetics
      {
        // Premium default card with sophisticated elevation
        "bg-white/95 backdrop-blur-sm border-neutral-200/60 shadow-elevation-2":
          variant === "default",

        // Enhanced glassmorphism with premium blur effects
        "glass-medium border-glass-border-medium shadow-glass-md":
          variant === "glass",

        // Premium neural-themed card with animated grid
        "bg-gradient-to-br from-primary-50/80 to-secondary-50/80 border-primary-200/60 neural-grid-primary shadow-neural-md backdrop-blur-sm":
          variant === "neural",
      },

      // Premium elevation system
      {
        "shadow-elevation-1": elevation === 1,
        "shadow-elevation-2": elevation === 2,
        "shadow-elevation-3": elevation === 3,
        "shadow-elevation-4": elevation === 4,
        "shadow-elevation-5": elevation === 5,
      },

      // Premium interactive effects
      {
        "interactive-premium cursor-pointer": interactive,
        "hover:border-primary-300/80 hover:shadow-glass-lg":
          interactive && variant === "default",
        "hover:border-glass-border-strong hover:shadow-glass-lg":
          interactive && variant === "glass",
        "hover:border-primary-300/80 hover:shadow-neural-lg":
          interactive && variant === "neural",
      },

      className
    );

    return (
      <div
        ref={ref}
        className={baseStyles}
        role={interactive ? "button" : undefined}
        tabIndex={interactive ? 0 : undefined}
        {...props}
      >
        {/* Neural grid overlay for neural variant */}
        {variant === "neural" && (
          <div className="absolute inset-0 opacity-20 pointer-events-none">
            <svg
              className="w-full h-full"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <defs>
                <pattern
                  id="neural-grid"
                  x="0"
                  y="0"
                  width="10"
                  height="10"
                  patternUnits="userSpaceOnUse"
                >
                  <circle
                    cx="5"
                    cy="5"
                    r="0.5"
                    fill="currentColor"
                    className="text-primary-600"
                  />
                  <line
                    x1="5"
                    y1="5"
                    x2="15"
                    y2="5"
                    stroke="currentColor"
                    strokeWidth="0.2"
                    className="text-primary-400"
                  />
                  <line
                    x1="5"
                    y1="5"
                    x2="5"
                    y2="15"
                    stroke="currentColor"
                    strokeWidth="0.2"
                    className="text-primary-400"
                  />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#neural-grid)" />
            </svg>
          </div>
        )}

        {children}
      </div>
    );
  }
);

Card.displayName = "Card";

// Card Header Component
const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

// Card Title Component
const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight text-neutral-900",
      "focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-sm",
      className
    )}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

// Card Description Component
const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-neutral-600 leading-relaxed", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

// Card Content Component
const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

// Card Footer Component
const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
};
