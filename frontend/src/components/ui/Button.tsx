/**
 * NeuroLens-X Button Component
 * Professional button with glassmorphism and neural styling
 * WCAG 2.1 AAA compliant with proper accessibility features
 */

import React from "react";
import { cn } from "@/lib/utils";
import type { ButtonProps } from "@/types";

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = "primary",
      size = "md",
      disabled = false,
      loading = false,
      onClick,
      className,
      type = "button",
      ...props
    },
    ref
  ) => {
    const baseStyles = cn(
      // Premium base button styles
      "inline-flex items-center justify-center rounded-lg font-medium transition-all duration-300 ease-out",
      "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 focus:ring-opacity-50",
      "disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none",
      "min-h-touch min-w-touch", // Accessibility: minimum touch target size
      "will-change-transform",
      "relative overflow-hidden",

      // Premium size variants with better proportions
      {
        "text-sm px-4 py-2.5 h-9 font-medium": size === "sm",
        "text-base px-6 py-3 h-11 font-medium": size === "md",
        "text-lg px-8 py-4 h-13 font-semibold": size === "lg",
      },

      // Premium variant styles with sophisticated effects
      {
        // Primary button - Premium Electric Blue with neural glow
        "bg-gradient-to-r from-primary-500 to-primary-600 text-white border border-primary-600/50 shadow-glass-md":
          variant === "primary",
        "hover:from-primary-600 hover:to-primary-700 hover:shadow-neural-md hover:scale-[1.02] hover:border-primary-500":
          variant === "primary" && !disabled,
        "active:from-primary-700 active:to-primary-800 active:scale-[0.98] active:shadow-glass-sm":
          variant === "primary" && !disabled,

        // Secondary button - Premium Teal with science precision
        "bg-gradient-to-r from-secondary-500 to-secondary-600 text-white border border-secondary-600/50 shadow-glass-md":
          variant === "secondary",
        "hover:from-secondary-600 hover:to-secondary-700 hover:shadow-neural-md hover:scale-[1.02] hover:border-secondary-500":
          variant === "secondary" && !disabled,
        "active:from-secondary-700 active:to-secondary-800 active:scale-[0.98] active:shadow-glass-sm":
          variant === "secondary" && !disabled,

        // Outline button - Premium glass border with neural hover
        "bg-white/5 backdrop-blur-sm text-primary-600 border border-primary-300/60 shadow-elevation-1":
          variant === "outline",
        "hover:bg-primary-50/80 hover:border-primary-400/80 hover:shadow-glass-sm hover:backdrop-blur-md":
          variant === "outline" && !disabled,
        "active:bg-primary-100/80 active:border-primary-500/80":
          variant === "outline" && !disabled,

        // Ghost button - Minimal with premium hover effects
        "bg-transparent text-neutral-700 border border-transparent":
          variant === "ghost",
        "hover:bg-neutral-100/80 hover:text-neutral-900 hover:backdrop-blur-sm":
          variant === "ghost" && !disabled,
        "active:bg-neutral-200/80": variant === "ghost" && !disabled,
      },

      className
    );

    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
      if (disabled || loading) {
        event.preventDefault();
        return;
      }
      onClick?.();
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
      // Enhanced keyboard accessibility
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        handleClick(event as unknown as React.MouseEvent<HTMLButtonElement>);
      }
    };

    return (
      <button
        ref={ref}
        type={type}
        className={baseStyles}
        disabled={disabled || loading}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        aria-disabled={disabled || loading}
        aria-busy={loading}
        {...props}
      >
        {/* Loading spinner */}
        {loading && (
          <svg
            className="animate-spin -ml-1 mr-2 h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}

        {/* Button content */}
        <span
          className={cn("flex items-center gap-2", { "opacity-0": loading })}
        >
          {children}
        </span>
      </button>
    );
  }
);

Button.displayName = "Button";

// Icon Button Component for minimal actions
const IconButton = React.forwardRef<
  HTMLButtonElement,
  Omit<ButtonProps, "children"> & {
    icon: React.ReactNode;
    "aria-label": string;
  }
>(({ icon, className, size = "md", ...props }, ref) => {
  const iconSizes = {
    sm: "w-8 h-8",
    md: "w-10 h-10",
    lg: "w-12 h-12",
  };

  return (
    <Button
      ref={ref}
      className={cn("p-0 aspect-square", iconSizes[size], className)}
      size={size}
      {...props}
    >
      {icon}
    </Button>
  );
});

IconButton.displayName = "IconButton";

// Button Group Component for related actions
const ButtonGroup = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    orientation?: "horizontal" | "vertical";
  }
>(({ children, className, orientation = "horizontal", ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "inline-flex",
      {
        "flex-row": orientation === "horizontal",
        "flex-col": orientation === "vertical",
      },
      "[&>button]:rounded-none",
      "[&>button:first-child]:rounded-l-md",
      "[&>button:last-child]:rounded-r-md",
      orientation === "vertical" &&
        "[&>button:first-child]:rounded-t-md [&>button:first-child]:rounded-l-none",
      orientation === "vertical" &&
        "[&>button:last-child]:rounded-b-md [&>button:last-child]:rounded-r-none",
      "[&>button:not(:first-child)]:border-l-0",
      orientation === "vertical" &&
        "[&>button:not(:first-child)]:border-l [&>button:not(:first-child)]:border-t-0",
      className
    )}
    role="group"
    {...props}
  >
    {children}
  </div>
));

ButtonGroup.displayName = "ButtonGroup";

export { Button, IconButton, ButtonGroup };
