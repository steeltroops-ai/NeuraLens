'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Card, Button } from '@/components/ui';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  override render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex min-h-screen items-center justify-center bg-surface-background p-4">
          <Card className="w-full max-w-lg p-8 text-center">
            <div className="mb-4 text-6xl">⚠️</div>
            <h1 className="mb-4 text-2xl font-bold text-text-primary">
              Something went wrong
            </h1>
            <p className="mb-6 text-text-secondary">
              We apologize for the inconvenience. An unexpected error occurred
              while loading this page.
            </p>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mb-6 text-left">
                <summary className="mb-2 cursor-pointer text-sm font-medium text-text-primary">
                  Error Details (Development)
                </summary>
                <div className="overflow-auto rounded-lg bg-surface-secondary p-4 font-mono text-xs text-text-secondary">
                  <div className="mb-2">
                    <strong>Error:</strong> {this.state.error.message}
                  </div>
                  <div className="mb-2">
                    <strong>Stack:</strong>
                    <pre className="whitespace-pre-wrap">
                      {this.state.error.stack}
                    </pre>
                  </div>
                  {this.state.errorInfo && (
                    <div>
                      <strong>Component Stack:</strong>
                      <pre className="whitespace-pre-wrap">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            )}

            <div className="space-y-3">
              <Button
                onClick={() => {
                  this.setState({
                    hasError: false,
                  });
                }}
                className="w-full"
              >
                Try Again
              </Button>

              <Button
                variant="secondary"
                onClick={() => {
                  if (typeof window !== 'undefined') {
                    window.location.href = '/';
                  }
                }}
                className="w-full"
              >
                Go Home
              </Button>
            </div>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Simple error fallback component
 */
export const ErrorFallback: React.FC<{
  error?: Error;
  resetError?: () => void;
}> = ({ error, resetError }) => (
  <div className="flex min-h-[400px] items-center justify-center p-4">
    <Card className="w-full max-w-md p-6 text-center">
      <div className="mb-4 text-4xl">😵</div>
      <h2 className="mb-2 text-xl font-semibold text-text-primary">
        Oops! Something went wrong
      </h2>
      <p className="mb-4 text-text-secondary">
        {error?.message || 'An unexpected error occurred'}
      </p>
      {resetError && (
        <Button onClick={resetError} size="sm">
          Try Again
        </Button>
      )}
    </Card>
  </div>
);
