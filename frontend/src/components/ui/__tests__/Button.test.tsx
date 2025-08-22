/**
 * NeuroLens-X Button Component Tests
 * Comprehensive testing for accessibility, functionality, and visual states
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button, IconButton, ButtonGroup } from '../Button';

describe('Button Component', () => {
  describe('Basic Functionality', () => {
    it('renders with default props', () => {
      render(<Button>Click me</Button>);
      
      const button = screen.getByRole('button', { name: /click me/i });
      expect(button).toBeInTheDocument();
      expect(button).toHaveClass('bg-primary-500');
      expect(button).toHaveAttribute('type', 'button');
    });

    it('handles click events', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(<Button onClick={handleClick}>Click me</Button>);
      
      const button = screen.getByRole('button', { name: /click me/i });
      await user.click(button);
      
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('prevents click when disabled', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(
        <Button disabled onClick={handleClick}>
          Disabled button
        </Button>
      );
      
      const button = screen.getByRole('button', { name: /disabled button/i });
      expect(button).toBeDisabled();
      
      await user.click(button);
      expect(handleClick).not.toHaveBeenCalled();
    });

    it('prevents click when loading', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(
        <Button loading onClick={handleClick}>
          Loading button
        </Button>
      );
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-busy', 'true');
      
      await user.click(button);
      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  describe('Variants', () => {
    it('renders primary variant correctly', () => {
      render(<Button variant="primary">Primary</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('bg-primary-500', 'text-white');
    });

    it('renders secondary variant correctly', () => {
      render(<Button variant="secondary">Secondary</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('bg-secondary-500', 'text-white');
    });

    it('renders outline variant correctly', () => {
      render(<Button variant="outline">Outline</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('bg-transparent', 'text-primary-600', 'border-primary-300');
    });

    it('renders ghost variant correctly', () => {
      render(<Button variant="ghost">Ghost</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('bg-transparent', 'text-neutral-700');
    });
  });

  describe('Sizes', () => {
    it('renders small size correctly', () => {
      render(<Button size="sm">Small</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('text-sm', 'px-3', 'py-2', 'h-9');
    });

    it('renders medium size correctly', () => {
      render(<Button size="md">Medium</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('text-base', 'px-4', 'py-2', 'h-10');
    });

    it('renders large size correctly', () => {
      render(<Button size="lg">Large</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('text-lg', 'px-6', 'py-3', 'h-12');
    });
  });

  describe('Loading State', () => {
    it('shows loading spinner when loading', () => {
      render(<Button loading>Loading</Button>);
      
      const button = screen.getByRole('button');
      const spinner = button.querySelector('svg');
      
      expect(spinner).toBeInTheDocument();
      expect(spinner).toHaveClass('animate-spin');
      expect(button).toHaveAttribute('aria-busy', 'true');
    });

    it('hides button content when loading', () => {
      render(<Button loading>Button Text</Button>);
      
      const content = screen.getByText('Button Text');
      expect(content).toHaveClass('opacity-0');
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA attributes', () => {
      render(<Button disabled>Disabled</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-disabled', 'true');
    });

    it('supports keyboard navigation', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(<Button onClick={handleClick}>Keyboard accessible</Button>);
      
      const button = screen.getByRole('button');
      button.focus();
      
      expect(button).toHaveFocus();
      
      await user.keyboard('{Enter}');
      expect(handleClick).toHaveBeenCalledTimes(1);
      
      await user.keyboard(' ');
      expect(handleClick).toHaveBeenCalledTimes(2);
    });

    it('has minimum touch target size', () => {
      render(<Button>Touch target</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('min-h-touch', 'min-w-touch');
    });

    it('has proper focus styles', () => {
      render(<Button>Focus test</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('focus:outline-none', 'focus:ring-2', 'focus:ring-primary-500');
    });
  });

  describe('Custom Props', () => {
    it('accepts custom className', () => {
      render(<Button className="custom-class">Custom</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('custom-class');
    });

    it('accepts custom type attribute', () => {
      render(<Button type="submit">Submit</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('type', 'submit');
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLButtonElement>();
      
      render(<Button ref={ref}>Ref test</Button>);
      
      expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    });
  });
});

describe('IconButton Component', () => {
  const TestIcon = () => <span data-testid="test-icon">🔍</span>;

  it('renders with icon and aria-label', () => {
    render(
      <IconButton icon={<TestIcon />} aria-label="Search button" />
    );
    
    const button = screen.getByRole('button', { name: /search button/i });
    const icon = screen.getByTestId('test-icon');
    
    expect(button).toBeInTheDocument();
    expect(icon).toBeInTheDocument();
    expect(button).toHaveClass('aspect-square');
  });

  it('requires aria-label prop', () => {
    // This test ensures TypeScript compilation would fail without aria-label
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    
    render(
      // @ts-expect-error - Testing missing aria-label
      <IconButton icon={<TestIcon />} />
    );
    
    consoleSpy.mockRestore();
  });
});

describe('ButtonGroup Component', () => {
  it('renders horizontal group by default', () => {
    render(
      <ButtonGroup>
        <Button>First</Button>
        <Button>Second</Button>
        <Button>Third</Button>
      </ButtonGroup>
    );
    
    const group = screen.getByRole('group');
    expect(group).toHaveClass('flex-row');
    expect(group).toHaveClass('[&>button:first-child]:rounded-l-md');
    expect(group).toHaveClass('[&>button:last-child]:rounded-r-md');
  });

  it('renders vertical group when specified', () => {
    render(
      <ButtonGroup orientation="vertical">
        <Button>First</Button>
        <Button>Second</Button>
      </ButtonGroup>
    );
    
    const group = screen.getByRole('group');
    expect(group).toHaveClass('flex-col');
  });

  it('applies proper border styles for grouped buttons', () => {
    render(
      <ButtonGroup>
        <Button>First</Button>
        <Button>Second</Button>
      </ButtonGroup>
    );
    
    const group = screen.getByRole('group');
    expect(group).toHaveClass('[&>button:not(:first-child)]:border-l-0');
  });
});

describe('Button Performance', () => {
  it('does not re-render unnecessarily', () => {
    const renderSpy = jest.fn();
    
    const TestButton = React.memo(() => {
      renderSpy();
      return <Button>Performance test</Button>;
    });
    
    const { rerender } = render(<TestButton />);
    expect(renderSpy).toHaveBeenCalledTimes(1);
    
    // Re-render with same props
    rerender(<TestButton />);
    expect(renderSpy).toHaveBeenCalledTimes(1);
  });

  it('handles rapid clicks gracefully', async () => {
    const handleClick = jest.fn();
    const user = userEvent.setup();
    
    render(<Button onClick={handleClick}>Rapid click test</Button>);
    
    const button = screen.getByRole('button');
    
    // Simulate rapid clicks
    await user.click(button);
    await user.click(button);
    await user.click(button);
    
    expect(handleClick).toHaveBeenCalledTimes(3);
  });
});

describe('Button Edge Cases', () => {
  it('handles empty children gracefully', () => {
    render(<Button>{null}</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
  });

  it('handles complex children', () => {
    render(
      <Button>
        <span>Complex</span>
        <strong>Children</strong>
      </Button>
    );
    
    const button = screen.getByRole('button');
    expect(button).toHaveTextContent('ComplexChildren');
  });

  it('maintains focus after state changes', async () => {
    const { rerender } = render(<Button>Initial</Button>);
    
    const button = screen.getByRole('button');
    button.focus();
    expect(button).toHaveFocus();
    
    rerender(<Button disabled>Updated</Button>);
    // Focus should be maintained even when disabled
    expect(button).toHaveFocus();
  });
});
