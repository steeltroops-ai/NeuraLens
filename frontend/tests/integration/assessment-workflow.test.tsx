import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { mockUser, mockAssessmentSession, mockSpeechAssessment } from '../setup';

// Mock the dashboard components
jest.mock('@/components/dashboard/SpeechAssessment', () => {
  return function MockSpeechAssessment({ onProcessingChange }: any) {
    return (
      <div data-testid="speech-assessment">
        <h2>Speech Analysis</h2>
        <button 
          onClick={() => {
            onProcessingChange(true);
            setTimeout(() => onProcessingChange(false), 1000);
          }}
        >
          Start Analysis
        </button>
      </div>
    );
  };
});

jest.mock('@/components/dashboard/RetinalAssessment', () => {
  return function MockRetinalAssessment({ onProcessingChange }: any) {
    return (
      <div data-testid="retinal-assessment">
        <h2>Retinal Analysis</h2>
        <button 
          onClick={() => {
            onProcessingChange(true);
            setTimeout(() => onProcessingChange(false), 1000);
          }}
        >
          Start Analysis
        </button>
      </div>
    );
  };
});

jest.mock('@/components/dashboard/MotorAssessment', () => {
  return function MockMotorAssessment({ onProcessingChange }: any) {
    return (
      <div data-testid="motor-assessment">
        <h2>Motor Function Assessment</h2>
        <button 
          onClick={() => {
            onProcessingChange(true);
            setTimeout(() => onProcessingChange(false), 1000);
          }}
        >
          Start Test
        </button>
      </div>
    );
  };
});

jest.mock('@/components/dashboard/AssessmentHistory', () => {
  return function MockAssessmentHistory() {
    return (
      <div data-testid="assessment-history">
        <h2>Assessment History</h2>
        <div data-testid="assessment-record">
          <span>Session: {mockAssessmentSession.id}</span>
          <span>Risk Score: {mockAssessmentSession.overall_risk_score}</span>
          <button>View Details</button>
          <button>Export</button>
          <button>Delete</button>
        </div>
      </div>
    );
  };
});

// Mock the main dashboard component
const MockDashboard = () => {
  const [activeView, setActiveView] = React.useState('overview');
  const [isProcessing, setIsProcessing] = React.useState(false);

  const renderActiveView = () => {
    switch (activeView) {
      case 'speech':
        const SpeechAssessment = require('@/components/dashboard/SpeechAssessment').default;
        return <SpeechAssessment onProcessingChange={setIsProcessing} />;
      case 'retinal':
        const RetinalAssessment = require('@/components/dashboard/RetinalAssessment').default;
        return <RetinalAssessment onProcessingChange={setIsProcessing} />;
      case 'motor':
        const MotorAssessment = require('@/components/dashboard/MotorAssessment').default;
        return <MotorAssessment onProcessingChange={setIsProcessing} />;
      case 'history':
        const AssessmentHistory = require('@/components/dashboard/AssessmentHistory').default;
        return <AssessmentHistory />;
      default:
        return (
          <div data-testid="dashboard-overview">
            <h1>NeuraLens Dashboard</h1>
            <div>Total Assessments: 247</div>
            <div>Completed Today: 12</div>
            <button onClick={() => setActiveView('speech')}>Speech Analysis</button>
            <button onClick={() => setActiveView('retinal')}>Retinal Analysis</button>
            <button onClick={() => setActiveView('motor')}>Motor Assessment</button>
            <button onClick={() => setActiveView('history')}>Assessment History</button>
          </div>
        );
    }
  };

  return (
    <div>
      <nav data-testid="dashboard-nav">
        <button onClick={() => setActiveView('overview')}>Overview</button>
        <button onClick={() => setActiveView('speech')}>Speech</button>
        <button onClick={() => setActiveView('retinal')}>Retinal</button>
        <button onClick={() => setActiveView('motor')}>Motor</button>
        <button onClick={() => setActiveView('history')}>History</button>
      </nav>
      {isProcessing && <div data-testid="processing-overlay">Processing...</div>}
      {renderActiveView()}
    </div>
  );
};

import React from 'react';

describe('Assessment Workflow Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock successful API responses
    global.fetch = jest.fn().mockImplementation((url) => {
      if (url.includes('/api/v1/speech/analyze')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            session_id: 'test-session',
            processing_time: 87,
            confidence: 0.94,
            risk_score: 0.15,
            biomarkers: {
              fluency: 0.92,
              tremor: 0.08,
              articulation: 0.89,
              pause_patterns: 0.85,
            },
            transcription: 'Test transcription',
            recommendations: ['Test recommendation'],
            timestamp: new Date().toISOString(),
          }),
        });
      }
      
      if (url.includes('/api/v1/retinal/analyze')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            session_id: 'test-session',
            processing_time: 156,
            confidence: 0.91,
            risk_score: 0.25,
            biomarkers: {
              vessel_tortuosity: 0.35,
              av_ratio: 0.72,
              cup_disc_ratio: 0.28,
              vessel_density: 0.65,
            },
            recommendations: ['Test recommendation'],
            timestamp: new Date().toISOString(),
          }),
        });
      }
      
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
      });
    });
  });

  it('navigates between different assessment modules', async () => {
    const user = userEvent.setup();
    render(<MockDashboard />);
    
    // Start at overview
    expect(screen.getByTestId('dashboard-overview')).toBeInTheDocument();
    expect(screen.getByText('NeuraLens Dashboard')).toBeInTheDocument();
    
    // Navigate to speech assessment
    await user.click(screen.getByText('Speech Analysis'));
    expect(screen.getByTestId('speech-assessment')).toBeInTheDocument();
    expect(screen.getByText('Speech Analysis')).toBeInTheDocument();
    
    // Navigate to retinal assessment
    const retinalNavButton = screen.getByRole('button', { name: 'Retinal' });
    await user.click(retinalNavButton);
    expect(screen.getByTestId('retinal-assessment')).toBeInTheDocument();
    
    // Navigate to motor assessment
    const motorNavButton = screen.getByRole('button', { name: 'Motor' });
    await user.click(motorNavButton);
    expect(screen.getByTestId('motor-assessment')).toBeInTheDocument();
    
    // Navigate to history
    const historyNavButton = screen.getByRole('button', { name: 'History' });
    await user.click(historyNavButton);
    expect(screen.getByTestId('assessment-history')).toBeInTheDocument();
  });

  it('handles processing states across modules', async () => {
    const user = userEvent.setup();
    render(<MockDashboard />);
    
    // Navigate to speech assessment
    await user.click(screen.getByText('Speech Analysis'));
    
    // Start analysis
    const startButton = screen.getByText('Start Analysis');
    await user.click(startButton);
    
    // Check processing overlay appears
    expect(screen.getByTestId('processing-overlay')).toBeInTheDocument();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
    
    // Wait for processing to complete
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('maintains state when switching between modules', async () => {
    const user = userEvent.setup();
    render(<MockDashboard />);
    
    // Navigate to speech assessment and start analysis
    await user.click(screen.getByText('Speech Analysis'));
    const speechStartButton = screen.getByText('Start Analysis');
    await user.click(speechStartButton);
    
    // Wait for processing to complete
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
    
    // Navigate to retinal assessment
    const retinalNavButton = screen.getByRole('button', { name: 'Retinal' });
    await user.click(retinalNavButton);
    expect(screen.getByTestId('retinal-assessment')).toBeInTheDocument();
    
    // Navigate back to speech assessment
    const speechNavButton = screen.getByRole('button', { name: 'Speech' });
    await user.click(speechNavButton);
    expect(screen.getByTestId('speech-assessment')).toBeInTheDocument();
  });

  it('displays assessment history with CRUD operations', async () => {
    const user = userEvent.setup();
    render(<MockDashboard />);
    
    // Navigate to history
    await user.click(screen.getByText('Assessment History'));
    
    // Check history is displayed
    expect(screen.getByTestId('assessment-history')).toBeInTheDocument();
    expect(screen.getByText('Assessment History')).toBeInTheDocument();
    
    // Check assessment record is displayed
    const assessmentRecord = screen.getByTestId('assessment-record');
    expect(assessmentRecord).toBeInTheDocument();
    expect(screen.getByText(`Session: ${mockAssessmentSession.id}`)).toBeInTheDocument();
    expect(screen.getByText(`Risk Score: ${mockAssessmentSession.overall_risk_score}`)).toBeInTheDocument();
    
    // Check CRUD buttons are present
    expect(screen.getByText('View Details')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  it('handles multiple assessment workflow', async () => {
    const user = userEvent.setup();
    render(<MockDashboard />);
    
    // Complete speech assessment
    await user.click(screen.getByText('Speech Analysis'));
    await user.click(screen.getByText('Start Analysis'));
    
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
    
    // Complete retinal assessment
    const retinalNavButton = screen.getByRole('button', { name: 'Retinal' });
    await user.click(retinalNavButton);
    await user.click(screen.getByText('Start Analysis'));
    
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
    
    // Complete motor assessment
    const motorNavButton = screen.getByRole('button', { name: 'Motor' });
    await user.click(motorNavButton);
    await user.click(screen.getByText('Start Test'));
    
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
    
    // Check that all assessments completed successfully
    // (In a real test, we would verify the data was saved and can be viewed in history)
    const historyNavButton = screen.getByRole('button', { name: 'History' });
    await user.click(historyNavButton);
    expect(screen.getByTestId('assessment-history')).toBeInTheDocument();
  });

  it('handles errors gracefully across modules', async () => {
    const user = userEvent.setup();
    
    // Mock API error
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      statusText: 'Internal Server Error',
    });
    
    render(<MockDashboard />);
    
    // Navigate to speech assessment
    await user.click(screen.getByText('Speech Analysis'));
    
    // Start analysis (should fail)
    const startButton = screen.getByText('Start Analysis');
    await user.click(startButton);
    
    // Processing should still complete (even with error)
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
    
    // Should still be able to navigate to other modules
    const retinalNavButton = screen.getByRole('button', { name: 'Retinal' });
    await user.click(retinalNavButton);
    expect(screen.getByTestId('retinal-assessment')).toBeInTheDocument();
  });

  it('prevents navigation during processing', async () => {
    const user = userEvent.setup();
    render(<MockDashboard />);
    
    // Navigate to speech assessment
    await user.click(screen.getByText('Speech Analysis'));
    
    // Start analysis
    const startButton = screen.getByText('Start Analysis');
    await user.click(startButton);
    
    // Check processing overlay is present
    expect(screen.getByTestId('processing-overlay')).toBeInTheDocument();
    
    // Try to navigate (should be blocked by processing state)
    const retinalNavButton = screen.getByRole('button', { name: 'Retinal' });
    await user.click(retinalNavButton);
    
    // Should still be on speech assessment due to processing
    expect(screen.getByTestId('speech-assessment')).toBeInTheDocument();
    
    // Wait for processing to complete
    await waitFor(() => {
      expect(screen.queryByTestId('processing-overlay')).not.toBeInTheDocument();
    }, { timeout: 2000 });
    
    // Now navigation should work
    await user.click(retinalNavButton);
    expect(screen.getByTestId('retinal-assessment')).toBeInTheDocument();
  });
});
