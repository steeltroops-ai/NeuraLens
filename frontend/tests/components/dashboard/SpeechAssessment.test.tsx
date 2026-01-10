import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SpeechAssessment from '@/components/dashboard/SpeechAssessment';

// Mock the component dependencies
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('SpeechAssessment', () => {
  const mockOnProcessingChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the speech assessment component', () => {
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    expect(screen.getByText('Speech Analysis')).toBeInTheDocument();
    expect(screen.getByText('Advanced voice analysis with Whisper-tiny integration')).toBeInTheDocument();
  });

  it('displays the recording interface by default', () => {
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    expect(screen.getByText('Voice Recording')).toBeInTheDocument();
    expect(screen.getByText('Start Recording')).toBeInTheDocument();
  });

  it('switches between recording and upload tabs', async () => {
    const user = userEvent.setup();
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Check initial state (record tab)
    expect(screen.getByText('Start Recording')).toBeInTheDocument();
    
    // Switch to upload tab
    const uploadTab = screen.getByText('Upload File');
    await user.click(uploadTab);
    
    expect(screen.getByText('Choose Audio File')).toBeInTheDocument();
  });

  it('starts recording when start button is clicked', async () => {
    const user = userEvent.setup();
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    const startButton = screen.getByText('Start Recording');
    await user.click(startButton);
    
    await waitFor(() => {
      expect(screen.getByText('Stop Recording')).toBeInTheDocument();
    });
  });

  it('handles file upload', async () => {
    const user = userEvent.setup();
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Switch to upload tab
    const uploadTab = screen.getByText('Upload File');
    await user.click(uploadTab);
    
    // Create a mock file
    const file = new File(['audio content'], 'test.wav', { type: 'audio/wav' });
    const fileInput = screen.getByLabelText(/choose audio file/i);
    
    await user.upload(fileInput, file);
    
    await waitFor(() => {
      expect(screen.getByText('test.wav')).toBeInTheDocument();
    });
  });

  it('validates file types during upload', async () => {
    const user = userEvent.setup();
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Switch to upload tab
    const uploadTab = screen.getByText('Upload File');
    await user.click(uploadTab);
    
    // Create an invalid file
    const file = new File(['invalid content'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByLabelText(/choose audio file/i);
    
    await user.upload(fileInput, file);
    
    await waitFor(() => {
      expect(screen.getByText(/please upload a valid audio file/i)).toBeInTheDocument();
    });
  });

  it('validates file size during upload', async () => {
    const user = userEvent.setup();
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Switch to upload tab
    const uploadTab = screen.getByText('Upload File');
    await user.click(uploadTab);
    
    // Create a large file (mock size)
    const file = new File(['x'.repeat(30 * 1024 * 1024)], 'large.wav', { type: 'audio/wav' });
    Object.defineProperty(file, 'size', { value: 30 * 1024 * 1024 });
    
    const fileInput = screen.getByLabelText(/choose audio file/i);
    await user.upload(fileInput, file);
    
    await waitFor(() => {
      expect(screen.getByText(/file size must be less than 25mb/i)).toBeInTheDocument();
    });
  });

  it('processes speech analysis after recording', async () => {
    const user = userEvent.setup();
    
    // Mock fetch for the API call
    global.fetch = jest.fn().mockResolvedValue({
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
    
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Start recording
    const startButton = screen.getByText('Start Recording');
    await user.click(startButton);
    
    // Stop recording
    await waitFor(() => {
      expect(screen.getByText('Stop Recording')).toBeInTheDocument();
    });
    
    const stopButton = screen.getByText('Stop Recording');
    await user.click(stopButton);
    
    // Analyze recording
    await waitFor(() => {
      expect(screen.getByText('Analyze Recording')).toBeInTheDocument();
    });
    
    const analyzeButton = screen.getByText('Analyze Recording');
    await user.click(analyzeButton);
    
    // Check that processing change callback is called
    expect(mockOnProcessingChange).toHaveBeenCalledWith(true);
    
    // Wait for analysis to complete
    await waitFor(() => {
      expect(screen.getByText('Analysis Complete')).toBeInTheDocument();
    }, { timeout: 3000 });
    
    expect(mockOnProcessingChange).toHaveBeenCalledWith(false);
  });

  it('displays analysis results correctly', async () => {
    const user = userEvent.setup();
    
    // Mock successful API response
    global.fetch = jest.fn().mockResolvedValue({
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
        transcription: 'The quick brown fox jumps over the lazy dog.',
        recommendations: ['Speech patterns appear normal', 'Continue regular exercises'],
        timestamp: new Date().toISOString(),
      }),
    });
    
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Simulate completing an analysis
    const startButton = screen.getByText('Start Recording');
    await user.click(startButton);
    
    await waitFor(() => {
      const stopButton = screen.getByText('Stop Recording');
      fireEvent.click(stopButton);
    });
    
    await waitFor(() => {
      const analyzeButton = screen.getByText('Analyze Recording');
      fireEvent.click(analyzeButton);
    });
    
    // Check analysis results
    await waitFor(() => {
      expect(screen.getByText('Analysis Complete')).toBeInTheDocument();
      expect(screen.getByText('92.0%')).toBeInTheDocument(); // Fluency score
      expect(screen.getByText('The quick brown fox jumps over the lazy dog.')).toBeInTheDocument();
      expect(screen.getByText('Speech patterns appear normal')).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('handles API errors gracefully', async () => {
    const user = userEvent.setup();
    
    // Mock failed API response
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      statusText: 'Internal Server Error',
    });
    
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Start and stop recording
    const startButton = screen.getByText('Start Recording');
    await user.click(startButton);
    
    await waitFor(() => {
      const stopButton = screen.getByText('Stop Recording');
      fireEvent.click(stopButton);
    });
    
    await waitFor(() => {
      const analyzeButton = screen.getByText('Analyze Recording');
      fireEvent.click(analyzeButton);
    });
    
    // Check error handling
    await waitFor(() => {
      expect(screen.getByText(/analysis failed/i)).toBeInTheDocument();
    }, { timeout: 3000 });
    
    expect(mockOnProcessingChange).toHaveBeenCalledWith(false);
  });

  it('resets analysis when reset button is clicked', async () => {
    const user = userEvent.setup();
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Start recording
    const startButton = screen.getByText('Start Recording');
    await user.click(startButton);
    
    // Stop recording
    await waitFor(() => {
      const stopButton = screen.getByText('Stop Recording');
      fireEvent.click(stopButton);
    });
    
    // Reset
    await waitFor(() => {
      const resetButton = screen.getByText('Record Again');
      fireEvent.click(resetButton);
    });
    
    // Check that we're back to initial state
    expect(screen.getByText('Start Recording')).toBeInTheDocument();
  });

  it('displays processing state correctly', async () => {
    const user = userEvent.setup();
    
    // Mock slow API response
    global.fetch = jest.fn().mockImplementation(() => 
      new Promise(resolve => 
        setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({
            session_id: 'test-session',
            processing_time: 87,
            confidence: 0.94,
            risk_score: 0.15,
            biomarkers: { fluency: 0.92, tremor: 0.08, articulation: 0.89, pause_patterns: 0.85 },
            transcription: 'Test',
            recommendations: [],
            timestamp: new Date().toISOString(),
          }),
        }), 1000)
      )
    );
    
    render(<SpeechAssessment onProcessingChange={mockOnProcessingChange} />);
    
    // Start and stop recording
    const startButton = screen.getByText('Start Recording');
    await user.click(startButton);
    
    await waitFor(() => {
      const stopButton = screen.getByText('Stop Recording');
      fireEvent.click(stopButton);
    });
    
    await waitFor(() => {
      const analyzeButton = screen.getByText('Analyze Recording');
      fireEvent.click(analyzeButton);
    });
    
    // Check processing state
    expect(screen.getByText(/processing audio/i)).toBeInTheDocument();
    expect(mockOnProcessingChange).toHaveBeenCalledWith(true);
  });
});
