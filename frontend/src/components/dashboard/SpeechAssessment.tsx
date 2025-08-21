'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Mic, 
  MicOff, 
  Play, 
  Square, 
  Upload,
  Activity,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Info,
  Volume2
} from 'lucide-react';

interface SpeechAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

interface SpeechResult {
  session_id: string;
  processing_time: number;
  confidence: number;
  risk_score: number;
  biomarkers: {
    fluency_score: number;
    voice_tremor: number;
    articulation_clarity: number;
    prosody_variation: number;
    speaking_rate: number;
    pause_frequency: number;
  };
  recommendations: string[];
}

export default function SpeechAssessment({ onProcessingChange }: SpeechAssessmentProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<SpeechResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setError(null);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      setError('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const analyzeAudio = async () => {
    if (!audioBlob) return;
    
    setIsAnalyzing(true);
    onProcessingChange(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      
      const response = await fetch('/api/speech/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const result: SpeechResult = await response.json();
      setResult(result);
      
    } catch (err) {
      setError('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
      onProcessingChange(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      setAudioBlob(file);
      setError(null);
    } else {
      setError('Please select a valid audio file.');
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getRiskLevel = (score: number) => {
    if (score < 0.25) return { level: 'Low', color: 'text-green-600', bg: 'bg-green-50' };
    if (score < 0.5) return { level: 'Moderate', color: 'text-yellow-600', bg: 'bg-yellow-50' };
    if (score < 0.75) return { level: 'High', color: 'text-orange-600', bg: 'bg-orange-50' };
    return { level: 'Very High', color: 'text-red-600', bg: 'bg-red-50' };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg">
            <Mic className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Speech Analysis</h1>
            <p className="text-slate-600">Analyze speech patterns for neurological indicators</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center space-x-2 text-slate-600">
            <Clock className="h-4 w-4" />
            <span>Processing Time: ~11.7ms</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <Activity className="h-4 w-4" />
            <span>Accuracy: 95%</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-600">
            <TrendingUp className="h-4 w-4" />
            <span>Real-time Analysis</span>
          </div>
        </div>
      </div>

      {/* Recording Interface */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Audio Recording</h2>
        
        <div className="space-y-4">
          {/* Recording Controls */}
          <div className="flex items-center justify-center space-x-4">
            {!isRecording ? (
              <motion.button
                onClick={startRecording}
                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Mic className="h-5 w-5" />
                <span>Start Recording</span>
              </motion.button>
            ) : (
              <motion.button
                onClick={stopRecording}
                className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Square className="h-5 w-5" />
                <span>Stop Recording</span>
              </motion.button>
            )}
            
            <div className="text-center">
              <div className="text-2xl font-mono font-bold text-slate-900">
                {formatTime(recordingTime)}
              </div>
              <div className="text-sm text-slate-500">Recording Time</div>
            </div>
          </div>

          {/* Recording Indicator */}
          {isRecording && (
            <motion.div 
              className="flex items-center justify-center space-x-2 text-red-600"
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <div className="w-3 h-3 bg-red-600 rounded-full"></div>
              <span className="font-medium">Recording in progress...</span>
            </motion.div>
          )}

          {/* File Upload Alternative */}
          <div className="border-t pt-4">
            <div className="text-center">
              <p className="text-sm text-slate-600 mb-2">Or upload an audio file</p>
              <label className="inline-flex items-center space-x-2 bg-slate-100 hover:bg-slate-200 text-slate-700 px-4 py-2 rounded-lg cursor-pointer transition-colors">
                <Upload className="h-4 w-4" />
                <span>Choose File</span>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Audio Preview */}
          {audioBlob && (
            <div className="bg-slate-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-700">Audio Ready</span>
                <span className="text-xs text-slate-500">
                  {(audioBlob.size / 1024).toFixed(1)} KB
                </span>
              </div>
              <audio controls className="w-full">
                <source src={URL.createObjectURL(audioBlob)} type="audio/wav" />
              </audio>
            </div>
          )}

          {/* Analyze Button */}
          {audioBlob && !isAnalyzing && (
            <motion.button
              onClick={analyzeAudio}
              className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white py-3 rounded-lg font-medium transition-all"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Analyze Speech
            </motion.button>
          )}

          {/* Processing Indicator */}
          {isAnalyzing && (
            <div className="flex items-center justify-center space-x-2 text-blue-600 py-3">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div>
              <span className="font-medium">Analyzing speech patterns...</span>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Risk Assessment */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">Risk Assessment</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className={`text-3xl font-bold ${getRiskLevel(result.risk_score).color} mb-2`}>
                    {(result.risk_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-slate-600">Risk Score</div>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mt-2 ${getRiskLevel(result.risk_score).color} ${getRiskLevel(result.risk_score).bg}`}>
                    {getRiskLevel(result.risk_score).level} Risk
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-2">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-slate-600">Confidence</div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600 mb-2">
                    {result.processing_time.toFixed(1)}ms
                  </div>
                  <div className="text-sm text-slate-600">Processing Time</div>
                </div>
              </div>
            </div>

            {/* Biomarkers */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">Speech Biomarkers</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <BiomarkerCard
                  title="Fluency Score"
                  value={result.biomarkers.fluency_score}
                  description="Speech flow and continuity"
                  unit=""
                />
                <BiomarkerCard
                  title="Voice Tremor"
                  value={result.biomarkers.voice_tremor}
                  description="Vocal instability indicator"
                  unit=""
                />
                <BiomarkerCard
                  title="Articulation Clarity"
                  value={result.biomarkers.articulation_clarity}
                  description="Speech precision and clarity"
                  unit=""
                />
                <BiomarkerCard
                  title="Speaking Rate"
                  value={result.biomarkers.speaking_rate}
                  description="Words per minute"
                  unit="WPM"
                />
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">Clinical Recommendations</h2>
              
              <div className="space-y-3">
                {result.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                    <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-slate-700">{recommendation}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Biomarker Card Component
function BiomarkerCard({ 
  title, 
  value, 
  description, 
  unit 
}: { 
  title: string; 
  value: number; 
  description: string; 
  unit: string;
}) {
  const getValueColor = (val: number) => {
    if (val > 0.8) return 'text-green-600';
    if (val > 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-slate-50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-slate-900">{title}</h3>
        <span className={`text-lg font-bold ${getValueColor(value)}`}>
          {unit === 'WPM' ? value.toFixed(0) : (value * 100).toFixed(1)}
          {unit === 'WPM' ? ' WPM' : unit === '' ? '%' : unit}
        </span>
      </div>
      <p className="text-xs text-slate-600">{description}</p>
    </div>
  );
}
