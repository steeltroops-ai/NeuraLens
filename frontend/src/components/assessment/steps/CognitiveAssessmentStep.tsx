'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Clock,
  Target,
  CheckCircle,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
} from 'lucide-react';
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Progress,
  Badge,
} from '@/components/ui';

interface CognitiveTest {
  id: string;
  name: string;
  description: string;
  duration: number;
  icon: React.ReactNode;
  instructions: string[];
  type: 'memory' | 'attention' | 'executive' | 'language';
}

interface CognitiveAssessmentStepProps {
  onComplete: (data: any) => void;
  onBack: () => void;
}

const cognitiveTests: CognitiveTest[] = [
  {
    id: 'memory',
    name: 'Memory Recall',
    description: 'Test your ability to remember and recall information',
    duration: 120,
    icon: <Brain className="h-6 w-6" />,
    type: 'memory',
    instructions: [
      'You will see a sequence of words for 30 seconds',
      'Try to memorize as many as possible',
      'After a brief delay, recall the words you remember',
      'Click "Start" when you are ready',
    ],
  },
  {
    id: 'attention',
    name: 'Sustained Attention',
    description: 'Measure your ability to maintain focus over time',
    duration: 180,
    icon: <Target className="h-6 w-6" />,
    type: 'attention',
    instructions: [
      'Watch for specific target stimuli on the screen',
      'Click when you see the target appear',
      'Ignore distractors and maintain focus',
      'The test will run for 3 minutes',
    ],
  },
  {
    id: 'executive',
    name: 'Executive Function',
    description: 'Assess planning, problem-solving, and cognitive flexibility',
    duration: 240,
    icon: <CheckCircle className="h-6 w-6" />,
    type: 'executive',
    instructions: [
      'Complete a series of problem-solving tasks',
      'Rules may change during the test',
      'Adapt your strategy as needed',
      'Work as quickly and accurately as possible',
    ],
  },
  {
    id: 'language',
    name: 'Language Processing',
    description: 'Evaluate language comprehension and verbal fluency',
    duration: 150,
    icon: <AlertCircle className="h-6 w-6" />,
    type: 'language',
    instructions: [
      'Complete word association tasks',
      'Answer questions about text passages',
      'Generate words from specific categories',
      'Speak clearly and at a comfortable pace',
    ],
  },
];

export function CognitiveAssessmentStep({
  onComplete,
  onBack,
}: CognitiveAssessmentStepProps) {
  const [currentTestIndex, setCurrentTestIndex] = useState(0);
  const [isTestActive, setIsTestActive] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [showInstructions, setShowInstructions] = useState(true);
  const [isCompleted, setIsCompleted] = useState(false);

  const currentTest = cognitiveTests[currentTestIndex];
  const progress =
    ((currentTestIndex + (isTestActive ? 0.5 : 0)) / cognitiveTests.length) *
    100;

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isTestActive && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            handleTestComplete();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isTestActive, timeRemaining]);

  const startTest = () => {
    if (!currentTest) return;
    setShowInstructions(false);
    setIsTestActive(true);
    setTimeRemaining(currentTest.duration);
  };

  const handleTestComplete = () => {
    setIsTestActive(false);

    // Simulate test results (in real implementation, this would come from the actual test)
    const mockResult = {
      testId: currentTest.id,
      score: Math.random() * 0.4 + 0.6, // Random score between 0.6-1.0
      accuracy: Math.random() * 0.3 + 0.7, // Random accuracy between 0.7-1.0
      reactionTime: Math.random() * 200 + 300, // Random RT between 300-500ms
      completedAt: new Date().toISOString(),
    };

    setTestResults((prev) => ({
      ...prev,
      [currentTest.id]: mockResult,
    }));

    // Move to next test or complete assessment
    if (currentTestIndex < cognitiveTests.length - 1) {
      setTimeout(() => {
        setCurrentTestIndex((prev) => prev + 1);
        setShowInstructions(true);
      }, 2000);
    } else {
      setIsCompleted(true);
      setTimeout(() => {
        handleAssessmentComplete();
      }, 3000);
    }
  };

  const handleAssessmentComplete = () => {
    const overallScore =
      Object.values(testResults).reduce(
        (acc: number, result: any) => acc + result.score,
        0
      ) / Object.keys(testResults).length;

    const assessmentData = {
      type: 'cognitive',
      tests: testResults,
      overallScore,
      completedAt: new Date().toISOString(),
      duration: cognitiveTests.reduce((acc, test) => acc + test.duration, 0),
      biomarkers: {
        memory_score: testResults.memory?.score || 0,
        attention_score: testResults.attention?.score || 0,
        executive_score: testResults.executive?.score || 0,
        language_score: testResults.language?.score || 0,
        processing_speed:
          Object.values(testResults).reduce(
            (acc: number, result: any) => acc + 1000 / result.reactionTime,
            0
          ) / Object.keys(testResults).length,
        cognitive_flexibility: testResults.executive?.score || 0,
      },
    };

    onComplete(assessmentData);
  };

  const resetCurrentTest = () => {
    setIsTestActive(false);
    setTimeRemaining(0);
    setShowInstructions(true);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (isCompleted) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto max-w-2xl space-y-6 text-center"
      >
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-green-100">
          <CheckCircle className="h-10 w-10 text-green-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900">
          Cognitive Assessment Complete!
        </h2>
        <p className="text-gray-600">
          All cognitive tests have been completed successfully. Your results are
          being processed.
        </p>
        <div className="mt-6 grid grid-cols-2 gap-4">
          {Object.entries(testResults).map(
            ([testId, result]: [string, any]) => (
              <Card key={testId} className="p-4">
                <div className="text-sm font-medium capitalize text-gray-600">
                  {testId}
                </div>
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round(result.score * 100)}%
                </div>
              </Card>
            )
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* Progress Header */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">
            Cognitive Assessment
          </h1>
          <Badge variant="outline" className="text-sm">
            Test {currentTestIndex + 1} of {cognitiveTests.length}
          </Badge>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <AnimatePresence mode="wait">
        {showInstructions ? (
          <motion.div
            key="instructions"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  {currentTest.icon}
                  {currentTest.name}
                </CardTitle>
                <p className="text-gray-600">{currentTest.description}</p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    Duration: {Math.floor(currentTest.duration / 60)}:
                    {(currentTest.duration % 60).toString().padStart(2, '0')}
                  </div>
                  <Badge variant="secondary" className="capitalize">
                    {currentTest.type}
                  </Badge>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium text-gray-900">Instructions:</h4>
                  <ul className="space-y-2">
                    {currentTest.instructions.map((instruction, index) => (
                      <li
                        key={index}
                        className="flex items-start gap-2 text-sm text-gray-600"
                      >
                        <span className="mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs text-blue-600">
                          {index + 1}
                        </span>
                        {instruction}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex gap-3 pt-4">
                  <Button
                    onClick={startTest}
                    className="flex items-center gap-2"
                  >
                    <Play className="h-4 w-4" />
                    Start Test
                  </Button>
                  {currentTestIndex > 0 && (
                    <Button variant="outline" onClick={onBack}>
                      Back
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <motion.div
            key="test-active"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-3">
                    {currentTest.icon}
                    {currentTest.name} - In Progress
                  </CardTitle>
                  <div className="flex items-center gap-4">
                    <div className="font-mono text-2xl font-bold text-blue-600">
                      {formatTime(timeRemaining)}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={resetCurrentTest}
                      className="flex items-center gap-1"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Reset
                    </Button>
                  </div>
                </div>
                <Progress
                  value={
                    ((currentTest.duration - timeRemaining) /
                      currentTest.duration) *
                    100
                  }
                  className="h-2"
                />
              </CardHeader>
              <CardContent>
                <div className="flex min-h-[300px] items-center justify-center rounded-lg bg-gray-50">
                  <div className="space-y-4 text-center">
                    <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-blue-100">
                      {currentTest.icon}
                    </div>
                    <p className="text-gray-600">
                      {currentTest.name} is now running...
                    </p>
                    <p className="text-sm text-gray-500">
                      Follow the on-screen instructions and complete the tasks
                      as accurately as possible.
                    </p>
                    <Button
                      onClick={handleTestComplete}
                      variant="outline"
                      className="mt-4"
                    >
                      Complete Test Early
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
