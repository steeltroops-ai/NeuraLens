/**
 * NeuroLens-X Dashboard Page
 * Full-screen dashboard with system overview and test options
 * No navbar - clean, focused interface for healthcare professionals
 */

"use client";

import React, { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Progress, CircularProgress } from "@/components/ui/Progress";
import { cn, formatProcessingTime, getRiskColor } from "@/lib/utils";

// Animation variants
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

// Test options data
const testOptions = [
  {
    id: "speech",
    title: "Speech Analysis",
    description: "Voice biomarker detection for neurological indicators",
    icon: "🎤",
    processingTime: 11.7,
    accuracy: 92.1,
    difficulty: "easy" as const,
    estimatedTime: 3,
    available: true,
    route: "/assessment/speech",
    color: "from-blue-500 to-blue-600",
  },
  {
    id: "retinal",
    title: "Retinal Screening",
    description: "Eye imaging analysis for neurological biomarkers",
    icon: "👁️",
    processingTime: 145.2,
    accuracy: 94.8,
    difficulty: "medium" as const,
    estimatedTime: 5,
    available: true,
    route: "/assessment/retinal",
    color: "from-green-500 to-green-600",
  },
  {
    id: "motor",
    title: "Motor Assessment",
    description: "Movement analysis for motor function evaluation",
    icon: "🤲",
    processingTime: 42.3,
    accuracy: 91.7,
    difficulty: "medium" as const,
    estimatedTime: 4,
    available: true,
    route: "/assessment/motor",
    color: "from-purple-500 to-purple-600",
  },
  {
    id: "cognitive",
    title: "Cognitive Evaluation",
    description: "Memory and attention assessment tasks",
    icon: "🧠",
    processingTime: 38.1,
    accuracy: 89.4,
    difficulty: "hard" as const,
    estimatedTime: 8,
    available: true,
    route: "/assessment/cognitive",
    color: "from-orange-500 to-orange-600",
  },
  {
    id: "full",
    title: "Full Assessment",
    description: "Complete multi-modal neurological evaluation",
    icon: "🔬",
    processingTime: 0.3,
    accuracy: 96.2,
    difficulty: "hard" as const,
    estimatedTime: 15,
    available: true,
    route: "/assessment",
    color: "from-primary-500 to-secondary-500",
  },
];

// System metrics
const systemMetrics = {
  totalAssessments: 1247,
  averageProcessingTime: 47.2,
  systemUptime: 99.97,
  activeUsers: 89,
  modelsLoaded: 5,
  apiLatency: 23.4,
};

// Recent activity mock data
const recentActivity = [
  {
    id: 1,
    type: "speech",
    timestamp: "2 minutes ago",
    result: "Low Risk",
    confidence: 94.2,
  },
  {
    id: 2,
    type: "retinal",
    timestamp: "5 minutes ago",
    result: "Moderate Risk",
    confidence: 87.1,
  },
  {
    id: 3,
    type: "motor",
    timestamp: "8 minutes ago",
    result: "Low Risk",
    confidence: 91.8,
  },
  {
    id: 4,
    type: "cognitive",
    timestamp: "12 minutes ago",
    result: "High Risk",
    confidence: 89.3,
  },
];

export default function DashboardPage() {
  const [selectedTest, setSelectedTest] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50/80 via-primary-50/60 to-secondary-50/80 neural-grid-primary">
      {/* Premium Header */}
      <header className="border-b border-glass-border-light glass-medium backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            {/* Premium Logo */}
            <Link href="/" className="flex items-center space-x-4 group">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-xl flex items-center justify-center shadow-neural-sm group-hover:shadow-neural-md transition-all duration-300">
                <span className="text-white font-bold text-lg">N</span>
              </div>
              <span className="text-2xl font-bold text-gradient-primary group-hover:scale-105 transition-transform duration-300">
                NeuroLens-X
              </span>
            </Link>

            {/* Premium System Status */}
            <div className="flex items-center space-x-6">
              <div className="glass-light rounded-full px-4 py-2 flex items-center space-x-3">
                <div className="w-3 h-3 bg-gradient-to-r from-green-400 to-green-500 rounded-full animate-pulse shadow-neural-sm" />
                <span className="text-neutral-700 font-medium">
                  System Operational
                </span>
              </div>
              <div className="glass-light rounded-full px-4 py-2">
                <span className="text-neutral-600 font-medium">
                  {systemMetrics.activeUsers} active users
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <motion.div
          className="mb-8"
          initial="initial"
          animate="animate"
          variants={fadeInUp}
        >
          <h1 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-2">
            Assessment Dashboard
          </h1>
          <p className="text-lg text-neutral-600">
            Select an assessment type to begin neurological risk evaluation
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Content - Test Options */}
          <div className="lg:col-span-2">
            {/* System Overview Cards */}
            <motion.div
              className="grid md:grid-cols-3 gap-4 mb-8"
              initial="initial"
              animate="animate"
              variants={staggerContainer}
            >
              <motion.div variants={fadeInUp}>
                <Card variant="glass">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-neutral-600">
                          Total Assessments
                        </p>
                        <p className="text-2xl font-bold text-primary-600">
                          {systemMetrics.totalAssessments.toLocaleString()}
                        </p>
                      </div>
                      <div className="text-2xl">📊</div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={fadeInUp}>
                <Card variant="glass">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-neutral-600">
                          Avg Processing
                        </p>
                        <p className="text-2xl font-bold text-secondary-600">
                          {formatProcessingTime(
                            systemMetrics.averageProcessingTime
                          )}
                        </p>
                      </div>
                      <div className="text-2xl">⚡</div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={fadeInUp}>
                <Card variant="glass">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-neutral-600">
                          System Uptime
                        </p>
                        <p className="text-2xl font-bold text-green-600">
                          {systemMetrics.systemUptime}%
                        </p>
                      </div>
                      <div className="text-2xl">🟢</div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Test Options Grid */}
            <motion.div
              className="grid md:grid-cols-2 gap-6"
              initial="initial"
              animate="animate"
              variants={staggerContainer}
            >
              {testOptions.map((test, index) => (
                <motion.div key={test.id} variants={fadeInUp}>
                  <Card
                    className={cn(
                      "h-full cursor-pointer transition-all duration-200",
                      selectedTest === test.id
                        ? "ring-2 ring-primary-500 shadow-elevation-3"
                        : "",
                      test.available
                        ? "hover:shadow-elevation-2 hover:scale-[1.02]"
                        : "opacity-60"
                    )}
                    variant="glass"
                    onClick={() => setSelectedTest(test.id)}
                  >
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-3">
                          <div
                            className={cn(
                              "w-12 h-12 rounded-lg flex items-center justify-center text-2xl",
                              `bg-gradient-to-br ${test.color}`
                            )}
                          >
                            <span className="filter drop-shadow-sm">
                              {test.icon}
                            </span>
                          </div>
                          <div>
                            <CardTitle className="text-lg">
                              {test.title}
                            </CardTitle>
                            <CardDescription className="text-sm">
                              {test.description}
                            </CardDescription>
                          </div>
                        </div>
                        <div className="text-right text-sm">
                          <div className="text-neutral-500">
                            ~{test.estimatedTime}min
                          </div>
                          <div
                            className={cn(
                              "text-xs px-2 py-1 rounded-full mt-1",
                              {
                                "bg-green-100 text-green-700":
                                  test.difficulty === "easy",
                                "bg-yellow-100 text-yellow-700":
                                  test.difficulty === "medium",
                                "bg-red-100 text-red-700":
                                  test.difficulty === "hard",
                              }
                            )}
                          >
                            {test.difficulty}
                          </div>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-neutral-600">Processing:</span>
                          <div className="font-semibold text-primary-600">
                            {formatProcessingTime(test.processingTime)}
                          </div>
                        </div>
                        <div>
                          <span className="text-neutral-600">Accuracy:</span>
                          <div className="font-semibold text-secondary-600">
                            {test.accuracy}%
                          </div>
                        </div>
                      </div>

                      <Progress
                        value={test.accuracy}
                        variant="neural"
                        className="mt-2"
                      />

                      <Button
                        className="w-full"
                        disabled={!test.available}
                        asChild={test.available}
                      >
                        {test.available ? (
                          <Link href={test.route}>Start {test.title}</Link>
                        ) : (
                          <span>Coming Soon</span>
                        )}
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>

          {/* Sidebar - System Info & Recent Activity */}
          <div className="space-y-6">
            {/* System Performance */}
            <motion.div initial="initial" animate="animate" variants={fadeInUp}>
              <Card variant="neural">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <span>🔧</span>
                    System Performance
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600">
                        Models Loaded
                      </span>
                      <span className="font-semibold">
                        {systemMetrics.modelsLoaded}/5
                      </span>
                    </div>
                    <Progress value={100} variant="success" size="sm" />

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600">
                        API Latency
                      </span>
                      <span className="font-semibold text-green-600">
                        {formatProcessingTime(systemMetrics.apiLatency)}
                      </span>
                    </div>
                    <Progress value={85} variant="neural" size="sm" />

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600">
                        Memory Usage
                      </span>
                      <span className="font-semibold text-blue-600">67%</span>
                    </div>
                    <Progress value={67} variant="default" size="sm" />
                  </div>

                  <div className="pt-4 border-t border-neutral-200">
                    <div className="flex items-center justify-center">
                      <CircularProgress
                        value={systemMetrics.systemUptime}
                        size={80}
                        showLabel
                        variant="neural"
                      />
                    </div>
                    <p className="text-center text-sm text-neutral-600 mt-2">
                      System Uptime
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Recent Activity */}
            <motion.div initial="initial" animate="animate" variants={fadeInUp}>
              <Card variant="glass">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <span>📈</span>
                    Recent Activity
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {recentActivity.map((activity) => (
                      <div
                        key={activity.id}
                        className="flex items-center justify-between p-3 rounded-lg bg-white/50 border border-white/20"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                            <span className="text-xs font-semibold text-primary-600">
                              {activity.type.charAt(0).toUpperCase()}
                            </span>
                          </div>
                          <div>
                            <div className="text-sm font-medium capitalize">
                              {activity.type} Test
                            </div>
                            <div className="text-xs text-neutral-500">
                              {activity.timestamp}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div
                            className={cn(
                              "text-sm font-semibold",
                              getRiskColor(
                                activity.result === "Low Risk"
                                  ? 20
                                  : activity.result === "Moderate Risk"
                                    ? 50
                                    : 80
                              )
                            )}
                          >
                            {activity.result}
                          </div>
                          <div className="text-xs text-neutral-500">
                            {activity.confidence}% confidence
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <Button variant="outline" className="w-full mt-4" size="sm">
                    View All Activity
                  </Button>
                </CardContent>
              </Card>
            </motion.div>

            {/* Quick Actions */}
            <motion.div initial="initial" animate="animate" variants={fadeInUp}>
              <Card variant="glass">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <span>⚡</span>
                    Quick Actions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    size="sm"
                    asChild
                  >
                    <Link href="/assessment">
                      <span className="mr-2">🚀</span>
                      Start Full Assessment
                    </Link>
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    size="sm"
                  >
                    <span className="mr-2">📊</span>
                    View Analytics
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    size="sm"
                  >
                    <span className="mr-2">⚙️</span>
                    System Settings
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    size="sm"
                    asChild
                  >
                    <Link href="/about">
                      <span className="mr-2">📖</span>
                      Documentation
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
