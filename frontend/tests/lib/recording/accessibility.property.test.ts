/**
 * Property-Based Tests for Recording Accessibility
 * 
 * **Property 10: Accessibility State Announcements**
 * **Validates: Requirements 8.2**
 * 
 * For any state change in Recording_Manager, an aria-live announcement
 * SHALL be triggered with the corresponding state message.
 */
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
    RecordingState,
    STATE_ANNOUNCEMENTS,
    VALID_TRANSITIONS,
    getTargetState,
} from '@/lib/recording/state-manager';
import {
    BUTTON_ARIA_LABELS,
    STATE_ARIA_DESCRIPTIONS,
    getStateAnnouncement,
    getAudioLevelAriaLabel,
    getRecordingTimeAriaLabel,
    FOCUS_STYLES,
} from '@/lib/recording/accessibility';

// All possible recording states
const ALL_STATES: RecordingState[] = [
    'idle',
    'initializing',
    'recording',
    'paused',
    'completed',
    'error',
];

// Arbitrary for generating random states
const stateArb = fc.constantFrom(...ALL_STATES);

// Arbitrary for generating random audio levels (0-100)
const audioLevelArb = fc.float({ min: 0, max: 100, noNaN: true });

// Arbitrary for generating random recording times (0-120 seconds)
const recordingTimeArb = fc.integer({ min: 0, max: 120 });

// Arbitrary for generating random max recording times (30-300 seconds)
const maxRecordingTimeArb = fc.integer({ min: 30, max: 300 });

// Arbitrary for generating error messages
const errorMessageArb = fc.string({ minLength: 1, maxLength: 200 });

describe('Recording Accessibility - Property Tests', () => {
    /**
     * Feature: speech-pipeline-fix, Property 10: Accessibility State Announcements
     * 
     * For any state change in Recording_Manager, an aria-live announcement
     * SHALL be triggered with the corresponding state message.
     */
    describe('Property 10: Accessibility State Announcements', () => {
        it('should have a defined announcement for every valid state', () => {
            fc.assert(
                fc.property(stateArb, (state) => {
                    // Every state should have a corresponding announcement
                    const announcement = STATE_ANNOUNCEMENTS[state];

                    expect(announcement).toBeDefined();
                    expect(typeof announcement).toBe('string');
                    expect(announcement.length).toBeGreaterThan(0);
                }),
                { numRuns: 100 }
            );
        });

        it('should have a defined ARIA description for every valid state', () => {
            fc.assert(
                fc.property(stateArb, (state) => {
                    // Every state should have a corresponding ARIA description
                    const description = STATE_ARIA_DESCRIPTIONS[state];

                    expect(description).toBeDefined();
                    expect(typeof description).toBe('string');
                    expect(description.length).toBeGreaterThan(0);
                }),
                { numRuns: 100 }
            );
        });

        it('should return valid announcement for any state with context', () => {
            fc.assert(
                fc.property(
                    stateArb,
                    recordingTimeArb,
                    errorMessageArb,
                    (state, recordingTime, errorMessage) => {
                        const announcement = getStateAnnouncement(state, {
                            recordingTime,
                            errorMessage,
                        });

                        // Announcement should always be a non-empty string
                        expect(typeof announcement).toBe('string');
                        expect(announcement.length).toBeGreaterThan(0);

                        // For error state with error message, announcement should contain the error
                        if (state === 'error' && errorMessage) {
                            expect(announcement).toContain(errorMessage);
                        }

                        // For recording state with time, announcement should mention recording
                        if (state === 'recording') {
                            expect(announcement.toLowerCase()).toContain('recording');
                        }
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should generate unique announcements for different states', () => {
            fc.assert(
                fc.property(
                    stateArb,
                    stateArb,
                    (state1, state2) => {
                        if (state1 !== state2) {
                            const announcement1 = STATE_ANNOUNCEMENTS[state1];
                            const announcement2 = STATE_ANNOUNCEMENTS[state2];

                            // Different states should have different announcements
                            expect(announcement1).not.toBe(announcement2);
                        }
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should have announcements that are screen reader friendly', () => {
            fc.assert(
                fc.property(stateArb, (state) => {
                    const announcement = STATE_ANNOUNCEMENTS[state];

                    // Announcements should not contain HTML tags
                    expect(announcement).not.toMatch(/<[^>]*>/);

                    // Announcements should not contain special characters that confuse screen readers
                    expect(announcement).not.toMatch(/[<>{}[\]]/);

                    // Announcements should be reasonably short (under 200 chars)
                    expect(announcement.length).toBeLessThan(200);
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Audio Level ARIA Labels', () => {
        it('should return valid ARIA label for any audio level', () => {
            fc.assert(
                fc.property(audioLevelArb, (level) => {
                    const label = getAudioLevelAriaLabel(level);

                    // Label should be a non-empty string
                    expect(typeof label).toBe('string');
                    expect(label.length).toBeGreaterThan(0);

                    // Label should contain the level percentage
                    expect(label).toContain('percent');

                    // Label should contain a level description
                    expect(label.toLowerCase()).toMatch(/level|audio/i);
                }),
                { numRuns: 100 }
            );
        });

        it('should provide appropriate guidance for low audio levels', () => {
            fc.assert(
                fc.property(fc.float({ min: 0, max: Math.fround(9), noNaN: true }), (level) => {
                    const label = getAudioLevelAriaLabel(level);

                    // Low audio should include "very low" and guidance to speak louder
                    expect(label.toLowerCase()).toContain('very low');
                    expect(label.toLowerCase()).toContain('louder');
                }),
                { numRuns: 100 }
            );
        });

        it('should indicate good audio levels appropriately', () => {
            fc.assert(
                fc.property(fc.float({ min: 30, max: Math.fround(69), noNaN: true }), (level) => {
                    const label = getAudioLevelAriaLabel(level);

                    // Good audio levels should be indicated
                    expect(label.toLowerCase()).toContain('good');
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Recording Time ARIA Labels', () => {
        it('should return valid ARIA label for any recording time', () => {
            fc.assert(
                fc.property(
                    recordingTimeArb,
                    maxRecordingTimeArb,
                    (seconds, maxSeconds) => {
                        // Ensure seconds doesn't exceed maxSeconds
                        const validSeconds = Math.min(seconds, maxSeconds);
                        const label = getRecordingTimeAriaLabel(validSeconds, maxSeconds);

                        // Label should be a non-empty string
                        expect(typeof label).toBe('string');
                        expect(label.length).toBeGreaterThan(0);

                        // Label should mention recording time
                        expect(label.toLowerCase()).toContain('recording time');

                        // Label should mention remaining time
                        expect(label.toLowerCase()).toContain('remaining');
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should correctly format time with minutes and seconds', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 60, max: 120 }),
                    fc.integer({ min: 121, max: 300 }),
                    (seconds, maxSeconds) => {
                        const label = getRecordingTimeAriaLabel(seconds, maxSeconds);

                        // Should contain minute(s) for times >= 60 seconds
                        expect(label.toLowerCase()).toContain('minute');
                    }
                ),
                { numRuns: 100 }
            );
        });
    });

    describe('Button ARIA Labels', () => {
        it('should have defined labels for all button types', () => {
            const buttonTypes = [
                'startRecording',
                'stopRecording',
                'pauseRecording',
                'resumeRecording',
                'tryAgain',
                'recordAgain',
                'continueAnalysis',
                'skipStep',
                'goBack',
            ] as const;

            fc.assert(
                fc.property(fc.constantFrom(...buttonTypes), (buttonType) => {
                    const label = BUTTON_ARIA_LABELS[buttonType];

                    // Every button type should have a label
                    expect(label).toBeDefined();
                    expect(typeof label).toBe('string');
                    expect(label.length).toBeGreaterThan(0);

                    // Labels should be descriptive (more than just one word)
                    expect(label.split(' ').length).toBeGreaterThanOrEqual(2);
                }),
                { numRuns: 100 }
            );
        });

        it('should have unique labels for each button type', () => {
            const labels = Object.values(BUTTON_ARIA_LABELS);
            const uniqueLabels = new Set(labels);

            // All labels should be unique
            expect(uniqueLabels.size).toBe(labels.length);
        });
    });

    describe('Focus Styles', () => {
        it('should have defined focus styles for all element types', () => {
            const styleTypes = ['default', 'button', 'input', 'card'] as const;

            fc.assert(
                fc.property(fc.constantFrom(...styleTypes), (styleType) => {
                    const style = FOCUS_STYLES[styleType];

                    // Every style type should have a defined style
                    expect(style).toBeDefined();
                    expect(typeof style).toBe('string');
                    expect(style.length).toBeGreaterThan(0);

                    // Focus styles should contain focus-related classes
                    expect(style).toMatch(/focus/i);
                }),
                { numRuns: 100 }
            );
        });

        it('should include ring styles for visibility', () => {
            const styleTypes = ['default', 'button', 'input', 'card'] as const;

            fc.assert(
                fc.property(fc.constantFrom(...styleTypes), (styleType) => {
                    const style = FOCUS_STYLES[styleType];

                    // Focus styles should include ring for visibility
                    expect(style).toContain('ring');
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('State Transition Announcements', () => {
        it('should provide appropriate announcement for any valid state transition', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...VALID_TRANSITIONS),
                    (transition) => {
                        const { from, to, action } = transition;

                        // Both states should have announcements
                        const fromAnnouncement = STATE_ANNOUNCEMENTS[from];
                        const toAnnouncement = STATE_ANNOUNCEMENTS[to];

                        expect(fromAnnouncement).toBeDefined();
                        expect(toAnnouncement).toBeDefined();

                        // The target state announcement should be different from source
                        // (unless it's a self-transition, which we don't have)
                        if (from !== to) {
                            expect(fromAnnouncement).not.toBe(toAnnouncement);
                        }
                    }
                ),
                { numRuns: 100 }
            );
        });
    });
});
