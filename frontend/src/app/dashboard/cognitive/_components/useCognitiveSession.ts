"use client";

import { useReducer, useCallback } from "react";
import {
  CognitiveSessionState,
  SessionState,
  TaskResult,
  CognitiveResponse,
  CognitiveSessionInput,
  submitCognitiveSession,
} from "../types";

// =============================================================================
// STATE MANAGEMENT
// =============================================================================

type Action =
  | { type: "START_TEST" }
  | { type: "COMPLETE_TEST"; task: TaskResult }
  | { type: "CANCEL_TEST" }
  | { type: "SUBMIT_START" }
  | { type: "SUBMIT_SUCCESS"; response: CognitiveResponse }
  | { type: "SUBMIT_PARTIAL"; response: CognitiveResponse }
  | { type: "SUBMIT_ERROR"; error: string }
  | { type: "RESET" }
  | { type: "RETRY" };

const initialState: CognitiveSessionState = {
  state: "idle",
  sessionId: null,
  tasks: [],
  response: null,
  error: null,
  retryCount: 0,
};

function reducer(
  state: CognitiveSessionState,
  action: Action,
): CognitiveSessionState {
  switch (action.type) {
    case "START_TEST":
      return { ...state, state: "testing" };

    case "COMPLETE_TEST":
      return {
        ...state,
        state: "idle",
        tasks: [...state.tasks, action.task],
      };

    case "CANCEL_TEST":
      return { ...state, state: "idle" };

    case "SUBMIT_START":
      return {
        ...state,
        state: "submitting",
        sessionId: `sess_${Date.now()}`,
        error: null,
      };

    case "SUBMIT_SUCCESS":
      return {
        ...state,
        state: "success",
        response: action.response,
        error: null,
        retryCount: 0,
      };

    case "SUBMIT_PARTIAL":
      return {
        ...state,
        state: "partial",
        response: action.response,
        error: action.response.error_message || "Partial analysis completed",
      };

    case "SUBMIT_ERROR":
      return {
        ...state,
        state: "error",
        error: action.error,
        retryCount: state.retryCount + 1,
      };

    case "RESET":
      return { ...initialState };

    case "RETRY":
      return {
        ...state,
        state: "idle",
        error: null,
      };

    default:
      return state;
  }
}

// =============================================================================
// HOOK
// =============================================================================

export function useCognitiveSession() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const startTest = useCallback(() => {
    dispatch({ type: "START_TEST" });
  }, []);

  const completeTest = useCallback((task: TaskResult) => {
    dispatch({ type: "COMPLETE_TEST", task });
  }, []);

  const cancelTest = useCallback(() => {
    dispatch({ type: "CANCEL_TEST" });
  }, []);

  const submitSession = useCallback(async () => {
    if (state.tasks.length === 0) {
      dispatch({ type: "SUBMIT_ERROR", error: "No tasks to submit" });
      return;
    }

    dispatch({ type: "SUBMIT_START" });

    // Wait for the dispatch to update state, then use the generated sessionId
    // Note: Because dispatch is sync, we need to generate the ID here to match the reducer
    const newSessionId = `sess_${Date.now()}`;

    const payload: CognitiveSessionInput = {
      session_id: newSessionId,
      tasks: state.tasks,
      user_metadata: {
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
      },
    };

    try {
      const response = await submitCognitiveSession(payload);

      if (response.status === "success") {
        dispatch({ type: "SUBMIT_SUCCESS", response });
      } else if (response.status === "partial") {
        dispatch({ type: "SUBMIT_PARTIAL", response });
      } else {
        dispatch({
          type: "SUBMIT_ERROR",
          error: response.error_message || "Analysis failed",
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      dispatch({ type: "SUBMIT_ERROR", error: message });
    }
  }, [state.tasks]);

  const reset = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  const retry = useCallback(() => {
    dispatch({ type: "RETRY" });
  }, []);

  return {
    state,
    actions: {
      startTest,
      completeTest,
      cancelTest,
      submitSession,
      reset,
      retry,
    },
  };
}
