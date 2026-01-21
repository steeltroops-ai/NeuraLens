"""
Cognitive Feature Extractor - Production Grade
Comprehensive task-specific feature extraction for research-grade cognitive assessment.

Supported Tasks:
- Reaction Time (PVT)
- N-Back (Working Memory)
- Go/No-Go (Inhibition)
- Trail Making A & B (Executive Function)
- Digit Symbol (Processing Speed)
- Stroop (Selective Attention)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from ..schemas import TaskSession, TaskMetrics, TaskEvent, TaskCompletionStatus
from ..errors.codes import ErrorCode, PipelineError
from ..config import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract cognitive features from raw task event data.
    Research-grade implementation with quality metrics and clinical validity checks.
    """
    
    def extract(self, tasks: List[TaskSession]) -> List[TaskMetrics]:
        """Extract features from all tasks"""
        results = []
        for task in tasks:
            try:
                metrics = self._process_task(task)
                results.append(metrics)
                logger.debug(f"[EXTRACTOR] Task {task.task_id}: score={metrics.performance_score:.1f}")
            except Exception as e:
                logger.error(f"[EXTRACTOR] Failed to process task {task.task_id}: {e}")
                results.append(TaskMetrics(
                    task_id=task.task_id,
                    completion_status=TaskCompletionStatus.INVALID,
                    performance_score=0.0,
                    parameters={"error": str(e)},
                    validity_flag=False,
                    quality_warnings=["Extraction failed"]
                ))
        return results

    def _process_task(self, task: TaskSession) -> TaskMetrics:
        """Dispatch to specific task processor"""
        task_handlers = {
            "reaction_time": self._process_reaction_time,
            "n_back": self._process_nback,
            "go_no_go": self._process_go_no_go,
            "trail_making": self._process_trail_making,
            "digit_symbol": self._process_digit_symbol,
            "stroop": self._process_stroop,
        }
        
        # Find matching handler
        for key, handler in task_handlers.items():
            if key in task.task_id:
                return handler(task)
        
        # Unknown task type
        logger.warning(f"[EXTRACTOR] Unknown task type: {task.task_id}")
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.UNKNOWN,
            performance_score=0.0,
            parameters={},
            validity_flag=False,
            quality_warnings=["Unknown task type"]
        )

    # =========================================================================
    # REACTION TIME (PVT)
    # =========================================================================
    def _process_reaction_time(self, task: TaskSession) -> TaskMetrics:
        """
        Process Reaction Time Task (PVT Protocol)
        
        Metrics:
        - mean_rt: Average reaction time (ms)
        - std_rt: Standard deviation
        - valid_trials: Count of valid responses
        - error_rate: Proportion of invalid responses
        - lapses: Count of very slow responses (>500ms)
        """
        rts: List[float] = []
        last_stim_time: Optional[float] = None
        valid_trials = 0
        errors = 0
        lapses = 0
        quality_warnings = []
        
        for event in task.events:
            if event.event_type == "stimulus_shown":
                last_stim_time = event.timestamp
            elif event.event_type == "response_received":
                if last_stim_time is not None:
                    dt = event.timestamp - last_stim_time
                    if config.MIN_REACTION_TIME_MS < dt < config.MAX_REACTION_TIME_MS:
                        rts.append(dt)
                        valid_trials += 1
                        if dt > 500:
                            lapses += 1
                    else:
                        errors += 1
                    last_stim_time = None
                else:
                    errors += 1
            elif event.event_type == "response_early":
                errors += 1
        
        if not rts:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={"error_count": float(errors)},
                validity_flag=False,
                quality_warnings=["No valid reaction times recorded"]
            )
        
        mean_rt = float(np.mean(rts))
        std_rt = float(np.std(rts))
        median_rt = float(np.median(rts))
        min_rt = float(np.min(rts))
        max_rt = float(np.max(rts))
        
        # Score: Combine speed and accuracy
        speed_score = max(0, 100 - (mean_rt - 200) / 5)
        accuracy_score = 100 * (valid_trials / (valid_trials + errors)) if (valid_trials + errors) > 0 else 0
        performance_score = min(100.0, (speed_score * 0.6 + accuracy_score * 0.4))
        
        if valid_trials < 5:
            quality_warnings.append("Fewer than 5 valid trials")
        if std_rt > mean_rt * 0.5:
            quality_warnings.append("High variability in response times")
        if lapses > valid_trials * 0.3:
            quality_warnings.append("Elevated lapse rate")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                "mean_rt": mean_rt,
                "std_rt": std_rt,
                "median_rt": median_rt,
                "min_rt": min_rt,
                "max_rt": max_rt,
                "valid_trials": float(valid_trials),
                "error_count": float(errors),
                "lapse_count": float(lapses),
                "error_rate": errors / (valid_trials + errors) if (valid_trials + errors) > 0 else 1.0
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # N-BACK (WORKING MEMORY)
    # =========================================================================
    def _process_nback(self, task: TaskSession) -> TaskMetrics:
        """
        Process N-Back Task (Working Memory)
        
        Metrics:
        - accuracy: Overall accuracy
        - d_prime: Signal detection sensitivity
        - hits/misses/false_alarms/correct_rejections
        """
        hits = 0
        misses = 0
        false_alarms = 0
        correct_rejections = 0
        hit_rts: List[float] = []
        quality_warnings = []
        
        for event in task.events:
            if event.event_type == "trial_result":
                res = event.payload.get("result")
                if res == "hit":
                    hits += 1
                    if "rt" in event.payload:
                        hit_rts.append(event.payload["rt"])
                elif res == "miss":
                    misses += 1
                elif res == "false_alarm":
                    false_alarms += 1
                elif res == "correct_rejection":
                    correct_rejections += 1
        
        total_targets = hits + misses
        total_nontargets = false_alarms + correct_rejections
        total = total_targets + total_nontargets
        
        if total == 0:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={},
                validity_flag=False,
                quality_warnings=["No trial results recorded"]
            )
        
        hit_rate = hits / total_targets if total_targets > 0 else 0
        fa_rate = false_alarms / total_nontargets if total_nontargets > 0 else 0
        accuracy = (hits + correct_rejections) / total
        
        # d-prime calculation
        hr_adj = min(0.99, max(0.01, hit_rate))
        fa_adj = min(0.99, max(0.01, fa_rate))
        try:
            from scipy.stats import norm
            d_prime = norm.ppf(hr_adj) - norm.ppf(fa_adj)
        except ImportError:
            d_prime = (hit_rate - fa_rate) * 2.5  # Approximation
        
        mean_hit_rt = float(np.mean(hit_rts)) if hit_rts else 0.0
        performance_score = accuracy * 100.0
        
        if total < 10:
            quality_warnings.append("Fewer than 10 trials completed")
        if hit_rate < 0.2 and fa_rate > 0.8:
            quality_warnings.append("Response pattern suggests guessing")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                "accuracy": float(accuracy),
                "hit_rate": float(hit_rate),
                "false_alarm_rate": float(fa_rate),
                "d_prime": float(d_prime),
                "hits": float(hits),
                "misses": float(misses),
                "false_alarms": float(false_alarms),
                "correct_rejections": float(correct_rejections),
                "mean_hit_rt": mean_hit_rt
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # GO/NO-GO (INHIBITION)
    # =========================================================================
    def _process_go_no_go(self, task: TaskSession) -> TaskMetrics:
        """
        Process Go/No-Go Task (Response Inhibition)
        
        Metrics:
        - commission_errors: False alarms on No-Go
        - omission_errors: Misses on Go
        - inhibition_score: 1 - commission_rate
        """
        go_hits = 0
        commission_errors = 0
        omission_errors = 0
        correct_rejections = 0
        go_rts: List[float] = []
        quality_warnings = []
        
        for event in task.events:
            if event.event_type == "trial_result":
                res = event.payload.get("result")
                if res == "hit":
                    go_hits += 1
                    if "rt" in event.payload:
                        go_rts.append(event.payload["rt"])
                elif res == "commission":
                    commission_errors += 1
                elif res == "omission":
                    omission_errors += 1
                elif res == "correct_rejection":
                    correct_rejections += 1
        
        total_go = go_hits + omission_errors
        total_nogo = commission_errors + correct_rejections
        total = total_go + total_nogo
        
        if total == 0:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={},
                validity_flag=False,
                quality_warnings=["No trial results recorded"]
            )
        
        # Key metric: inhibition success rate
        inhibition_rate = correct_rejections / total_nogo if total_nogo > 0 else 1.0
        go_accuracy = go_hits / total_go if total_go > 0 else 0
        
        # Weighted score: inhibition is primary measure
        performance_score = (inhibition_rate * 0.6 + go_accuracy * 0.4) * 100.0
        mean_go_rt = float(np.mean(go_rts)) if go_rts else 0.0
        
        if total < 20:
            quality_warnings.append("Fewer than 20 trials completed")
        if commission_errors > correct_rejections:
            quality_warnings.append("High commission error rate suggests impulsivity")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=min(100.0, performance_score),
            parameters={
                "go_hits": float(go_hits),
                "commission_errors": float(commission_errors),
                "omission_errors": float(omission_errors),
                "correct_rejections": float(correct_rejections),
                "inhibition_rate": float(inhibition_rate),
                "go_accuracy": float(go_accuracy),
                "mean_go_rt": mean_go_rt
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # TRAIL MAKING (EXECUTIVE FUNCTION)
    # =========================================================================
    def _process_trail_making(self, task: TaskSession) -> TaskMetrics:
        """
        Process Trail Making Test (Parts A and B)
        
        Metrics:
        - completion_time_ms: Total time to complete
        - errors: Number of wrong node selections
        - path_efficiency: Optimal vs actual path
        """
        quality_warnings = []
        completion_time = 0.0
        errors = 0
        nodes_completed = 0
        
        for event in task.events:
            if event.event_type == "node_selected":
                nodes_completed += 1
                if not event.payload.get("correct", False):
                    errors += 1
            elif event.event_type == "test_end":
                completion_time = event.payload.get("completion_time_ms", 0)
        
        part = task.metadata.get("part", "A")
        expected_nodes = 15 if part == "A" else 16
        
        if nodes_completed < expected_nodes * 0.8:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={"nodes_completed": float(nodes_completed)},
                validity_flag=False,
                quality_warnings=["Test not completed"]
            )
        
        # Normative data (simplified): Part A ~30s, Part B ~60s
        expected_time = 30000 if part == "A" else 60000
        time_score = max(0, 100 - (completion_time - expected_time) / 500)
        error_penalty = errors * 5
        
        performance_score = max(0, min(100, time_score - error_penalty))
        
        if completion_time > expected_time * 2:
            quality_warnings.append("Completion time significantly above average")
        if errors > 3:
            quality_warnings.append("Multiple errors may indicate confusion")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                "completion_time_ms": completion_time,
                "errors": float(errors),
                "nodes_completed": float(nodes_completed),
                "part": part
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # DIGIT SYMBOL (PROCESSING SPEED)
    # =========================================================================
    def _process_digit_symbol(self, task: TaskSession) -> TaskMetrics:
        """
        Process Digit Symbol Substitution Test
        
        Metrics:
        - items_completed: Total items within time limit
        - accuracy: Correct responses
        - avg_rt: Average response time
        """
        quality_warnings = []
        items_completed = 0
        correct_count = 0
        rts: List[float] = []
        
        for event in task.events:
            if event.event_type == "response":
                items_completed += 1
                if event.payload.get("correct", False):
                    correct_count += 1
                if "rt" in event.payload:
                    rts.append(event.payload["rt"])
        
        if items_completed == 0:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={},
                validity_flag=False,
                quality_warnings=["No responses recorded"]
            )
        
        accuracy = correct_count / items_completed
        avg_rt = float(np.mean(rts)) if rts else 0.0
        
        # Normative: ~50 items in 90s with 90% accuracy
        items_score = min(100, (items_completed / 50) * 100)
        accuracy_score = accuracy * 100
        
        performance_score = items_score * 0.6 + accuracy_score * 0.4
        
        if items_completed < 20:
            quality_warnings.append("Fewer than 20 items completed")
        if accuracy < 0.7:
            quality_warnings.append("Accuracy below 70%")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=min(100.0, performance_score),
            parameters={
                "items_completed": float(items_completed),
                "correct_count": float(correct_count),
                "accuracy": float(accuracy),
                "avg_rt_ms": avg_rt
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # STROOP (SELECTIVE ATTENTION)
    # =========================================================================
    def _process_stroop(self, task: TaskSession) -> TaskMetrics:
        """
        Process Stroop Test
        
        Metrics:
        - stroop_effect: RT difference (incongruent - congruent)
        - accuracy: Overall accuracy
        - congruent_rt: Average RT for congruent trials
        - incongruent_rt: Average RT for incongruent trials
        """
        quality_warnings = []
        congruent_rts: List[float] = []
        incongruent_rts: List[float] = []
        correct = 0
        errors = 0
        
        for event in task.events:
            if event.event_type == "response":
                is_correct = event.payload.get("correct", False)
                is_congruent = event.payload.get("congruent", True)
                rt = event.payload.get("rt", 0)
                
                if is_correct:
                    correct += 1
                    if is_congruent:
                        congruent_rts.append(rt)
                    else:
                        incongruent_rts.append(rt)
                else:
                    errors += 1
        
        total = correct + errors
        
        if total < 10:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={},
                validity_flag=False,
                quality_warnings=["Insufficient trials"]
            )
        
        avg_congruent = float(np.mean(congruent_rts)) if congruent_rts else 0
        avg_incongruent = float(np.mean(incongruent_rts)) if incongruent_rts else 0
        stroop_effect = avg_incongruent - avg_congruent
        accuracy = correct / total
        
        # Score: Lower stroop effect and higher accuracy = better
        # Typical stroop effect: 50-150ms
        stroop_score = max(0, 100 - stroop_effect / 2)
        accuracy_score = accuracy * 100
        
        performance_score = stroop_score * 0.5 + accuracy_score * 0.5
        
        if stroop_effect > 200:
            quality_warnings.append("Very high stroop interference")
        if accuracy < 0.7:
            quality_warnings.append("Low accuracy")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=min(100.0, performance_score),
            parameters={
                "stroop_effect_ms": stroop_effect,
                "avg_congruent_rt": avg_congruent,
                "avg_incongruent_rt": avg_incongruent,
                "accuracy": float(accuracy),
                "correct_count": float(correct),
                "error_count": float(errors)
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )
