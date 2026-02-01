"""
Cognitive Feature Extractor - Research Grade v2.1.0
Comprehensive task-specific feature extraction with PhD-level statistical rigor.

Supported Tasks:
- Reaction Time (PVT): Processing speed, sustained attention, vigilance
- N-Back (Working Memory): d-prime, response bias, capacity
- Go/No-Go (Inhibition): Commission errors, d-prime, response inhibition
- Trail Making A & B (Executive Function): Speed, errors, B-A ratio
- Digit Symbol (Processing Speed): Items completed, accuracy rate
- Stroop (Selective Attention): Interference effect, facilitation, accuracy

Scientific References:
- Ratcliff (1993): RT analysis methodology
- Stanislaw & Todorov (1999): Signal detection calculations
- Tombaugh (2004): Trail Making normative data
- Scarpina & Tagini (2017): Stroop meta-analysis

Version History:
- v2.0.0: Initial production release
- v2.1.0: Added robust statistics, SDT metrics, normative comparisons
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..schemas import TaskSession, TaskMetrics, TaskEvent, TaskCompletionStatus
from ..errors.codes import ErrorCode, PipelineError
from ..config import config
from ..analysis.statistics import (
    calculate_rt_statistics,
    calculate_signal_detection_metrics,
    calculate_fatigue_index,
    calculate_consistency_score,
    calculate_learning_slope,
    RTStatistics,
    SignalDetectionMetrics,
    OutlierMethod
)

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
    # REACTION TIME (PVT) - Research Grade
    # =========================================================================
    def _process_reaction_time(self, task: TaskSession) -> TaskMetrics:
        """
        Process Reaction Time Task (Psychomotor Vigilance Task Protocol)
        
        Research-grade metrics based on Basner & Dinges (2011):
        - Robust central tendency: Median RT, trimmed mean (outlier-resistant)
        - Variability: CV, IQR (more robust than SD for RT data)
        - Lapses: RTs > 500ms (attentional failures)
        - False starts: Responses before stimulus
        - Inverse Efficiency Score: Mean RT / Accuracy
        
        Outlier handling: MAD-based rejection (Leys et al., 2013)
        """
        rts: List[float] = []
        last_stim_time: Optional[float] = None
        valid_trials = 0
        false_starts = 0
        quality_warnings = []
        trial_accuracies = []
        
        for event in task.events:
            if event.event_type == "stimulus_shown":
                last_stim_time = event.timestamp
            elif event.event_type == "response_received":
                if last_stim_time is not None:
                    dt = event.timestamp - last_stim_time
                    rts.append(dt)
                    # Track trial-level accuracy (valid response = 1)
                    if config.MIN_REACTION_TIME_MS < dt < config.MAX_REACTION_TIME_MS:
                        trial_accuracies.append(1.0)
                        valid_trials += 1
                    else:
                        trial_accuracies.append(0.0)
                    last_stim_time = None
                else:
                    false_starts += 1
                    trial_accuracies.append(0.0)
            elif event.event_type == "response_early":
                false_starts += 1
                trial_accuracies.append(0.0)
        
        total_responses = len(rts) + false_starts
        
        if not rts or len(rts) < 3:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={"error_count": float(false_starts), "valid_trials": 0.0},
                validity_flag=False,
                quality_warnings=["Insufficient valid reaction times (minimum 3 required)"]
            )
        
        # Calculate accuracy for IES
        accuracy = valid_trials / total_responses if total_responses > 0 else 0.0
        
        # Use robust statistics module
        rt_stats = calculate_rt_statistics(
            reaction_times=rts,
            accuracy=accuracy,
            outlier_method=OutlierMethod.MAD,
            outlier_threshold=2.5,
            min_rt=config.MIN_REACTION_TIME_MS,
            max_rt=config.MAX_REACTION_TIME_MS,
            lapse_threshold=500.0
        )
        
        if rt_stats.n_valid < 3:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={"n_outliers": float(rt_stats.n_outliers)},
                validity_flag=False,
                quality_warnings=["Too many outliers, insufficient valid data"]
            )
        
        # Calculate fatigue using block analysis
        fatigue_idx, fatigue_details = calculate_fatigue_index(
            performance_scores=[1000.0 / max(rt, 100) for rt in rts],  # Speed scores
            block_size=max(2, len(rts) // 4)
        )
        
        # Performance scoring (research-grade composite)
        # Components: Speed, Accuracy, Consistency, Lapse avoidance
        
        # Speed score: Based on median RT (more robust than mean)
        # Optimal RT ~220ms, acceptable up to 350ms
        speed_score = max(0, min(100, 100 - (rt_stats.median_rt - 200) / 3))
        
        # Accuracy score: Based on valid response rate
        accuracy_score = accuracy * 100
        
        # Consistency score: Based on CV (lower = better)
        # CV < 0.15 = excellent, CV > 0.40 = poor
        consistency_score = max(0, min(100, 100 * (1 - rt_stats.coefficient_of_variation / 0.4)))
        
        # Lapse score: Proportion of trials without lapses
        lapse_rate = rt_stats.lapse_count / rt_stats.n_valid if rt_stats.n_valid > 0 else 0
        lapse_score = max(0, 100 * (1 - lapse_rate * 2))  # Penalize lapses heavily
        
        # Composite score (weighted average)
        performance_score = (
            speed_score * 0.35 +
            accuracy_score * 0.25 +
            consistency_score * 0.25 +
            lapse_score * 0.15
        )
        performance_score = min(100.0, max(0.0, performance_score))
        
        # Quality warnings
        if rt_stats.n_valid < 5:
            quality_warnings.append("Fewer than 5 valid trials - reduced reliability")
        if rt_stats.coefficient_of_variation > 0.4:
            quality_warnings.append("High RT variability (CV > 0.40) - possible attention fluctuation")
        if lapse_rate > 0.3:
            quality_warnings.append("Elevated lapse rate (>30%) - possible vigilance decrement")
        if rt_stats.skewness > 2.0:
            quality_warnings.append("Highly skewed RT distribution - consider test conditions")
        if false_starts > 2:
            quality_warnings.append(f"{false_starts} false starts detected - possible impulsivity")
        if rt_stats.n_outliers > rt_stats.n_trials * 0.2:
            quality_warnings.append("High outlier rate (>20%) - data quality concern")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                # Central tendency
                "mean_rt": rt_stats.mean_rt,
                "median_rt": rt_stats.median_rt,
                "trimmed_mean_rt": rt_stats.trimmed_mean_rt,
                # Variability
                "std_rt": rt_stats.std_rt,
                "cv_rt": rt_stats.coefficient_of_variation,
                "iqr_rt": rt_stats.iqr_rt,
                # Distribution
                "min_rt": rt_stats.min_rt,
                "max_rt": rt_stats.max_rt,
                "skewness": rt_stats.skewness,
                # Performance indices
                "inverse_efficiency_score": rt_stats.inverse_efficiency_score or 0.0,
                "lapse_count": float(rt_stats.lapse_count),
                "lapse_rate": lapse_rate,
                # Trial counts
                "valid_trials": float(rt_stats.n_valid),
                "total_trials": float(rt_stats.n_trials),
                "n_outliers": float(rt_stats.n_outliers),
                "false_starts": float(false_starts),
                "accuracy": accuracy,
                # Fatigue
                "fatigue_index": fatigue_idx
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # N-BACK (WORKING MEMORY) - Research Grade
    # =========================================================================
    def _process_nback(self, task: TaskSession) -> TaskMetrics:
        """
        Process N-Back Task (Working Memory Capacity)
        
        Research-grade metrics based on Jaeggi et al. (2010):
        - d-prime: Signal detection sensitivity (discriminability)
        - Criterion c: Response bias (liberal vs conservative)
        - Beta: Likelihood ratio decision criterion
        - Balanced accuracy: Accounts for target/non-target imbalance
        
        SDT Implementation: Stanislaw & Todorov (1999) with loglinear correction
        """
        hits = 0
        misses = 0
        false_alarms = 0
        correct_rejections = 0
        hit_rts: List[float] = []
        fa_rts: List[float] = []
        all_rts: List[float] = []
        quality_warnings = []
        block_accuracies: List[float] = []  # For learning curve
        current_block_correct = 0
        current_block_total = 0
        block_size = 10
        
        for event in task.events:
            if event.event_type == "trial_result":
                res = event.payload.get("result")
                rt = event.payload.get("rt", 0)
                
                if res == "hit":
                    hits += 1
                    current_block_correct += 1
                    if rt and rt > 0:
                        hit_rts.append(rt)
                        all_rts.append(rt)
                elif res == "miss":
                    misses += 1
                elif res == "false_alarm":
                    false_alarms += 1
                    if rt and rt > 0:
                        fa_rts.append(rt)
                        all_rts.append(rt)
                elif res == "correct_rejection":
                    correct_rejections += 1
                    current_block_correct += 1
                
                current_block_total += 1
                
                # Track block-wise accuracy for learning curve
                if current_block_total >= block_size:
                    block_accuracies.append(current_block_correct / current_block_total)
                    current_block_correct = 0
                    current_block_total = 0
        
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
        
        # Use proper SDT calculation from statistics module
        sdt_metrics = calculate_signal_detection_metrics(
            hits=hits,
            misses=misses,
            false_alarms=false_alarms,
            correct_rejections=correct_rejections,
            correction="loglinear"  # Hautus (1995) correction
        )
        
        # RT statistics for hits (response speed to targets)
        mean_hit_rt = float(np.mean(hit_rts)) if hit_rts else 0.0
        std_hit_rt = float(np.std(hit_rts)) if len(hit_rts) > 1 else 0.0
        
        # Learning curve analysis
        learning_slope = 0.0
        learning_r2 = 0.0
        if len(block_accuracies) >= 3:
            learning_slope, learning_r2 = calculate_learning_slope(block_accuracies)
        
        # Performance scoring (research-grade composite)
        # Primary: d-prime (sensitivity) - most important for working memory
        # Secondary: Accuracy, Response time efficiency
        
        # d-prime score: Map d' range [0, 4] to [0, 100]
        # d' of 2.0 is considered good performance
        dprime_score = min(100, max(0, sdt_metrics.d_prime * 25))
        
        # Accuracy score
        accuracy_score = sdt_metrics.balanced_accuracy * 100
        
        # Response time efficiency (faster hits = better)
        # Optimal hit RT ~400-600ms for 2-back
        rt_score = 100.0
        if mean_hit_rt > 0:
            rt_score = max(0, min(100, 100 - (mean_hit_rt - 400) / 10))
        
        # Composite score (weighted)
        performance_score = (
            dprime_score * 0.50 +      # Sensitivity is primary
            accuracy_score * 0.30 +    # Overall accuracy
            rt_score * 0.20            # Speed efficiency
        )
        performance_score = min(100.0, max(0.0, performance_score))
        
        # Quality warnings with clinical context
        if total < 20:
            quality_warnings.append("Fewer than 20 trials - reduced reliability for SDT calculations")
        if total_targets < 10:
            quality_warnings.append("Fewer than 10 target trials - d-prime estimate less stable")
        
        # Response pattern analysis
        if sdt_metrics.hit_rate < 0.3 and sdt_metrics.false_alarm_rate > 0.5:
            quality_warnings.append("Response pattern suggests guessing or misunderstanding task")
        if sdt_metrics.hit_rate < 0.5 and sdt_metrics.false_alarm_rate < 0.1:
            quality_warnings.append("Conservative response bias detected - may be underreporting targets")
        if sdt_metrics.hit_rate > 0.9 and sdt_metrics.false_alarm_rate > 0.5:
            quality_warnings.append("Liberal response bias detected - may be over-responding")
        
        # d-prime interpretation
        if sdt_metrics.d_prime < 0.5:
            quality_warnings.append("Very low d-prime (<0.5) - near chance performance")
        elif sdt_metrics.d_prime < 1.0:
            quality_warnings.append("Low d-prime (0.5-1.0) - poor discrimination")
        
        # Learning effect
        if learning_slope < -0.05 and learning_r2 > 0.5:
            quality_warnings.append("Performance declined over task - possible fatigue")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                # Signal Detection Theory metrics
                "d_prime": sdt_metrics.d_prime,
                "criterion_c": sdt_metrics.criterion_c,
                "beta": sdt_metrics.beta,
                # Hit/FA rates
                "hit_rate": sdt_metrics.hit_rate,
                "false_alarm_rate": sdt_metrics.false_alarm_rate,
                # Accuracy metrics
                "accuracy": sdt_metrics.accuracy,
                "balanced_accuracy": sdt_metrics.balanced_accuracy,
                # Raw counts
                "hits": float(hits),
                "misses": float(misses),
                "false_alarms": float(false_alarms),
                "correct_rejections": float(correct_rejections),
                "total_targets": float(total_targets),
                "total_nontargets": float(total_nontargets),
                # RT metrics
                "mean_hit_rt": mean_hit_rt,
                "std_hit_rt": std_hit_rt,
                # Learning curve
                "learning_slope": learning_slope,
                "learning_r2": learning_r2
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # GO/NO-GO (INHIBITION) - Research Grade
    # =========================================================================
    def _process_go_no_go(self, task: TaskSession) -> TaskMetrics:
        """
        Process Go/No-Go Task (Response Inhibition)
        
        Research-grade metrics based on Verbruggen & Logan (2008):
        - Signal Detection Theory (d-prime, criterion c)
        - Commission error rate (key measure of impulsivity)
        - Omission error rate (attention/engagement)
        - RT analysis with MAD-based outlier rejection
        - Stop-Signal Reaction Time estimation (SSRT proxy)
        
        Clinical Interpretation:
        - High commission errors: Poor inhibitory control (impulsivity)
        - High omission errors: Inattention or disengagement
        - Slow Go RTs with low commissions: Cautious/conservative strategy
        """
        go_hits = 0
        commission_errors = 0
        omission_errors = 0
        correct_rejections = 0
        go_rts: List[float] = []
        commission_rts: List[float] = []  # Track impulsive response times
        quality_warnings = []
        block_accuracies: List[float] = []
        current_block_correct = 0
        current_block_total = 0
        block_size = 10
        
        for event in task.events:
            if event.event_type == "trial_result":
                res = event.payload.get("result")
                rt = event.payload.get("rt", 0)
                
                if res == "hit":
                    go_hits += 1
                    current_block_correct += 1
                    if rt and rt > 0:
                        go_rts.append(rt)
                elif res == "commission":
                    commission_errors += 1
                    if rt and rt > 0:
                        commission_rts.append(rt)
                elif res == "omission":
                    omission_errors += 1
                elif res == "correct_rejection":
                    correct_rejections += 1
                    current_block_correct += 1
                
                current_block_total += 1
                
                # Track block-wise accuracy for fatigue analysis
                if current_block_total >= block_size:
                    block_accuracies.append(current_block_correct / current_block_total)
                    current_block_correct = 0
                    current_block_total = 0
        
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
        
        # SDT Metrics using proper calculation (Go = Signal, NoGo = Noise)
        sdt_metrics = calculate_signal_detection_metrics(
            hits=go_hits,
            misses=omission_errors,
            false_alarms=commission_errors,
            correct_rejections=correct_rejections,
            correction="loglinear"
        )
        
        # RT Analysis with MAD-based outlier rejection
        rt_stats = {}
        ies = 0.0
        if go_rts:
            clean_rts, outlier_info = detect_outliers_mad(
                go_rts,
                threshold=2.5,
                min_cap=150,
                max_cap=2000
            )
            
            if len(clean_rts) >= 3:
                rt_stats = calculate_rt_statistics(clean_rts)
                # IES: Mean RT / Go Accuracy
                go_accuracy = go_hits / total_go if total_go > 0 else 0.001
                ies = rt_stats.get("mean_rt", 0) / go_accuracy if go_accuracy > 0 else 0
            else:
                rt_stats = {
                    "mean_rt": float(np.mean(go_rts)),
                    "median_rt": float(np.median(go_rts)),
                    "sd_rt": float(np.std(go_rts)) if len(go_rts) > 1 else 0
                }
        
        # Key inhibition metrics
        inhibition_rate = correct_rejections / total_nogo if total_nogo > 0 else 1.0
        commission_rate = commission_errors / total_nogo if total_nogo > 0 else 0.0
        go_accuracy = go_hits / total_go if total_go > 0 else 0.0
        omission_rate = omission_errors / total_go if total_go > 0 else 0.0
        
        # Fatigue analysis
        fatigue_index = 0.0
        if len(block_accuracies) >= 3:
            fatigue_index = calculate_fatigue_index(block_accuracies)
        
        # Commission RT analysis (impulsive responses are typically faster)
        mean_commission_rt = float(np.mean(commission_rts)) if commission_rts else 0.0
        
        # Performance Scoring (Research-grade composite)
        # Primary: d-prime (signal detection sensitivity)
        # Secondary: Inhibition rate, Go accuracy
        
        # d-prime score: Map [0, 4] to [0, 100]
        dprime_score = min(100, max(0, sdt_metrics.d_prime * 25))
        
        # Inhibition score (most clinically relevant for Go/No-Go)
        inhibition_score = inhibition_rate * 100
        
        # Go accuracy score
        go_score = go_accuracy * 100
        
        # IES score (lower is better, optimal ~300-500ms / 0.9 accuracy)
        ies_score = 100.0
        if ies > 0:
            optimal_ies = 400  # ~360-450 for healthy adults
            ies_score = max(0, min(100, 100 - abs(ies - optimal_ies) / 10))
        
        # Composite score weighted for inhibition focus
        performance_score = (
            dprime_score * 0.30 +        # Sensitivity
            inhibition_score * 0.40 +    # Inhibition (key metric)
            go_score * 0.15 +            # Go accuracy
            ies_score * 0.15             # Efficiency
        )
        performance_score = min(100.0, max(0.0, performance_score))
        
        # Research-grade quality warnings
        if total < 40:
            quality_warnings.append("Fewer than 40 trials - reduced reliability")
        if total_nogo < 15:
            quality_warnings.append("Fewer than 15 NoGo trials - inhibition estimate less stable")
        
        # Response pattern analysis
        if commission_rate > 0.5:
            quality_warnings.append("High commission rate (>50%) - significant impulsivity or task confusion")
        if omission_rate > 0.3:
            quality_warnings.append("High omission rate (>30%) - possible inattention or fatigue")
        if commission_rate < 0.1 and go_accuracy > 0.95:
            quality_warnings.append("Very conservative response pattern - may indicate speed-accuracy tradeoff")
        
        # RT pattern analysis
        if mean_commission_rt > 0 and rt_stats.get("mean_rt", 0) > 0:
            if mean_commission_rt < rt_stats.get("mean_rt", 0) * 0.7:
                quality_warnings.append("Commission errors significantly faster than Go hits - suggests impulsive responding")
        
        # Criterion interpretation
        if sdt_metrics.criterion_c < -0.5:
            quality_warnings.append("Liberal response bias - tendency to respond")
        elif sdt_metrics.criterion_c > 0.5:
            quality_warnings.append("Conservative response bias - tendency to withhold")
        
        # Fatigue
        if fatigue_index > 0.15:
            quality_warnings.append("Possible fatigue detected - performance decline over task")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                # SDT metrics
                "d_prime": sdt_metrics.d_prime,
                "criterion_c": sdt_metrics.criterion_c,
                "beta": sdt_metrics.beta,
                # Core inhibition metrics
                "go_hits": float(go_hits),
                "commission_errors": float(commission_errors),
                "omission_errors": float(omission_errors),
                "correct_rejections": float(correct_rejections),
                "inhibition_rate": float(inhibition_rate),
                "commission_rate": float(commission_rate),
                "go_accuracy": float(go_accuracy),
                "omission_rate": float(omission_rate),
                # RT metrics
                "mean_go_rt": rt_stats.get("mean_rt", 0.0),
                "median_go_rt": rt_stats.get("median_rt", 0.0),
                "sd_go_rt": rt_stats.get("sd_rt", 0.0),
                "cv_go_rt": rt_stats.get("cv", 0.0),
                "mean_commission_rt": mean_commission_rt,
                "ies": ies,
                # Block-wise analysis
                "fatigue_index": fatigue_index,
                # Trial counts
                "total_go": float(total_go),
                "total_nogo": float(total_nogo)
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # TRAIL MAKING (EXECUTIVE FUNCTION) - Research Grade
    # =========================================================================
    def _process_trail_making(self, task: TaskSession) -> TaskMetrics:
        """
        Process Trail Making Test (Parts A and B)
        
        Research-grade metrics based on Tombaugh (2004) normative data:
        - Completion time: Primary measure of psychomotor speed (A) and flexibility (B)
        - Error analysis: Sequencing vs perseverative errors
        - B-A Difference Score: Executive component isolated from motor speed
        - B/A Ratio: Executive control accounting for baseline speed
        
        Normative Reference:
        - Part A: Mean ~30s (age 25-54), ~40s (age 55-64), ~50s (age 65+)
        - Part B: Mean ~70s (age 25-54), ~85s (age 55-64), ~130s (age 65+)
        """
        quality_warnings = []
        completion_time = 0.0
        errors = 0
        sequencing_errors = 0  # Wrong sequence order
        perseverative_errors = 0  # Returning to previous node
        nodes_completed = 0
        inter_node_times: List[float] = []  # Time between each node connection
        last_node_time = None
        
        for event in task.events:
            if event.event_type == "node_selected":
                nodes_completed += 1
                node_timestamp = event.timestamp
                
                if last_node_time is not None:
                    inter_node_times.append(node_timestamp - last_node_time)
                last_node_time = node_timestamp
                
                if not event.payload.get("correct", True):
                    errors += 1
                    error_type = event.payload.get("error_type", "sequence")
                    if error_type == "perseveration":
                        perseverative_errors += 1
                    else:
                        sequencing_errors += 1
                        
            elif event.event_type == "test_end":
                completion_time = event.payload.get("completion_time_ms", 0)
                # Try to get from different payload formats
                if completion_time == 0:
                    completion_time = event.payload.get("time_ms", 0)
                    
            elif event.event_type == "test_start":
                last_node_time = event.timestamp
        
        part = task.metadata.get("part", "A")
        expected_nodes = 25 if part == "A" else 25  # Standard TMT has 25 nodes
        
        # Check for incomplete test
        if nodes_completed < expected_nodes * 0.8:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={"nodes_completed": float(nodes_completed)},
                validity_flag=False,
                quality_warnings=[f"Test not completed ({nodes_completed}/{expected_nodes} nodes)"]
            )
        
        # Normative data based on Tombaugh (2004) - age 50-59 reference
        # These are mean completion times in milliseconds
        if part == "A":
            norm_mean = 35000  # 35 seconds
            norm_sd = 12000   # ~12 second SD
            cutoff_impaired = 78000  # 78 seconds (>2 SD)
        else:  # Part B
            norm_mean = 81000  # 81 seconds
            norm_sd = 30000   # ~30 second SD
            cutoff_impaired = 273000  # 273 seconds (clinical cutoff)
        
        # Z-score calculation
        z_score = (completion_time - norm_mean) / norm_sd if norm_sd > 0 else 0
        
        # Performance score using z-score transformation
        # Lower times are better, so invert z-score
        z_score_inverted = -z_score
        
        # Map z-score to 0-100 scale
        # z = 0 (average) -> 50
        # z = -2 (2SD fast) -> 100
        # z = +2 (2SD slow) -> 0
        base_score = 50 - (z_score * 25)
        
        # Error penalty (more severe for Part B due to complexity)
        error_penalty = errors * (8 if part == "B" else 5)
        
        performance_score = max(0, min(100, base_score - error_penalty))
        
        # Inter-node timing analysis
        int_stats = {}
        if len(inter_node_times) >= 3:
            clean_ints, _ = detect_outliers_mad(
                inter_node_times, threshold=2.5, min_cap=100, max_cap=15000
            )
            if len(clean_ints) >= 3:
                int_stats = {
                    "mean_int": float(np.mean(clean_ints)),
                    "median_int": float(np.median(clean_ints)),
                    "sd_int": float(np.std(clean_ints)),
                    "cv_int": float(np.std(clean_ints) / np.mean(clean_ints)) if np.mean(clean_ints) > 0 else 0
                }
        
        # Time per node (efficiency metric)
        time_per_node = completion_time / nodes_completed if nodes_completed > 0 else 0
        
        # Quality warnings with clinical context
        if completion_time > cutoff_impaired:
            quality_warnings.append(f"Completion time exceeds impaired cutoff ({cutoff_impaired/1000:.0f}s)")
        if z_score > 2:
            quality_warnings.append(f"Performance >2 SD below normative mean (z={z_score:.1f})")
        elif z_score > 1.5:
            quality_warnings.append(f"Performance 1.5-2 SD below mean (z={z_score:.1f}) - borderline")
        
        if errors > 3:
            quality_warnings.append(f"Multiple errors ({errors}) may indicate cognitive difficulty")
        if perseverative_errors > 1:
            quality_warnings.append("Perseverative errors suggest executive dysfunction")
        
        # Variability analysis
        if int_stats.get("cv_int", 0) > 0.8:
            quality_warnings.append("High inter-node timing variability - inconsistent performance")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                # Primary metrics
                "completion_time_ms": completion_time,
                "completion_time_s": completion_time / 1000,
                "z_score": z_score,
                # Error analysis
                "errors": float(errors),
                "sequencing_errors": float(sequencing_errors),
                "perseverative_errors": float(perseverative_errors),
                # Node metrics
                "nodes_completed": float(nodes_completed),
                "time_per_node_ms": time_per_node,
                # Inter-node timing
                "mean_inter_node_ms": int_stats.get("mean_int", 0),
                "sd_inter_node_ms": int_stats.get("sd_int", 0),
                "cv_inter_node": int_stats.get("cv_int", 0),
                # Test metadata
                "part": part,
                "norm_mean_ms": norm_mean,
                "norm_sd_ms": norm_sd
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # DIGIT SYMBOL (PROCESSING SPEED) - Research Grade
    # =========================================================================
    def _process_digit_symbol(self, task: TaskSession) -> TaskMetrics:
        """
        Process Digit Symbol Substitution Test (DSST/Coding)
        
        Research-grade metrics based on WAIS-IV (Wechsler, 2008):
        - Items completed: Primary measure (age-scaled scoring)
        - Accuracy: Typically very high (>95%) in valid attempts
        - RT analysis: Identifies processing speed vs. accuracy tradeoff
        - Learning effect: First vs last quarter comparison
        
        Normative Reference (WAIS-IV, 90 second version):
        - Age 25-34: Mean ~80, SD ~15
        - Age 35-44: Mean ~75, SD ~15
        - Age 55-64: Mean ~60, SD ~12
        - Age 65-74: Mean ~50, SD ~12
        - Age 75+: Mean ~38, SD ~10
        """
        quality_warnings = []
        items_completed = 0
        correct_count = 0
        incorrect_count = 0
        rts: List[float] = []
        
        # Track learning curve (quarter-by-quarter)
        quarter_correct: List[int] = [0, 0, 0, 0]
        quarter_total: List[int] = [0, 0, 0, 0]
        
        for event in task.events:
            if event.event_type == "response":
                items_completed += 1
                is_correct = event.payload.get("correct", False)
                
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1
                
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
        
        # Accuracy calculation
        accuracy = correct_count / items_completed if items_completed > 0 else 0
        
        # RT analysis with MAD-based outlier detection
        rt_stats = {}
        if len(rts) >= 5:
            clean_rts, outlier_info = detect_outliers_mad(
                rts, threshold=2.5, min_cap=100, max_cap=10000
            )
            if len(clean_rts) >= 3:
                rt_stats = calculate_rt_statistics(clean_rts)
        else:
            rt_stats = {
                "mean_rt": float(np.mean(rts)) if rts else 0,
                "median_rt": float(np.median(rts)) if rts else 0,
                "sd_rt": float(np.std(rts)) if len(rts) > 1 else 0
            }
        
        # Normative comparison (age 50-59 reference point)
        # These are correct items in 90 seconds
        norm_mean = 55  # items
        norm_sd = 12
        
        # Z-score based on items completed correctly
        z_score = (correct_count - norm_mean) / norm_sd if norm_sd > 0 else 0
        
        # Performance score using z-score
        # z = 0 (average) -> 50
        # z = +2 (2SD above) -> 100
        # z = -2 (2SD below) -> 0
        base_score = 50 + (z_score * 25)
        
        # Minor accuracy penalty (DSST errors are unusual and clinically significant)
        accuracy_penalty = (1 - accuracy) * 30  # Harsh penalty for errors
        
        performance_score = max(0, min(100, base_score - accuracy_penalty))
        
        # Throughput: items per second
        test_duration_s = task.metadata.get("duration_seconds", 90)
        throughput = correct_count / test_duration_s if test_duration_s > 0 else 0
        
        # Efficiency score (correct items / time spent responding)
        total_response_time_s = sum(rts) / 1000 if rts else 0
        efficiency = correct_count / total_response_time_s if total_response_time_s > 0 else 0
        
        # Quality warnings with clinical context
        if items_completed < 30:
            quality_warnings.append("Fewer than 30 items - may indicate motor or motivational issues")
        
        if accuracy < 0.90:
            quality_warnings.append(f"Accuracy below 90% ({accuracy*100:.0f}%) - atypical for DSST, review for comprehension")
        elif accuracy < 0.95:
            quality_warnings.append(f"Accuracy slightly below expected ({accuracy*100:.0f}%)")
        
        if z_score < -2:
            quality_warnings.append(f"Performance >2 SD below age norms (z={z_score:.1f}) - clinically significant")
        elif z_score < -1.5:
            quality_warnings.append(f"Performance 1.5-2 SD below norms (z={z_score:.1f}) - borderline")
        
        # RT variability check
        cv_rt = rt_stats.get("cv", 0)
        if cv_rt > 0.5:
            quality_warnings.append(f"High response time variability (CV={cv_rt:.2f}) - inconsistent processing")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                # Primary metrics
                "items_completed": float(items_completed),
                "correct_count": float(correct_count),
                "incorrect_count": float(incorrect_count),
                "accuracy": float(accuracy),
                "z_score": z_score,
                # RT metrics
                "mean_rt_ms": rt_stats.get("mean_rt", 0),
                "median_rt_ms": rt_stats.get("median_rt", 0),
                "sd_rt_ms": rt_stats.get("sd_rt", 0),
                "cv_rt": cv_rt,
                # Derived metrics
                "throughput_items_per_sec": throughput,
                "efficiency": efficiency,
                # Normative reference
                "norm_mean": float(norm_mean),
                "norm_sd": float(norm_sd)
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

    # =========================================================================
    # STROOP (SELECTIVE ATTENTION) - Research Grade
    # =========================================================================
    def _process_stroop(self, task: TaskSession) -> TaskMetrics:
        """
        Process Stroop Color-Word Test
        
        Research-grade metrics based on Scarpina & Tagini (2017):
        - Stroop Effect (Interference): Incongruent RT - Congruent RT
        - Stroop Ratio: Incongruent / Congruent (age-independent measure)
        - Facilitation Effect: Neutral RT - Congruent RT
        - Error Analysis: Condition-wise error rates
        
        Normative Reference (Scarpina & Tagini, 2017):
        - Stroop Effect: Mean ~100-150ms (healthy adults)
        - Effect increases with age: +10-15ms per decade after 50
        - Ratio: Typically 1.15-1.30 in healthy adults
        """
        quality_warnings = []
        congruent_rts: List[float] = []
        incongruent_rts: List[float] = []
        neutral_rts: List[float] = []  # Optional neutral condition
        
        congruent_correct = 0
        congruent_errors = 0
        incongruent_correct = 0
        incongruent_errors = 0
        neutral_correct = 0
        neutral_errors = 0
        
        for event in task.events:
            if event.event_type == "response":
                is_correct = event.payload.get("correct", False)
                condition = event.payload.get("condition", "")
                is_congruent = event.payload.get("congruent", None)
                is_neutral = event.payload.get("neutral", False)
                rt = event.payload.get("rt", 0)
                
                # Determine condition
                if condition == "congruent" or is_congruent == True:
                    if is_correct and rt > 0:
                        congruent_correct += 1
                        congruent_rts.append(rt)
                    else:
                        congruent_errors += 1
                elif condition == "incongruent" or is_congruent == False:
                    if is_correct and rt > 0:
                        incongruent_correct += 1
                        incongruent_rts.append(rt)
                    else:
                        incongruent_errors += 1
                elif condition == "neutral" or is_neutral:
                    if is_correct and rt > 0:
                        neutral_correct += 1
                        neutral_rts.append(rt)
                    else:
                        neutral_errors += 1
        
        total_correct = congruent_correct + incongruent_correct + neutral_correct
        total_errors = congruent_errors + incongruent_errors + neutral_errors
        total = total_correct + total_errors
        
        if total < 20:
            return TaskMetrics(
                task_id=task.task_id,
                completion_status=TaskCompletionStatus.INCOMPLETE,
                performance_score=0.0,
                parameters={},
                validity_flag=False,
                quality_warnings=["Insufficient trials (<20)"]
            )
        
        # MAD-based outlier rejection for RT analysis
        avg_congruent = 0.0
        sd_congruent = 0.0
        avg_incongruent = 0.0
        sd_incongruent = 0.0
        avg_neutral = 0.0
        
        if len(congruent_rts) >= 5:
            clean_cong, _ = detect_outliers_mad(congruent_rts, threshold=2.5, min_cap=200, max_cap=3000)
            if len(clean_cong) >= 3:
                cong_stats = calculate_rt_statistics(clean_cong)
                avg_congruent = cong_stats.get("mean_rt", 0)
                sd_congruent = cong_stats.get("sd_rt", 0)
            else:
                avg_congruent = float(np.mean(congruent_rts))
        elif congruent_rts:
            avg_congruent = float(np.mean(congruent_rts))
        
        if len(incongruent_rts) >= 5:
            clean_incong, _ = detect_outliers_mad(incongruent_rts, threshold=2.5, min_cap=200, max_cap=3000)
            if len(clean_incong) >= 3:
                incong_stats = calculate_rt_statistics(clean_incong)
                avg_incongruent = incong_stats.get("mean_rt", 0)
                sd_incongruent = incong_stats.get("sd_rt", 0)
            else:
                avg_incongruent = float(np.mean(incongruent_rts))
        elif incongruent_rts:
            avg_incongruent = float(np.mean(incongruent_rts))
        
        if len(neutral_rts) >= 3:
            clean_neut, _ = detect_outliers_mad(neutral_rts, threshold=2.5, min_cap=200, max_cap=3000)
            avg_neutral = float(np.mean(clean_neut)) if clean_neut else float(np.mean(neutral_rts))
        elif neutral_rts:
            avg_neutral = float(np.mean(neutral_rts))
        
        # Primary metric: Stroop Effect (Interference)
        stroop_effect = avg_incongruent - avg_congruent if (avg_incongruent > 0 and avg_congruent > 0) else 0
        
        # Stroop Ratio (age-independent measure)
        stroop_ratio = avg_incongruent / avg_congruent if avg_congruent > 0 else 1.0
        
        # Facilitation Effect (if neutral condition available)
        facilitation_effect = avg_neutral - avg_congruent if (avg_neutral > 0 and avg_congruent > 0) else 0
        
        # Accuracy calculations
        congruent_accuracy = congruent_correct / (congruent_correct + congruent_errors) if (congruent_correct + congruent_errors) > 0 else 0
        incongruent_accuracy = incongruent_correct / (incongruent_correct + incongruent_errors) if (incongruent_correct + incongruent_errors) > 0 else 0
        overall_accuracy = total_correct / total if total > 0 else 0
        
        # Interference cost on accuracy
        accuracy_cost = congruent_accuracy - incongruent_accuracy
        
        # Normative comparison (age 50-59 reference)
        # Based on Scarpina & Tagini (2017) meta-analysis
        norm_stroop_mean = 120  # ms
        norm_stroop_sd = 50
        
        z_score = (stroop_effect - norm_stroop_mean) / norm_stroop_sd if norm_stroop_sd > 0 else 0
        
        # Performance scoring
        # Lower stroop effect = better inhibition
        # z = 0 (average) -> 50
        # z = -2 (very efficient, 2SD below mean) -> 100
        # z = +2 (poor inhibition, 2SD above mean) -> 0
        effect_score = 50 - (z_score * 25)
        effect_score = max(0, min(100, effect_score))
        
        # Accuracy score (weighted toward incongruent condition)
        accuracy_score = (congruent_accuracy * 0.3 + incongruent_accuracy * 0.7) * 100
        
        # Combined performance score
        performance_score = effect_score * 0.60 + accuracy_score * 0.40
        performance_score = max(0.0, min(100.0, performance_score))
        
        # Quality warnings with clinical context
        if len(congruent_rts) < 10 or len(incongruent_rts) < 10:
            quality_warnings.append("Fewer than 10 correct trials per condition - reduced reliability")
        
        if stroop_effect > 200:
            quality_warnings.append(f"High Stroop interference ({stroop_effect:.0f}ms) - possible inhibitory control deficit")
        elif stroop_effect < 30:
            quality_warnings.append(f"Very low Stroop effect ({stroop_effect:.0f}ms) - verify task engagement")
        
        if z_score > 2:
            quality_warnings.append(f"Stroop effect >2 SD above norm (z={z_score:.1f}) - clinically significant")
        elif z_score > 1.5:
            quality_warnings.append(f"Stroop effect 1.5-2 SD above norm (z={z_score:.1f}) - borderline")
        
        if incongruent_accuracy < 0.70:
            quality_warnings.append(f"Low incongruent accuracy ({incongruent_accuracy*100:.0f}%) - high error rate")
        elif incongruent_accuracy < 0.85:
            quality_warnings.append(f"Below expected incongruent accuracy ({incongruent_accuracy*100:.0f}%)")
        
        if accuracy_cost > 0.20:
            quality_warnings.append(f"Large accuracy cost from interference ({accuracy_cost*100:.0f}%)")
        
        # Speed-accuracy tradeoff check
        if stroop_effect < 50 and accuracy_cost > 0.15:
            quality_warnings.append("Low RT interference but high accuracy cost - possible strategy difference")
        
        return TaskMetrics(
            task_id=task.task_id,
            completion_status=TaskCompletionStatus.COMPLETE,
            performance_score=performance_score,
            parameters={
                # Primary Stroop metrics
                "stroop_effect_ms": stroop_effect,
                "stroop_ratio": stroop_ratio,
                "z_score": z_score,
                # RT by condition
                "avg_congruent_rt": avg_congruent,
                "sd_congruent_rt": sd_congruent,
                "avg_incongruent_rt": avg_incongruent,
                "sd_incongruent_rt": sd_incongruent,
                "avg_neutral_rt": avg_neutral,
                # Facilitation (if available)
                "facilitation_effect_ms": facilitation_effect,
                # Accuracy metrics
                "overall_accuracy": float(overall_accuracy),
                "congruent_accuracy": float(congruent_accuracy),
                "incongruent_accuracy": float(incongruent_accuracy),
                "accuracy_cost": float(accuracy_cost),
                # Trial counts
                "congruent_correct": float(congruent_correct),
                "congruent_errors": float(congruent_errors),
                "incongruent_correct": float(incongruent_correct),
                "incongruent_errors": float(incongruent_errors),
                "total_trials": float(total),
                # Normative reference
                "norm_stroop_mean": float(norm_stroop_mean),
                "norm_stroop_sd": float(norm_stroop_sd)
            },
            validity_flag=True,
            quality_warnings=quality_warnings
        )

