"""
Clinical Safety Gates for Retinal Analysis v5.0

Implements safety mechanisms for clinical deployment:
1. Quality gating - Reject inadequate images
2. Uncertainty gating - Flag low-confidence predictions
3. Referral gating - Auto-escalate high-risk cases
4. Consistency checking - Validate biomarker coherence

Optimized for:
- Minimizing false negatives (sensitivity > specificity)
- Clinical workflow integration
- Regulatory compliance (IEC 62304)

Author: NeuraLens Medical AI Team
Version: 5.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class GateStatus(str, Enum):
    """Status of a safety gate."""
    PASSED = "passed"
    WARNING = "warning"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


class ClinicalAction(str, Enum):
    """Required clinical action."""
    NONE = "none"
    REVIEW = "review"
    REPEAT_SCAN = "repeat_scan"
    CLINICIAN_REVIEW = "clinician_review"
    URGENT_REFERRAL = "urgent_referral"
    IMMEDIATE_REFERRAL = "immediate_referral"


class ReferralUrgency(str, Enum):
    """Referral urgency levels aligned with clinical protocols."""
    ROUTINE_12M = "routine_12_months"
    ROUTINE_6M = "routine_6_months"
    ENHANCED_3M = "enhanced_3_months"
    REFER_1M = "refer_1_month"
    REFER_2W = "refer_2_weeks"
    URGENT_24H = "urgent_24_hours"
    EMERGENCY = "emergency"


# =============================================================================
# SAFETY GATE RESULT
# =============================================================================

@dataclass
class SafetyGateResult:
    """Result from a single safety gate."""
    gate_name: str
    status: GateStatus
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    action_required: ClinicalAction = ClinicalAction.NONE
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SafetyCheckResult:
    """Aggregated result from all safety gates."""
    overall_status: GateStatus
    passed: bool
    blocked: bool
    escalated: bool
    
    # Individual gate results
    gates: List[SafetyGateResult] = field(default_factory=list)
    
    # Actions
    primary_action: ClinicalAction = ClinicalAction.NONE
    referral_urgency: Optional[ReferralUrgency] = None
    
    # Messages
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    
    # Metadata
    confidence_override: Optional[float] = None  # Force lower confidence
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "blocked": self.blocked,
            "escalated": self.escalated,
            "primary_action": self.primary_action.value,
            "referral_urgency": self.referral_urgency.value if self.referral_urgency else None,
            "summary": self.summary,
            "warnings": self.warnings,
            "blocking_reasons": self.blocking_reasons,
            "gates": [
                {"name": g.gate_name, "status": g.status.value, "message": g.message}
                for g in self.gates
            ],
        }


# =============================================================================
# THRESHOLD CONFIGURATIONS
# =============================================================================

@dataclass
class SafetyThresholds:
    """
    Configurable safety thresholds.
    
    Note: Thresholds are set conservatively to minimize false negatives.
    This means we prefer false positives (unnecessary referrals) over
    missing sight-threatening conditions.
    """
    # Quality thresholds
    min_quality_score: float = 0.30  # Below this = ungradable
    recommended_quality_score: float = 0.60  # Below this = warning
    
    # Uncertainty thresholds
    max_uncertainty_pass: float = 0.15  # Above this = warning
    max_uncertainty_block: float = 0.30  # Above this = require review
    
    # Confidence thresholds
    min_confidence_pass: float = 0.70  # Below this = warning
    min_confidence_block: float = 0.50  # Below this = require review
    
    # DR grade referral thresholds (low = high sensitivity)
    dr_grade_2_threshold: float = 0.25  # P(grade >= 2) > this = refer
    dr_grade_3_threshold: float = 0.15  # P(grade >= 3) > this = urgent
    dr_grade_4_threshold: float = 0.10  # P(grade >= 4) > this = immediate
    
    # Biomarker critical thresholds
    critical_cdr: float = 0.7  # Cup-to-disc ratio above this = urgent
    critical_av_ratio_low: float = 0.5  # AVR below this = urgent
    critical_hemorrhage_count: int = 10  # Above this = moderate-severe NPDR
    
    # DME thresholds
    dme_probability_threshold: float = 0.20  # Low threshold for sensitivity


DEFAULT_THRESHOLDS = SafetyThresholds()


# =============================================================================
# QUALITY GATE
# =============================================================================

class QualityGate:
    """
    Gate based on image quality.
    
    Prevents unreliable predictions on poor quality images.
    """
    
    def __init__(self, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
    
    def check(
        self,
        quality_score: float,
        quality_issues: List[str] = None,
    ) -> SafetyGateResult:
        """
        Check image quality against thresholds.
        
        Args:
            quality_score: Overall quality score (0-1)
            quality_issues: List of identified quality issues
            
        Returns:
            SafetyGateResult
        """
        quality_issues = quality_issues or []
        
        # Critical issues that always block
        critical_issues = [
            "optic_disc_not_visible",
            "image_not_fundus",
            "severe_blur",
        ]
        
        has_critical = any(
            issue in quality_issues for issue in critical_issues
        )
        
        if has_critical or quality_score < self.thresholds.min_quality_score:
            return SafetyGateResult(
                gate_name="quality_gate",
                status=GateStatus.BLOCKED,
                passed=False,
                message=f"Image quality insufficient for analysis (score: {quality_score:.2f})",
                details={
                    "quality_score": quality_score,
                    "issues": quality_issues,
                    "threshold": self.thresholds.min_quality_score,
                },
                action_required=ClinicalAction.REPEAT_SCAN,
            )
        
        if quality_score < self.thresholds.recommended_quality_score:
            return SafetyGateResult(
                gate_name="quality_gate",
                status=GateStatus.WARNING,
                passed=True,
                message=f"Image quality suboptimal (score: {quality_score:.2f})",
                details={
                    "quality_score": quality_score,
                    "issues": quality_issues,
                    "threshold": self.thresholds.recommended_quality_score,
                },
                action_required=ClinicalAction.REVIEW,
            )
        
        return SafetyGateResult(
            gate_name="quality_gate",
            status=GateStatus.PASSED,
            passed=True,
            message="Image quality adequate",
            details={"quality_score": quality_score},
        )


# =============================================================================
# UNCERTAINTY GATE
# =============================================================================

class UncertaintyGate:
    """
    Gate based on prediction uncertainty.
    
    Flags high-uncertainty predictions for clinician review.
    """
    
    def __init__(self, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
    
    def check(
        self,
        uncertainty_std: float,
        model_confidence: float,
        ci_width: Optional[float] = None,
    ) -> SafetyGateResult:
        """
        Check prediction uncertainty.
        
        Args:
            uncertainty_std: Standard deviation of prediction
            model_confidence: Model's confidence score
            ci_width: Width of confidence interval
            
        Returns:
            SafetyGateResult
        """
        # Use CI width if available, otherwise use std
        uncertainty = ci_width / 2 if ci_width else uncertainty_std
        
        # Blocked: very high uncertainty or very low confidence
        if (uncertainty > self.thresholds.max_uncertainty_block or 
            model_confidence < self.thresholds.min_confidence_block):
            return SafetyGateResult(
                gate_name="uncertainty_gate",
                status=GateStatus.BLOCKED,
                passed=False,
                message="Prediction uncertainty too high for reliable diagnosis",
                details={
                    "uncertainty": uncertainty,
                    "confidence": model_confidence,
                    "ci_width": ci_width,
                },
                action_required=ClinicalAction.CLINICIAN_REVIEW,
            )
        
        # Warning: moderate uncertainty
        if (uncertainty > self.thresholds.max_uncertainty_pass or
            model_confidence < self.thresholds.min_confidence_pass):
            return SafetyGateResult(
                gate_name="uncertainty_gate",
                status=GateStatus.WARNING,
                passed=True,
                message="Moderate prediction uncertainty - clinician review recommended",
                details={
                    "uncertainty": uncertainty,
                    "confidence": model_confidence,
                },
                action_required=ClinicalAction.REVIEW,
            )
        
        return SafetyGateResult(
            gate_name="uncertainty_gate",
            status=GateStatus.PASSED,
            passed=True,
            message="Prediction confidence acceptable",
            details={
                "uncertainty": uncertainty,
                "confidence": model_confidence,
            },
        )


# =============================================================================
# REFERRAL GATE
# =============================================================================

class ReferralGate:
    """
    Gate for automatic referral escalation.
    
    Uses conservative (low) thresholds to maximize sensitivity.
    Prefers false positives over false negatives for sight-threatening conditions.
    """
    
    def __init__(self, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
    
    def check(
        self,
        dr_grade: int,
        dr_probabilities: Dict[str, float],
        dme_present: bool,
        dme_probability: float,
        neovascularization_detected: bool,
    ) -> SafetyGateResult:
        """
        Check if automatic referral is needed.
        
        Args:
            dr_grade: Predicted DR grade (0-4)
            dr_probabilities: Probability per grade
            dme_present: DME detection result
            dme_probability: DME probability
            neovascularization_detected: NV detection result
            
        Returns:
            SafetyGateResult with referral urgency
        """
        # Calculate cumulative probabilities
        prob_moderate_or_worse = sum(
            dr_probabilities.get(f"grade_{i}", dr_probabilities.get(str(i), 0))
            for i in [2, 3, 4]
        )
        prob_severe_or_worse = sum(
            dr_probabilities.get(f"grade_{i}", dr_probabilities.get(str(i), 0))
            for i in [3, 4]
        )
        prob_pdr = dr_probabilities.get("grade_4", dr_probabilities.get("4", 0))
        
        # Handle different key formats in probabilities dict
        if prob_moderate_or_worse == 0:
            prob_keys = list(dr_probabilities.keys())
            if any("Moderate" in str(k) for k in prob_keys):
                prob_moderate_or_worse = sum(
                    v for k, v in dr_probabilities.items()
                    if any(x in str(k) for x in ["Moderate", "Severe", "Proliferative"])
                )
                prob_severe_or_worse = sum(
                    v for k, v in dr_probabilities.items()
                    if any(x in str(k) for x in ["Severe", "Proliferative"])
                )
                prob_pdr = sum(
                    v for k, v in dr_probabilities.items()
                    if "Proliferative" in str(k)
                )
        
        # EMERGENCY: PDR or NV with high probability
        if neovascularization_detected or prob_pdr > self.thresholds.dr_grade_4_threshold:
            return SafetyGateResult(
                gate_name="referral_gate",
                status=GateStatus.ESCALATED,
                passed=True,
                message="Sight-threatening condition detected - IMMEDIATE referral required",
                details={
                    "trigger": "neovascularization" if neovascularization_detected else "high_pdr_probability",
                    "pdr_probability": prob_pdr,
                },
                action_required=ClinicalAction.IMMEDIATE_REFERRAL,
            )
        
        # URGENT: Severe NPDR or CSME
        if (prob_severe_or_worse > self.thresholds.dr_grade_3_threshold or
            (dme_present and dme_probability > 0.5)):
            return SafetyGateResult(
                gate_name="referral_gate",
                status=GateStatus.ESCALATED,
                passed=True,
                message="Severe condition detected - urgent referral required within 24 hours",
                details={
                    "severe_npdr_probability": prob_severe_or_worse,
                    "dme_present": dme_present,
                },
                action_required=ClinicalAction.URGENT_REFERRAL,
            )
        
        # REFER: Moderate NPDR or possible DME
        if (prob_moderate_or_worse > self.thresholds.dr_grade_2_threshold or
            dme_probability > self.thresholds.dme_probability_threshold):
            return SafetyGateResult(
                gate_name="referral_gate",
                status=GateStatus.WARNING,
                passed=True,
                message="Moderate findings - ophthalmology referral recommended",
                details={
                    "moderate_or_worse_probability": prob_moderate_or_worse,
                    "dme_probability": dme_probability,
                },
                action_required=ClinicalAction.CLINICIAN_REVIEW,
            )
        
        return SafetyGateResult(
            gate_name="referral_gate",
            status=GateStatus.PASSED,
            passed=True,
            message="No immediate referral required - continue routine screening",
            details={
                "dr_grade": dr_grade,
                "moderate_probability": prob_moderate_or_worse,
            },
        )


# =============================================================================
# CONSISTENCY GATE
# =============================================================================

class ConsistencyGate:
    """
    Gate for biomarker consistency checking.
    
    Validates that biomarker predictions are internally consistent.
    Flags contradictory findings for review.
    """
    
    # Consistency rules: (condition, expected_consequence, severity)
    CONSISTENCY_RULES = [
        # DR Grade 4 should have neovascularization
        (
            lambda b: b.get("dr_grade", 0) == 4,
            lambda b: b.get("neovascularization", False) == True,
            "DR grade 4 but no neovascularization detected",
            "warning"
        ),
        # High hemorrhage count should correlate with DR grade
        (
            lambda b: b.get("hemorrhage_count", 0) > 10,
            lambda b: b.get("dr_grade", 0) >= 2,
            "High hemorrhage count but low DR grade",
            "warning"
        ),
        # DME should correlate with macular changes
        (
            lambda b: b.get("dme_present", False),
            lambda b: b.get("macular_thickness", 250) > 280,
            "DME detected but macular thickness normal",
            "info"
        ),
        # High CDR should show rim changes
        (
            lambda b: b.get("cup_disc_ratio", 0.3) > 0.7,
            lambda b: b.get("rim_area_mm2", 1.5) < 1.2,
            "High CDR without corresponding rim thinning",
            "warning"
        ),
    ]
    
    def check(
        self,
        biomarkers: Dict[str, Any],
    ) -> SafetyGateResult:
        """
        Check biomarker consistency.
        
        Args:
            biomarkers: Dictionary of all biomarker values
            
        Returns:
            SafetyGateResult
        """
        violations = []
        warnings = []
        
        for condition_fn, expected_fn, message, severity in self.CONSISTENCY_RULES:
            try:
                if condition_fn(biomarkers) and not expected_fn(biomarkers):
                    if severity == "warning":
                        warnings.append(message)
                    violations.append({"rule": message, "severity": severity})
            except Exception:
                continue  # Skip rules that can't be evaluated
        
        if len(warnings) > 2:
            # Multiple inconsistencies - require review
            return SafetyGateResult(
                gate_name="consistency_gate",
                status=GateStatus.WARNING,
                passed=True,
                message="Multiple biomarker inconsistencies detected",
                details={"violations": violations},
                action_required=ClinicalAction.CLINICIAN_REVIEW,
            )
        
        if warnings:
            return SafetyGateResult(
                gate_name="consistency_gate",
                status=GateStatus.WARNING,
                passed=True,
                message=f"Minor inconsistency: {warnings[0]}",
                details={"violations": violations},
                action_required=ClinicalAction.REVIEW,
            )
        
        return SafetyGateResult(
            gate_name="consistency_gate",
            status=GateStatus.PASSED,
            passed=True,
            message="Biomarkers internally consistent",
            details={},
        )


# =============================================================================
# CRITICAL BIOMARKER GATE
# =============================================================================

class CriticalBiomarkerGate:
    """
    Gate for critical biomarker values.
    
    Flags individual biomarkers that exceed critical thresholds
    regardless of overall risk score.
    """
    
    def __init__(self, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
    
    def check(
        self,
        cup_disc_ratio: float,
        av_ratio: float,
        hemorrhage_count: int,
        neovascularization: bool,
        macular_thickness: Optional[float] = None,
    ) -> SafetyGateResult:
        """
        Check individual critical biomarkers.
        """
        critical_findings = []
        
        # Glaucoma risk
        if cup_disc_ratio > self.thresholds.critical_cdr:
            critical_findings.append({
                "biomarker": "cup_disc_ratio",
                "value": cup_disc_ratio,
                "threshold": self.thresholds.critical_cdr,
                "condition": "High glaucoma risk",
            })
        
        # Hypertensive retinopathy
        if av_ratio < self.thresholds.critical_av_ratio_low:
            critical_findings.append({
                "biomarker": "av_ratio",
                "value": av_ratio,
                "threshold": self.thresholds.critical_av_ratio_low,
                "condition": "Severe arterial narrowing",
            })
        
        # Severe hemorrhage
        if hemorrhage_count > self.thresholds.critical_hemorrhage_count:
            critical_findings.append({
                "biomarker": "hemorrhage_count",
                "value": hemorrhage_count,
                "threshold": self.thresholds.critical_hemorrhage_count,
                "condition": "Significant retinal hemorrhage",
            })
        
        # Proliferative changes
        if neovascularization:
            critical_findings.append({
                "biomarker": "neovascularization",
                "value": True,
                "threshold": None,
                "condition": "Proliferative retinopathy",
            })
        
        if critical_findings:
            urgency = ClinicalAction.URGENT_REFERRAL
            if neovascularization:
                urgency = ClinicalAction.IMMEDIATE_REFERRAL
            
            return SafetyGateResult(
                gate_name="critical_biomarker_gate",
                status=GateStatus.ESCALATED,
                passed=True,
                message=f"Critical biomarker(s) detected: {', '.join(f['condition'] for f in critical_findings)}",
                details={"critical_findings": critical_findings},
                action_required=urgency,
            )
        
        return SafetyGateResult(
            gate_name="critical_biomarker_gate",
            status=GateStatus.PASSED,
            passed=True,
            message="No critical biomarker values",
            details={},
        )


# =============================================================================
# INTEGRATED SAFETY CHECKER
# =============================================================================

class ClinicalSafetyChecker:
    """
    Integrated safety checker combining all gates.
    
    Runs all safety gates and aggregates results.
    Provides overall safety status and required actions.
    """
    
    def __init__(
        self,
        thresholds: SafetyThresholds = DEFAULT_THRESHOLDS,
    ):
        self.thresholds = thresholds
        
        self.quality_gate = QualityGate(thresholds)
        self.uncertainty_gate = UncertaintyGate(thresholds)
        self.referral_gate = ReferralGate(thresholds)
        self.consistency_gate = ConsistencyGate()
        self.biomarker_gate = CriticalBiomarkerGate(thresholds)
    
    def check_all(
        self,
        quality_score: float,
        quality_issues: List[str],
        uncertainty_std: float,
        model_confidence: float,
        dr_grade: int,
        dr_probabilities: Dict[str, float],
        dme_present: bool,
        dme_probability: float,
        biomarkers: Dict[str, Any],
        ci_width: Optional[float] = None,
    ) -> SafetyCheckResult:
        """
        Run all safety gates and aggregate results.
        
        Args:
            quality_score: Image quality score (0-1)
            quality_issues: List of quality issues
            uncertainty_std: Prediction uncertainty
            model_confidence: Model confidence
            dr_grade: Predicted DR grade
            dr_probabilities: Per-grade probabilities
            dme_present: DME detection result
            dme_probability: DME probability
            biomarkers: All biomarker values
            ci_width: Confidence interval width
            
        Returns:
            SafetyCheckResult with aggregated status
        """
        gate_results = []
        
        # Run each gate
        gate_results.append(self.quality_gate.check(
            quality_score, quality_issues
        ))
        
        gate_results.append(self.uncertainty_gate.check(
            uncertainty_std, model_confidence, ci_width
        ))
        
        gate_results.append(self.referral_gate.check(
            dr_grade,
            dr_probabilities,
            dme_present,
            dme_probability,
            biomarkers.get("neovascularization", False),
        ))
        
        gate_results.append(self.consistency_gate.check(biomarkers))
        
        gate_results.append(self.biomarker_gate.check(
            cup_disc_ratio=biomarkers.get("cup_disc_ratio", 0.3),
            av_ratio=biomarkers.get("av_ratio", 0.7),
            hemorrhage_count=int(biomarkers.get("hemorrhage_count", 0)),
            neovascularization=biomarkers.get("neovascularization", False),
        ))
        
        return self._aggregate_results(gate_results)
    
    def _aggregate_results(
        self,
        gate_results: List[SafetyGateResult],
    ) -> SafetyCheckResult:
        """Aggregate individual gate results."""
        blocked = any(g.status == GateStatus.BLOCKED for g in gate_results)
        escalated = any(g.status == GateStatus.ESCALATED for g in gate_results)
        warnings = [g for g in gate_results if g.status == GateStatus.WARNING]
        
        # Determine overall status
        if blocked:
            overall_status = GateStatus.BLOCKED
        elif escalated:
            overall_status = GateStatus.ESCALATED
        elif warnings:
            overall_status = GateStatus.WARNING
        else:
            overall_status = GateStatus.PASSED
        
        # Collect messages
        blocking_reasons = [
            g.message for g in gate_results if g.status == GateStatus.BLOCKED
        ]
        warning_messages = [
            g.message for g in gate_results if g.status == GateStatus.WARNING
        ]
        
        # Determine primary action (highest severity)
        action_priority = [
            ClinicalAction.IMMEDIATE_REFERRAL,
            ClinicalAction.URGENT_REFERRAL,
            ClinicalAction.CLINICIAN_REVIEW,
            ClinicalAction.REPEAT_SCAN,
            ClinicalAction.REVIEW,
            ClinicalAction.NONE,
        ]
        
        primary_action = ClinicalAction.NONE
        for action in action_priority:
            if any(g.action_required == action for g in gate_results):
                primary_action = action
                break
        
        # Determine referral urgency
        referral_urgency = None
        if primary_action == ClinicalAction.IMMEDIATE_REFERRAL:
            referral_urgency = ReferralUrgency.EMERGENCY
        elif primary_action == ClinicalAction.URGENT_REFERRAL:
            referral_urgency = ReferralUrgency.URGENT_24H
        elif primary_action == ClinicalAction.CLINICIAN_REVIEW:
            referral_urgency = ReferralUrgency.REFER_2W
        
        # Generate summary
        if blocked:
            summary = f"Analysis blocked: {'; '.join(blocking_reasons)}"
        elif escalated:
            summary = "Urgent clinical action required - see referral recommendations"
        elif warnings:
            summary = f"{len(warnings)} warning(s) - review recommended"
        else:
            summary = "All safety checks passed"
        
        return SafetyCheckResult(
            overall_status=overall_status,
            passed=not blocked,
            blocked=blocked,
            escalated=escalated,
            gates=gate_results,
            primary_action=primary_action,
            referral_urgency=referral_urgency,
            summary=summary,
            warnings=warning_messages,
            blocking_reasons=blocking_reasons,
        )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

clinical_safety_checker = ClinicalSafetyChecker()
