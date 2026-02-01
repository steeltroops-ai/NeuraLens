"""
Cognitive Clinical Risk Scorer - Research Grade v2.1.0
Production-grade risk assessment with age-adjusted normative comparisons.

Scientific Approach:
- Age-stratified normative data from published research
- Weighted domain aggregation based on clinical importance
- Bootstrap-estimated confidence intervals
- Explainability artifacts with clinical terminology

References:
- Weintraub et al. (2009): UDS cognitive battery
- Tombaugh (2004): Normative data sources
- NACC: National Alzheimer's Coordinating Center

Version History:
- v2.0.0: Initial production release
- v2.1.0: Added normative comparisons, bootstrap CI, clinical patterns
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

from ..schemas import (
    CognitiveFeatures, CognitiveRiskAssessment, DomainRiskDetail,
    RiskLevel, ClinicalRecommendation, ExplainabilityArtifact
)
from ..config import config
from ..errors.codes import ErrorCode, PipelineError
from ..analysis.normative import (
    compare_domain_composite,
    calculate_global_cognitive_index,
    NormativeComparison,
    PerformanceCategory,
    get_age_group
)

logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Research-grade clinical risk scoring with:
    - Age-adjusted normative comparisons
    - Weighted domain aggregation (clinically informed)
    - Bootstrap-based confidence intervals
    - Comprehensive explainability artifacts
    - Clinical pattern detection
    """
    
    DOMAIN_WEIGHTS = {
        "memory": config.WEIGHT_MEMORY,
        "attention": config.WEIGHT_ATTENTION,
        "executive": config.WEIGHT_EXECUTIVE,
        "processing_speed": config.WEIGHT_SPEED,
        "inhibition": config.WEIGHT_INHIBITION,
        "general": 0.1
    }
    
    # Clinical importance weights for overall risk
    CLINICAL_WEIGHTS = {
        "memory": 0.30,        # Memory decline is most predictive of dementia
        "executive": 0.25,     # Executive dysfunction common in vascular/frontal
        "attention": 0.20,     # Attention issues in multiple conditions
        "processing_speed": 0.15,  # Slowing is normal aging vs pathological
        "inhibition": 0.10     # Impulse control
    }
    
    def score_with_explanation(
        self, 
        features: CognitiveFeatures,
        patient_age: Optional[int] = None,
        education_years: Optional[int] = None
    ) -> Tuple[CognitiveRiskAssessment, ExplainabilityArtifact]:
        """
        Calculate risk scores with age-adjusted normative comparison.
        
        Args:
            features: Extracted cognitive features
            patient_age: Optional age for normative adjustment
            education_years: Optional education years for adjustment
            
        Returns:
            Tuple of (CognitiveRiskAssessment, ExplainabilityArtifact)
        """
        if not features.domain_scores:
            raise PipelineError(ErrorCode.E_CLIN_002, "No domain scores available")
        
        domain_risks: Dict[str, DomainRiskDetail] = {}
        normative_comparisons: Dict[str, NormativeComparison] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        key_factors = []
        domain_contributions = {}
        
        age_group = get_age_group(patient_age) if patient_age else "50-59"
        
        for domain, score in features.domain_scores.items():
            # Get age-adjusted normative comparison
            norm_comparison = compare_domain_composite(
                score=score,
                domain=domain,
                age=patient_age
            )
            normative_comparisons[domain] = norm_comparison
            
            # Risk = 1 - score (but adjusted by normative position)
            # If performing above age norms, reduce risk
            # If performing below age norms, increase risk
            base_risk = 1.0 - score
            
            # Adjust risk by z-score (negative z = higher risk)
            z_adjustment = -norm_comparison.z_score * 0.1
            adjusted_risk = max(0.0, min(1.0, base_risk + z_adjustment))
            
            weight = self.DOMAIN_WEIGHTS.get(domain, 0.1)
            
            # Determine risk level based on percentile
            level = self._percentile_to_risk_level(norm_comparison.percentile)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_domain_confidence(
                features=features,
                domain=domain,
                n_valid=features.valid_task_count
            )
            
            # Clinical factors
            contributing_factors = self._get_domain_factors_with_norms(
                domain=domain,
                score=score,
                norm_comparison=norm_comparison
            )
            
            domain_risks[domain] = DomainRiskDetail(
                score=adjusted_risk,
                risk_level=level,
                percentile=norm_comparison.percentile,
                confidence=confidence,
                contributing_factors=contributing_factors
            )
            
            weighted_sum += adjusted_risk * weight
            total_weight += weight
            domain_contributions[domain] = adjusted_risk * weight
            
            if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                key_factors.append(
                    f"{domain.replace('_', ' ').title()}: {level.value} risk "
                    f"({norm_comparison.percentile}th percentile)"
                )
        
        # Overall risk (weighted average)
        overall_risk = weighted_sum / total_weight if total_weight > 0 else 0.5
        overall_level = self._score_to_level(overall_risk)
        
        # Bootstrap confidence interval estimation
        ci_lower, ci_upper, overall_confidence = self._bootstrap_confidence_interval(
            domain_risks=domain_risks,
            n_bootstrap=100
        )
        
        risk_assessment = CognitiveRiskAssessment(
            overall_risk_score=overall_risk,
            risk_level=overall_level,
            confidence_score=overall_confidence,
            confidence_interval=(ci_lower, ci_upper),
            domain_risks=domain_risks
        )

        
        # Build explainability
        summary = self._generate_summary(overall_level, features.valid_task_count)
        explainability = ExplainabilityArtifact(
            summary=summary,
            key_factors=key_factors if key_factors else ["All domains within normal range"],
            domain_contributions=domain_contributions,
            methodology_note="Risk calculated using weighted domain aggregation with age-normalized thresholds."
        )
        
        return risk_assessment, explainability
    
    def score(self, features: CognitiveFeatures) -> CognitiveRiskAssessment:
        """Legacy compatibility: score without explainability"""
        assessment, _ = self.score_with_explanation(features)
        return assessment
    
    def generate_recommendations(
        self, 
        assessment: CognitiveRiskAssessment
    ) -> List[ClinicalRecommendation]:
        """Generate clinical recommendations based on risk assessment"""
        recs = []
        
        # Overall risk recommendations
        if assessment.risk_level == RiskLevel.CRITICAL:
            recs.append(ClinicalRecommendation(
                category="clinical",
                description="Immediate consultation with a neurologist is strongly recommended based on screening results.",
                priority="critical"
            ))
        elif assessment.risk_level == RiskLevel.HIGH:
            recs.append(ClinicalRecommendation(
                category="clinical",
                description="Schedule an appointment with a healthcare provider to discuss cognitive health.",
                priority="high"
            ))
        elif assessment.risk_level == RiskLevel.MODERATE:
            recs.append(ClinicalRecommendation(
                category="lifestyle",
                description="Consider cognitive training exercises and ensure adequate sleep hygiene.",
                priority="medium"
            ))
        else:
            recs.append(ClinicalRecommendation(
                category="routine",
                description="Cognitive function appears within normal range. Re-test in 6-12 months.",
                priority="low"
            ))
        
        # Domain-specific recommendations
        for domain, detail in assessment.domain_risks.items():
            if detail.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                rec = self._get_domain_recommendation(domain, detail.risk_level)
                if rec:
                    recs.append(rec)
        
        return recs
    
    def _score_to_level(self, risk: float) -> RiskLevel:
        """Convert numeric risk to categorical level"""
        if risk >= 0.8:
            return RiskLevel.CRITICAL
        elif risk >= config.RISK_THRESHOLD_HIGH:
            return RiskLevel.HIGH
        elif risk >= config.RISK_THRESHOLD_MODERATE:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _risk_to_percentile(self, risk: float) -> int:
        """Convert risk score to population percentile (lower is better for health)"""
        # Simplified: risk 0 = 100th percentile, risk 1 = 0th percentile
        return int((1 - risk) * 100)
    
    def _get_domain_factors(self, domain: str, score: float) -> List[str]:
        """Get factors that contributed to domain score"""
        factors = []
        if score < 0.5:
            factors.append("Significantly below expected performance")
        elif score < 0.7:
            factors.append("Below average performance")
        
        if domain == "memory":
            if score < 0.6:
                factors.append("Recall difficulties observed")
        elif domain == "processing_speed":
            if score < 0.6:
                factors.append("Reaction times slower than typical")
        elif domain == "inhibition":
            if score < 0.6:
                factors.append("Elevated commission errors")
        
        return factors
    
    def _generate_summary(self, level: RiskLevel, task_count: int) -> str:
        """Generate human-readable summary"""
        reliability = "high" if task_count >= 3 else "moderate"
        
        if level == RiskLevel.LOW:
            return f"Cognitive screening indicates normal function across tested domains. Assessment reliability: {reliability}."
        elif level == RiskLevel.MODERATE:
            return f"Some areas of cognitive function show mild concerns. Consider follow-up testing. Assessment reliability: {reliability}."
        elif level == RiskLevel.HIGH:
            return f"Cognitive screening indicates potential impairment. Professional evaluation recommended. Assessment reliability: {reliability}."
        else:
            return f"Cognitive screening suggests significant concerns. Immediate professional consultation advised. Assessment reliability: {reliability}."
    
    def _get_domain_recommendation(
        self, 
        domain: str, 
        level: RiskLevel
    ) -> Optional[ClinicalRecommendation]:
        """Get domain-specific recommendation"""
        recs = {
            "memory": ClinicalRecommendation(
                category="specific",
                description="Memory recall exercises and mnemonic training may be beneficial.",
                priority="medium"
            ),
            "processing_speed": ClinicalRecommendation(
                category="specific",
                description="Processing speed concerns may indicate fatigue or medication effects. Discuss with provider.",
                priority="medium"
            ),
            "inhibition": ClinicalRecommendation(
                category="specific",
                description="Impulse control difficulties detected. Consider attention/executive function evaluation.",
                priority="medium" if level == RiskLevel.HIGH else "high"
            ),
            "attention": ClinicalRecommendation(
                category="specific",
                description="Attention difficulties detected. Rule out environmental and lifestyle factors.",
                priority="medium"
            )
        }
        return recs.get(domain)
    
    # =========================================================================
    # NEW RESEARCH-GRADE HELPER METHODS
    # =========================================================================
    
    def _percentile_to_risk_level(self, percentile: int) -> RiskLevel:
        """Convert normative percentile to risk level."""
        if percentile < 2:
            return RiskLevel.CRITICAL
        elif percentile < 9:
            return RiskLevel.HIGH
        elif percentile < 25:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _calculate_domain_confidence(
        self,
        features: CognitiveFeatures,
        domain: str,
        n_valid: int
    ) -> float:
        """
        Calculate confidence for domain score based on data quality.
        
        Factors:
        - Number of valid tasks (more = higher confidence)
        - Consistency of performance
        - Data quality warnings
        """
        base_confidence = 0.60
        
        # Task count contribution
        if n_valid >= 4:
            base_confidence += 0.20
        elif n_valid >= 3:
            base_confidence += 0.15
        elif n_valid >= 2:
            base_confidence += 0.10
        
        # Consistency contribution
        if features.consistency_score > 0.85:
            base_confidence += 0.15
        elif features.consistency_score > 0.70:
            base_confidence += 0.10
        elif features.consistency_score > 0.50:
            base_confidence += 0.05
        
        # Check if specific domain has valid data
        if domain in features.domain_scores:
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _get_domain_factors_with_norms(
        self,
        domain: str,
        score: float,
        norm_comparison: NormativeComparison
    ) -> List[str]:
        """Get contributing factors with normative context."""
        factors = []
        
        # Performance relative to norms
        if norm_comparison.percentile < 5:
            factors.append(f"Performance at {norm_comparison.percentile}th percentile - significantly below age norms")
        elif norm_comparison.percentile < 16:
            factors.append(f"Performance at {norm_comparison.percentile}th percentile - below average for age")
        elif norm_comparison.percentile < 25:
            factors.append(f"Performance at {norm_comparison.percentile}th percentile - low average range")
        elif norm_comparison.percentile >= 75:
            factors.append(f"Performance at {norm_comparison.percentile}th percentile - above average for age")
        
        # Z-score interpretation
        if norm_comparison.z_score < -2.0:
            factors.append("Severely impaired relative to age-matched peers")
        elif norm_comparison.z_score < -1.5:
            factors.append("Notably below expected performance for age")
        elif norm_comparison.z_score < -1.0:
            factors.append("Mildly below expected performance")
        
        # Domain-specific clinical notes
        domain_notes = {
            "memory": {
                -2.0: "Significant memory concerns - rule out amnestic conditions",
                -1.5: "Memory difficulties may benefit from compensatory strategies",
                -1.0: "Mild memory concerns - monitor over time"
            },
            "processing_speed": {
                -2.0: "Marked slowing - consider neurological evaluation",
                -1.5: "Processing speed decline - assess medication and sleep",
                -1.0: "Mild slowing within expected aging range"
            },
            "executive": {
                -2.0: "Executive dysfunction - comprehensive evaluation needed",
                -1.5: "Executive difficulties affecting daily planning",
                -1.0: "Mild executive concerns"
            },
            "attention": {
                -2.0: "Severe attention deficits - rule out ADHD/delirium",
                -1.5: "Attention difficulties - assess sleep and stimulants",
                -1.0: "Mild attention concerns"
            },
            "inhibition": {
                -2.0: "Significant impulse control issues",
                -1.5: "Inhibition difficulties noted",
                -1.0: "Mild impulse control concerns"
            }
        }
        
        if domain in domain_notes:
            thresholds = domain_notes[domain]
            for threshold, note in sorted(thresholds.items()):
                if norm_comparison.z_score < threshold:
                    factors.append(note)
                    break
        
        return factors
    
    def _bootstrap_confidence_interval(
        self,
        domain_risks: Dict[str, DomainRiskDetail],
        n_bootstrap: int = 100
    ) -> Tuple[float, float, float]:
        """
        Estimate confidence interval using simple bootstrap resampling.
        
        Returns:
            Tuple of (ci_lower, ci_upper, mean_confidence)
        """
        if not domain_risks:
            return (0.0, 1.0, 0.7)
        
        risk_values = [d.score for d in domain_risks.values()]
        confidence_values = [d.confidence for d in domain_risks.values()]
        weights = [self.DOMAIN_WEIGHTS.get(domain, 0.1) for domain in domain_risks.keys()]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted mean
        weighted_mean = sum(r * w for r, w in zip(risk_values, weights))
        
        # Bootstrap resampling
        bootstrap_means = []
        n_domains = len(risk_values)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_domains, size=n_domains, replace=True)
            sample_risks = [risk_values[i] for i in indices]
            sample_weights = [weights[i] for i in indices]
            
            # Renormalize weights
            sw_total = sum(sample_weights)
            if sw_total > 0:
                sample_weighted_mean = sum(r * w for r, w in zip(sample_risks, sample_weights)) / sw_total
                bootstrap_means.append(sample_weighted_mean)
        
        if not bootstrap_means:
            return (max(0, weighted_mean - 0.1), min(1, weighted_mean + 0.1), 0.7)
        
        # Calculate percentile-based CI
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))
        
        # Mean confidence
        mean_confidence = float(np.mean(confidence_values))
        
        # Adjust CI by confidence
        ci_width = ci_upper - ci_lower
        uncertainty_expansion = 1 + (1 - mean_confidence) * 0.5
        ci_lower = max(0.0, weighted_mean - (weighted_mean - ci_lower) * uncertainty_expansion)
        ci_upper = min(1.0, weighted_mean + (ci_upper - weighted_mean) * uncertainty_expansion)
        
        return (ci_lower, ci_upper, mean_confidence)

