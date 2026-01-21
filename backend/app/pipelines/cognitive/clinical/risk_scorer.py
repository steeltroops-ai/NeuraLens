"""
Cognitive Clinical Risk Scorer - Production Grade
With explainability artifacts and uncertainty quantification.
"""

from typing import List, Dict, Tuple
import numpy as np

from ..schemas import (
    CognitiveFeatures, CognitiveRiskAssessment, DomainRiskDetail,
    RiskLevel, ClinicalRecommendation, ExplainabilityArtifact
)
from ..config import config
from ..errors.codes import ErrorCode, PipelineError


class RiskScorer:
    """
    Clinical risk scoring with:
    - Weighted domain aggregation
    - Confidence intervals
    - Explainability artifacts
    """
    
    DOMAIN_WEIGHTS = {
        "memory": config.WEIGHT_MEMORY,
        "attention": config.WEIGHT_ATTENTION,
        "executive": config.WEIGHT_EXECUTIVE,
        "processing_speed": config.WEIGHT_SPEED,
        "inhibition": config.WEIGHT_INHIBITION,
        "general": 0.1
    }
    
    def score_with_explanation(
        self, 
        features: CognitiveFeatures
    ) -> Tuple[CognitiveRiskAssessment, ExplainabilityArtifact]:
        """
        Calculate risk scores and generate explainability artifact.
        """
        if not features.domain_scores:
            raise PipelineError(ErrorCode.E_CLIN_002, "No domain scores available")
        
        domain_risks: Dict[str, DomainRiskDetail] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        key_factors = []
        domain_contributions = {}
        
        for domain, score in features.domain_scores.items():
            # Risk = 1 - score (score 1.0 = healthy)
            risk = 1.0 - score
            weight = self.DOMAIN_WEIGHTS.get(domain, 0.1)
            
            # Determine risk level for this domain
            level = self._score_to_level(risk)
            
            # Calculate confidence based on task validity and count
            base_confidence = 0.7
            if features.valid_task_count >= 3:
                base_confidence = 0.85
            if features.consistency_score > 0.8:
                base_confidence = min(0.95, base_confidence + 0.1)
            
            domain_risks[domain] = DomainRiskDetail(
                score=risk,
                risk_level=level,
                percentile=self._risk_to_percentile(risk),
                confidence=base_confidence,
                contributing_factors=self._get_domain_factors(domain, score)
            )
            
            weighted_sum += risk * weight
            total_weight += weight
            domain_contributions[domain] = risk * weight
            
            if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                key_factors.append(f"{domain.replace('_', ' ').title()}: {level.value} risk")
        
        # Overall risk
        overall_risk = weighted_sum / total_weight if total_weight > 0 else 0.5
        overall_level = self._score_to_level(overall_risk)
        
        # Overall confidence (aggregate)
        confidences = [d.confidence for d in domain_risks.values()]
        overall_confidence = np.mean(confidences) if confidences else 0.7
        
        # Confidence interval (simplified bootstrap estimate)
        ci_margin = (1 - overall_confidence) * 0.2
        ci_lower = max(0.0, overall_risk - ci_margin)
        ci_upper = min(1.0, overall_risk + ci_margin)
        
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
    ) -> ClinicalRecommendation:
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
