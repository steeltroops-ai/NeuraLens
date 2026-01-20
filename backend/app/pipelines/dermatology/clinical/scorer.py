"""
Dermatology Pipeline Clinical Scorer

Risk stratification and clinical scoring.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import RISK_CONFIG, RISK_TIERS
from ..schemas import (
    MelanomaResult,
    MalignancyResult,
    SubtypeResult,
    ABCDEFeatures,
    RiskTierResult,
    Escalation
)

logger = logging.getLogger(__name__)


class RiskStratifier:
    """
    Assigns clinical risk tier based on all analysis outputs.
    """
    
    def __init__(self, config=None):
        self.config = config or RISK_CONFIG
        self.tiers = RISK_TIERS
    
    def stratify(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures
    ) -> RiskTierResult:
        """Assign risk tier."""
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            melanoma, malignancy, subtype, abcde
        )
        
        # Determine tier
        tier = self._score_to_tier(risk_score)
        tier_info = self.tiers[tier]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            tier, melanoma, malignancy, subtype, abcde
        )
        
        # Contributing factors
        factors = self._get_contributing_factors(
            melanoma, malignancy, subtype, abcde
        )
        
        # Check for escalations
        escalations = self._check_escalations(
            melanoma, malignancy, subtype, abcde
        )
        
        return RiskTierResult(
            tier=tier,
            tier_name=tier_info["name"],
            risk_score=risk_score,
            action=tier_info["action"],
            urgency=tier_info["urgency"],
            reasoning=reasoning,
            contributing_factors=factors,
            escalations=escalations
        )
    
    def _calculate_risk_score(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures
    ) -> float:
        """Calculate composite risk score (0-100)."""
        score = 0.0
        
        # Melanoma probability (40%)
        score += melanoma.probability * 40
        
        # Malignancy probability (25%)
        score += malignancy.malignant_probability * 25
        
        # Subtype risk (20%)
        if subtype.is_malignant:
            score += subtype.primary_probability * 20
        
        # ABCDE criteria (15%)
        abcde_contribution = (abcde.criteria_met / 5) * 15
        score += abcde_contribution
        
        return min(score, 100)
    
    def _score_to_tier(self, score: float) -> int:
        """Convert score to tier."""
        if score >= self.config.tier_1_threshold:
            return 1  # Critical
        elif score >= self.config.tier_2_threshold:
            return 2  # High
        elif score >= self.config.tier_3_threshold:
            return 3  # Moderate
        elif score >= self.config.tier_4_threshold:
            return 4  # Low
        else:
            return 5  # Benign
    
    def _generate_reasoning(
        self,
        tier: int,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures
    ) -> str:
        """Generate clinical reasoning text."""
        parts = []
        
        # Primary finding
        if tier <= 2:
            parts.append(
                f"Analysis indicates {melanoma.classification.value.replace('_', ' ')} "
                f"for melanoma ({melanoma.probability * 100:.0f}% probability)."
            )
        elif tier == 3:
            parts.append(
                f"Analysis shows moderate risk indicators. "
                f"Melanoma probability: {melanoma.probability * 100:.0f}%."
            )
        elif tier == 4:
            parts.append(
                f"Analysis shows low risk indicators. "
                f"Melanoma unlikely ({(1 - melanoma.probability) * 100:.0f}% benign)."
            )
        else:
            parts.append(
                "Analysis indicates benign characteristics. "
                "No significant concerning features detected."
            )
        
        # ABCDE summary
        if abcde.criteria_met > 0:
            criteria = []
            if abcde.asymmetry.is_concerning:
                criteria.append("asymmetry")
            if abcde.border.is_concerning:
                criteria.append("border irregularity")
            if abcde.color.is_concerning:
                criteria.append("color variation")
            if abcde.diameter.is_concerning:
                criteria.append("diameter > 6mm")
            if abcde.evolution.is_concerning:
                criteria.append("evolution indicators")
            
            parts.append(
                f"ABCDE analysis: {abcde.criteria_met}/5 criteria concerning "
                f"({', '.join(criteria)})."
            )
        
        # Primary subtype
        parts.append(
            f"Most likely type: {subtype.primary_subtype.replace('_', ' ')} "
            f"({subtype.primary_probability * 100:.0f}% confidence)."
        )
        
        return " ".join(parts)
    
    def _get_contributing_factors(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures
    ) -> List[str]:
        """Get list of contributing factors."""
        factors = []
        
        # Add concerning features
        factors.extend(melanoma.concerning_features)
        
        # Add color specifics
        if abcde.color.has_blue_white_veil:
            factors.append("Blue-white veil detected (strong melanoma indicator)")
        
        if abcde.color.num_colors >= 4:
            factors.append(f"Multiple colors detected ({abcde.color.num_colors})")
        
        # Add size
        if abcde.diameter.max_dimension_mm > 10:
            factors.append(f"Large lesion ({abcde.diameter.max_dimension_mm:.1f}mm)")
        
        return factors
    
    def _check_escalations(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures
    ) -> List[Escalation]:
        """Check for escalation triggers."""
        escalations = []
        
        # High melanoma probability
        if melanoma.probability > 0.70:
            escalations.append(Escalation(
                rule_name="high_melanoma_probability",
                action="urgent_referral",
                reason="Melanoma probability > 70%",
                priority=1
            ))
        
        # Multiple ABCDE criteria
        if abcde.criteria_met >= 4:
            escalations.append(Escalation(
                rule_name="multiple_abcde_criteria",
                action="priority_referral",
                reason=f"{abcde.criteria_met}/5 ABCDE criteria concerning",
                priority=2
            ))
        
        # Blue-white veil
        if abcde.color.has_blue_white_veil:
            escalations.append(Escalation(
                rule_name="blue_white_veil",
                action="priority_referral",
                reason="Blue-white veil detected (strong melanoma indicator)",
                priority=2
            ))
        
        # High uncertainty with moderate risk
        if melanoma.uncertainty > 0.4 and melanoma.probability > 0.25:
            escalations.append(Escalation(
                rule_name="high_uncertainty_moderate_risk",
                action="expert_review",
                reason="Moderate risk with high analysis uncertainty",
                priority=3
            ))
        
        return sorted(escalations, key=lambda x: x.priority)


class ClinicalScorer:
    """
    Complete clinical scoring system.
    """
    
    def __init__(self, config=None):
        self.stratifier = RiskStratifier(config)
    
    def score(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures
    ) -> RiskTierResult:
        """Calculate clinical risk score and tier."""
        return self.stratifier.stratify(
            melanoma, malignancy, subtype, abcde
        )
