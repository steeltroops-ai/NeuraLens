"""
Retinal Pipeline - Condition Grader
Disease-specific grading for DR, AMD, Glaucoma.

Matches speech/clinical/condition_classifier.py structure.

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..config import DR_GRADE_CRITERIA, DRGrade

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """Result from disease grading."""
    grade: int
    grade_name: str
    description: str
    confidence: float
    referral_urgency: str
    icd10_code: str
    contributing_features: List[str]


class DiabeticRetinopathyGrader:
    """
    ICDR-based Diabetic Retinopathy grading.
    
    Grades:
    0 - No DR
    1 - Mild NPDR
    2 - Moderate NPDR
    3 - Severe NPDR
    4 - Proliferative DR
    """
    
    def grade(
        self,
        biomarkers: Dict[str, float],
        image_quality: float = 0.8
    ) -> GradingResult:
        """
        Grade diabetic retinopathy from biomarkers.
        
        Args:
            biomarkers: Extracted lesion counts and metrics
            image_quality: Quality score for confidence adjustment
            
        Returns:
            GradingResult with ICDR grade
        """
        ma_count = biomarkers.get("microaneurysm_count", 0)
        hemorrhage_count = biomarkers.get("hemorrhage_count", 0)
        exudate_area = biomarkers.get("exudate_area_percent", 0)
        cws_count = biomarkers.get("cotton_wool_spots", 0)
        nv_present = biomarkers.get("neovascularization", 0) > 0
        irma_present = biomarkers.get("irma_count", 0) > 0
        
        features = []
        
        # PDR check first
        if nv_present:
            grade = 4
            features.append("Neovascularization present")
        # Severe NPDR (4-2-1 rule simplified)
        elif hemorrhage_count >= 20 or (cws_count >= 1 and irma_present):
            grade = 3
            if hemorrhage_count >= 20:
                features.append(f"Extensive hemorrhages ({hemorrhage_count})")
            if cws_count >= 1:
                features.append(f"Cotton wool spots ({cws_count})")
            if irma_present:
                features.append("IRMA present")
        # Moderate NPDR
        elif hemorrhage_count >= 5 or exudate_area >= 1.0:
            grade = 2
            if hemorrhage_count >= 5:
                features.append(f"Hemorrhages ({hemorrhage_count})")
            if exudate_area >= 1.0:
                features.append(f"Exudates ({exudate_area:.1f}%)")
        # Mild NPDR
        elif ma_count >= 1:
            grade = 1
            features.append(f"Microaneurysms ({ma_count})")
        # No DR
        else:
            grade = 0
        
        # Get grade info
        dr_grade = DRGrade(grade)
        grade_info = DR_GRADE_CRITERIA.get(dr_grade, {})
        
        # Confidence based on quality and feature clarity
        confidence = min(0.95, 0.7 + image_quality * 0.25)
        
        # ICD-10 codes
        icd10_codes = {
            0: "E11.319",
            1: "E11.329",
            2: "E11.339",
            3: "E11.349",
            4: "E11.359"
        }
        
        return GradingResult(
            grade=grade,
            grade_name=grade_info.get("description", f"Grade {grade}"),
            description=grade_info.get("criteria", ""),
            confidence=confidence,
            referral_urgency=grade_info.get("referral", "Routine"),
            icd10_code=icd10_codes.get(grade, "E11.319"),
            contributing_features=features
        )


class GlaucomaRiskGrader:
    """
    Glaucoma risk grading based on optic disc parameters.
    """
    
    def grade(
        self,
        biomarkers: Dict[str, float],
        image_quality: float = 0.8
    ) -> GradingResult:
        """Grade glaucoma risk from CDR and disc parameters."""
        cdr = biomarkers.get("cup_disc_ratio", 0.35)
        cdr_h = biomarkers.get("cup_disc_ratio_h", cdr)
        rim_area = biomarkers.get("rim_area_mm2", 1.5)
        isnt_compliant = biomarkers.get("isnt_compliant", 1.0) > 0.5
        
        features = []
        
        # Grading based on CDR
        if cdr >= 0.8:
            grade = 3
            features.append(f"Large CDR ({cdr:.2f})")
        elif cdr >= 0.6:
            grade = 2
            features.append(f"Elevated CDR ({cdr:.2f})")
        elif cdr >= 0.5:
            grade = 1
            features.append(f"Borderline CDR ({cdr:.2f})")
        else:
            grade = 0
        
        if not isnt_compliant:
            grade = max(grade, 2)
            features.append("ISNT rule violation")
        
        if rim_area < 1.0:
            grade = max(grade, 2)
            features.append(f"Thin rim ({rim_area:.2f} mm2)")
        
        grade_names = {
            0: "Normal",
            1: "Glaucoma Suspect",
            2: "Moderate Risk",
            3: "High Risk"
        }
        
        referrals = {
            0: "Routine 12 months",
            1: "Follow-up 6 months",
            2: "Refer 4 weeks",
            3: "Urgent referral"
        }
        
        icd10_codes = {
            0: "H40.10X0",
            1: "H40.11X1",
            2: "H40.11X2",
            3: "H40.11X3"
        }
        
        return GradingResult(
            grade=grade,
            grade_name=grade_names.get(grade, "Unknown"),
            description=f"CDR: {cdr:.2f}, Rim: {rim_area:.2f} mm2",
            confidence=min(0.90, 0.65 + image_quality * 0.25),
            referral_urgency=referrals.get(grade, "Routine"),
            icd10_code=icd10_codes.get(grade, "H40.10X0"),
            contributing_features=features
        )


class AMDGrader:
    """
    Age-related Macular Degeneration grading (AREDS-based).
    """
    
    def grade(
        self,
        biomarkers: Dict[str, float],
        image_quality: float = 0.8
    ) -> GradingResult:
        """Grade AMD from macular biomarkers."""
        drusen_count = biomarkers.get("drusen_count", 0)
        drusen_size = biomarkers.get("drusen_size_mm", 0)
        rpe_changes = biomarkers.get("rpe_changes", 0) > 0
        cnv_present = biomarkers.get("cnv", 0) > 0
        ga_present = biomarkers.get("geographic_atrophy", 0) > 0
        
        features = []
        
        # Wet AMD
        if cnv_present:
            grade = 4
            features.append("Choroidal neovascularization")
        # Advanced Dry AMD (GA)
        elif ga_present:
            grade = 3
            features.append("Geographic atrophy")
        # Intermediate AMD
        elif drusen_count > 20 or drusen_size > 0.125:
            grade = 2
            if drusen_count > 20:
                features.append(f"Large drusen count ({drusen_count})")
            if drusen_size > 0.125:
                features.append(f"Large drusen present")
            if rpe_changes:
                features.append("RPE changes")
        # Early AMD
        elif drusen_count > 5 or rpe_changes:
            grade = 1
            if drusen_count > 5:
                features.append(f"Drusen ({drusen_count})")
            if rpe_changes:
                features.append("RPE changes")
        else:
            grade = 0
        
        grade_names = {
            0: "No AMD",
            1: "Early AMD",
            2: "Intermediate AMD",
            3: "Advanced Dry AMD",
            4: "Wet AMD"
        }
        
        referrals = {
            0: "Routine 12 months",
            1: "Follow-up 6 months",
            2: "Refer 4 weeks",
            3: "Refer 2 weeks",
            4: "Urgent referral"
        }
        
        icd10_codes = {
            0: "H35.30",
            1: "H35.31",
            2: "H35.31",
            3: "H35.31",
            4: "H35.32"
        }
        
        return GradingResult(
            grade=grade,
            grade_name=grade_names.get(grade, "Unknown"),
            description=f"Drusen: {drusen_count}, RPE: {'Yes' if rpe_changes else 'No'}",
            confidence=min(0.85, 0.60 + image_quality * 0.25),
            referral_urgency=referrals.get(grade, "Routine"),
            icd10_code=icd10_codes.get(grade, "H35.30"),
            contributing_features=features
        )


# Singleton instances
dr_grader = DiabeticRetinopathyGrader()
glaucoma_grader = GlaucomaRiskGrader()
amd_grader = AMDGrader()
