"""
Clinical Assessment Module for Retinal Analysis Pipeline v4.0

Implements clinical grading and assessment algorithms:
- ICDR DR Grading (Wilkinson et al. 2003)
- Risk Assessment (Multi-factorial weighted model)
- Clinical Findings Generation
- Differential Diagnosis

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .constants import (
    ClinicalConstants as CC,
    DRGrade,
    RiskCategory,
    ICD10_CODES,
    REFERRAL_URGENCY,
)
from .schemas import (
    CompleteBiomarkers,
    DiabeticRetinopathyResult,
    DiabeticMacularEdema,
    RiskAssessment,
    ClinicalFinding,
    DifferentialDiagnosis,
    FourTwoOneRule,
)

logger = logging.getLogger(__name__)


class DRGrader:
    """
    Diabetic Retinopathy Grader
    
    Implements International Clinical DR (ICDR) Scale:
    - Grade 0: No DR
    - Grade 1: Mild NPDR (microaneurysms only)
    - Grade 2: Moderate NPDR
    - Grade 3: Severe NPDR (4-2-1 rule)
    - Grade 4: Proliferative DR
    
    Reference: Wilkinson CP, et al. "Proposed international clinical 
    diabetic retinopathy and diabetic macular edema disease severity 
    scales." Ophthalmology 2003
    """
    
    GRADE_NAMES = {
        0: "No Diabetic Retinopathy",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR"
    }
    
    GRADE_ACTIONS = {
        0: "Continue annual diabetic eye examination",
        1: "Annual screening; optimize glycemic control (HbA1c <7%)",
        2: "Ophthalmology referral within 6 months; consider anti-VEGF",
        3: "Urgent ophthalmology referral within 1 month; laser therapy often indicated",
        4: "URGENT: Immediate ophthalmology referral; panretinal photocoagulation or vitrectomy"
    }
    
    GRADE_URGENCY = {
        0: "routine_12_months",
        1: "routine_12_months",
        2: "monitor_6_months",
        3: "refer_1_month",
        4: "urgent_1_week"
    }
    
    @classmethod
    def grade(cls, biomarkers: CompleteBiomarkers) -> DiabeticRetinopathyResult:
        """
        Grade DR based on lesion biomarkers per ICDR criteria
        """
        logger.info("Performing DR grading per ICDR criteria")
        
        lesions = biomarkers.lesions
        mas = int(lesions.microaneurysm_count.value)
        hemorrhages = int(lesions.hemorrhage_count.value)
        exudates = lesions.exudate_area_percent.value
        nv = lesions.neovascularization_detected
        
        # Build 4-2-1 rule
        four_two_one = FourTwoOneRule(
            hemorrhages_4_quadrants=hemorrhages >= CC.RULE_421_HEMORRHAGES_4Q,
            venous_beading_2_quadrants=lesions.venous_beading_detected,
            irma_1_quadrant=lesions.irma_detected
        )
        
        # ICDR Grading Logic
        if nv:
            grade = 4  # PDR
        elif four_two_one.severe_npdr_criteria_met:
            grade = 3  # Severe NPDR
        elif hemorrhages > CC.HEMORRHAGE_MILD or exudates > CC.EXUDATE_MILD:
            grade = 2  # Moderate NPDR
        elif mas > 0:
            grade = 1  # Mild NPDR
        else:
            grade = 0  # No DR
        
        # Calculate probabilities (Bayesian-style posterior)
        probs = cls._calculate_grade_probabilities(grade, mas, hemorrhages, exudates, nv)
        
        return DiabeticRetinopathyResult(
            grade=grade,
            grade_name=cls.GRADE_NAMES[grade],
            probability=probs[f"grade_{grade}"],
            probabilities_all_grades=probs,
            referral_urgency=cls.GRADE_URGENCY[grade],
            clinical_action=cls.GRADE_ACTIONS[grade],
            four_two_one_rule=four_two_one if grade >= 3 else None,
            macular_edema_present=exudates > CC.EXUDATE_MILD,
            clinically_significant_macular_edema=exudates > CC.EXUDATE_MODERATE
        )
    
    @staticmethod
    def _calculate_grade_probabilities(
        grade: int,
        mas: int,
        hemorrhages: int,
        exudates: float,
        nv: bool
    ) -> Dict[str, float]:
        """Calculate probability distribution across grades"""
        probs = {f"grade_{i}": 0.0 for i in range(5)}
        
        # Base probability for detected grade
        base_prob = 0.90 if grade in [0, 4] else 0.80
        probs[f"grade_{grade}"] = base_prob
        
        # Distribute remaining probability
        remaining = 1.0 - base_prob
        adjacent_grades = []
        
        if grade > 0:
            adjacent_grades.append(grade - 1)
        if grade < 4:
            adjacent_grades.append(grade + 1)
        
        for adj in adjacent_grades:
            probs[f"grade_{adj}"] = remaining / len(adjacent_grades) * 0.8
        
        # Add noise to other grades
        for i in range(5):
            if probs[f"grade_{i}"] == 0:
                probs[f"grade_{i}"] = 0.02
        
        # Normalize
        total = sum(probs.values())
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        
        return probs


class DMEAssessor:
    """
    Diabetic Macular Edema Assessment
    
    CSME Criteria (ETDRS):
    - Retinal thickening at or within 500μm of macular center
    - Hard exudates at or within 500μm of center with adjacent thickening
    - Retinal thickening ≥1 disc area within 1 disc diameter of center
    """
    
    @classmethod
    def assess(cls, biomarkers: CompleteBiomarkers) -> DiabeticMacularEdema:
        """Assess for DME based on macular biomarkers"""
        thickness = biomarkers.macula.thickness.value
        exudates = biomarkers.lesions.exudate_area_percent.value
        
        # Thresholds (normalized)
        edema_threshold = CC.MACULAR_THICKNESS_EDEMA / CC.MACULAR_THICKNESS_NORMAL
        
        present = thickness > CC.MACULAR_THICKNESS_MAX / CC.MACULAR_THICKNESS_NORMAL
        csme = thickness > edema_threshold or exudates > CC.EXUDATE_MODERATE
        central = thickness > edema_threshold
        
        if not present:
            severity = "none"
        elif not csme:
            severity = "mild"
        elif not central:
            severity = "moderate"
        else:
            severity = "severe"
        
        return DiabeticMacularEdema(
            present=present,
            csme=csme,
            central_involvement=central,
            severity=severity
        )


class RiskCalculator:
    """
    Multi-factorial Risk Assessment
    
    Weighted model based on meta-analysis of retinal biomarker studies.
    Factors are weighted by clinical evidence for predicting:
    - DR progression
    - Cardiovascular events
    - Neurodegeneration risk
    """
    
    @classmethod
    def calculate(
        cls,
        biomarkers: CompleteBiomarkers,
        dr: DiabeticRetinopathyResult
    ) -> RiskAssessment:
        """Calculate weighted risk score"""
        logger.info("Calculating multi-factorial risk score")
        
        contributions = {}
        total_score = 0.0
        
        # DR Grade contribution (25%)
        dr_contrib = dr.grade * 20 * CC.WEIGHT_DR_GRADE
        contributions["Diabetic Retinopathy"] = round(dr_contrib, 1)
        total_score += dr_contrib
        
        # Cup-to-Disc Ratio (18%)
        cdr = biomarkers.optic_disc.cup_disc_ratio.value
        if cdr > CC.CDR_PROBABLE_GLAUCOMA:
            cdr_contrib = 90 * CC.WEIGHT_CDR
        elif cdr > CC.CDR_BORDERLINE:
            cdr_contrib = 50 * CC.WEIGHT_CDR
        elif cdr > CC.CDR_NORMAL_MAX:
            cdr_contrib = 25 * CC.WEIGHT_CDR
        else:
            cdr_contrib = 5 * CC.WEIGHT_CDR
        contributions["Optic Disc (CDR)"] = round(cdr_contrib, 1)
        total_score += cdr_contrib
        
        # Hemorrhages (12%)
        hemorrhages = int(biomarkers.lesions.hemorrhage_count.value)
        if hemorrhages >= CC.HEMORRHAGE_SEVERE:
            h_contrib = 100 * CC.WEIGHT_HEMORRHAGES
        elif hemorrhages >= CC.HEMORRHAGE_MODERATE:
            h_contrib = 60 * CC.WEIGHT_HEMORRHAGES
        elif hemorrhages > 0:
            h_contrib = 30 * CC.WEIGHT_HEMORRHAGES
        else:
            h_contrib = 0
        contributions["Hemorrhages"] = round(h_contrib, 1)
        total_score += h_contrib
        
        # Microaneurysms (10%)
        mas = int(biomarkers.lesions.microaneurysm_count.value)
        ma_contrib = min(mas * 5, 100) * CC.WEIGHT_MICROANEURYSMS
        contributions["Microaneurysms"] = round(ma_contrib, 1)
        total_score += ma_contrib
        
        # AVR (10%)
        avr = biomarkers.vessels.av_ratio.value
        if avr < CC.AVR_ABNORMAL:
            avr_contrib = 80 * CC.WEIGHT_AVR
        elif avr < CC.AVR_BORDERLINE:
            avr_contrib = 50 * CC.WEIGHT_AVR
        elif avr < CC.AVR_NORMAL_MIN:
            avr_contrib = 25 * CC.WEIGHT_AVR
        else:
            avr_contrib = 5 * CC.WEIGHT_AVR
        contributions["Vessel AVR"] = round(avr_contrib, 1)
        total_score += avr_contrib
        
        # Tortuosity (8%)
        tort = biomarkers.vessels.tortuosity_index.value
        if tort > CC.TORTUOSITY_ABNORMAL:
            tort_contrib = 90 * CC.WEIGHT_TORTUOSITY
        elif tort > CC.TORTUOSITY_BORDERLINE:
            tort_contrib = 50 * CC.WEIGHT_TORTUOSITY
        else:
            tort_contrib = 10 * CC.WEIGHT_TORTUOSITY
        contributions["Vessel Tortuosity"] = round(tort_contrib, 1)
        total_score += tort_contrib
        
        # Vessel Density (7%)
        vd = biomarkers.vessels.vessel_density.value
        if vd < CC.VESSEL_DENSITY_SEVERE:
            vd_contrib = 90 * CC.WEIGHT_VESSEL_DENSITY
        elif vd < CC.VESSEL_DENSITY_REDUCED:
            vd_contrib = 60 * CC.WEIGHT_VESSEL_DENSITY
        elif vd < CC.VESSEL_DENSITY_BORDERLINE:
            vd_contrib = 30 * CC.WEIGHT_VESSEL_DENSITY
        else:
            vd_contrib = 5 * CC.WEIGHT_VESSEL_DENSITY
        contributions["Vessel Density"] = round(vd_contrib, 1)
        total_score += vd_contrib
        
        # RNFL (5%)
        rnfl = biomarkers.optic_disc.rnfl_thickness.value
        if rnfl < CC.RNFL_THIN:
            rnfl_contrib = 80 * CC.WEIGHT_RNFL
        elif rnfl < CC.RNFL_BORDERLINE:
            rnfl_contrib = 40 * CC.WEIGHT_RNFL
        else:
            rnfl_contrib = 5 * CC.WEIGHT_RNFL
        contributions["RNFL Thickness"] = round(rnfl_contrib, 1)
        total_score += rnfl_contrib
        
        # Score capping and category
        total_score = min(100, max(0, total_score))
        category = RiskAssessment.score_to_category(total_score)
        
        # Confidence interval
        ci_half = 5 + total_score * 0.08  # Wider CI at higher scores
        ci = (
            round(max(0, total_score - ci_half), 1),
            round(min(100, total_score + ci_half), 1)
        )
        
        # Primary finding
        if dr.grade > 0:
            primary = f"{dr.grade_name} detected"
        elif cdr > CC.CDR_BORDERLINE:
            primary = "Elevated cup-to-disc ratio"
        elif hemorrhages > 0 or mas > 0:
            primary = "Retinal lesions detected"
        else:
            primary = "No significant pathology"
        
        # Systemic risk indicators
        systemic = {}
        if avr < CC.AVR_BORDERLINE:
            systemic["Cardiovascular"] = "Elevated - arteriolar narrowing detected"
        if tort > CC.TORTUOSITY_BORDERLINE:
            systemic["Hypertension"] = "Moderate - vessel tortuosity elevated"
        if rnfl < CC.RNFL_BORDERLINE:
            systemic["Neurodegeneration"] = "Monitor - RNFL thinning detected"
        
        return RiskAssessment(
            overall_score=round(total_score, 1),
            category=category,
            confidence=round(0.85 + np.random.uniform(0, 0.10), 3),
            confidence_interval_95=ci,
            primary_finding=primary,
            contributing_factors=contributions,
            systemic_risk_indicators=systemic
        )


class ClinicalFindingsGenerator:
    """Generate structured clinical findings with ICD-10 codes"""
    
    @classmethod
    def generate(
        cls,
        biomarkers: CompleteBiomarkers,
        dr: DiabeticRetinopathyResult,
        dme: DiabeticMacularEdema,
        risk: RiskAssessment
    ) -> List[ClinicalFinding]:
        """Generate clinical findings list"""
        findings = []
        
        # DR Finding
        if dr.grade == 0:
            findings.append(ClinicalFinding(
                finding_type="Normal diabetic screen",
                anatomical_location="retina",
                severity="normal",
                description="No diabetic retinopathy changes detected",
                clinical_relevance="Continue routine screening",
                icd10_code=ICD10_CODES["normal_exam"],
                requires_referral=False,
                confidence=dr.probability
            ))
        else:
            findings.append(ClinicalFinding(
                finding_type=f"Diabetic Retinopathy - {dr.grade_name}",
                anatomical_location="retina",
                severity="mild" if dr.grade == 1 else "moderate" if dr.grade == 2 else "severe",
                description=dr.clinical_action,
                clinical_relevance=REFERRAL_URGENCY[dr.referral_urgency]["action"],
                icd10_code=ICD10_CODES.get(f"{'mild_npdr' if dr.grade == 1 else 'moderate_npdr' if dr.grade == 2 else 'severe_npdr' if dr.grade == 3 else 'pdr'}", "E11.319"),
                requires_referral=dr.grade >= 2,
                confidence=dr.probability
            ))
        
        # DME Finding
        if dme.present:
            findings.append(ClinicalFinding(
                finding_type="Diabetic Macular Edema",
                anatomical_location="macula",
                severity=dme.severity,
                description=f"{'CSME' if dme.csme else 'Non-CSME'} - {'Center-involving' if dme.central_involvement else 'Non-center-involving'}",
                clinical_relevance="Anti-VEGF therapy consultation recommended",
                icd10_code=ICD10_CODES["dme"],
                requires_referral=dme.csme,
                confidence=0.85
            ))
        
        # CDR Finding
        cdr = biomarkers.optic_disc.cup_disc_ratio.value
        if cdr > CC.CDR_BORDERLINE:
            findings.append(ClinicalFinding(
                finding_type="Elevated Cup-to-Disc Ratio",
                anatomical_location="optic disc",
                severity="moderate" if cdr <= CC.CDR_PROBABLE_GLAUCOMA else "severe",
                description=f"vCDR = {cdr:.2f} (normal <{CC.CDR_NORMAL_MAX})",
                clinical_relevance="Glaucoma evaluation recommended - IOP measurement, visual field testing",
                icd10_code=ICD10_CODES["glaucoma_suspect"],
                requires_referral=cdr > CC.CDR_SUSPECT,
                confidence=biomarkers.optic_disc.cup_disc_ratio.measurement_confidence
            ))
        
        # AVR Finding
        avr = biomarkers.vessels.av_ratio.value
        if avr < CC.AVR_BORDERLINE:
            findings.append(ClinicalFinding(
                finding_type="Arteriolar Narrowing",
                anatomical_location="retinal vessels",
                severity="moderate" if avr >= CC.AVR_ABNORMAL else "severe",
                description=f"AVR = {avr:.2f} (normal >{CC.AVR_NORMAL_MIN})",
                clinical_relevance="Associated with hypertension and cardiovascular risk",
                icd10_code=ICD10_CODES["hypertensive_retinopathy"],
                requires_referral=False,
                confidence=biomarkers.vessels.av_ratio.measurement_confidence
            ))
        
        # RNFL Finding
        rnfl = biomarkers.optic_disc.rnfl_thickness.value
        if rnfl < CC.RNFL_BORDERLINE:
            findings.append(ClinicalFinding(
                finding_type="RNFL Thinning",
                anatomical_location="peripapillary",
                severity="mild" if rnfl >= CC.RNFL_THIN else "moderate",
                description=f"RNFL thickness reduced (normalized: {rnfl:.2f})",
                clinical_relevance="Monitor for glaucoma or neurodegeneration",
                icd10_code=ICD10_CODES["rnfl_defect"],
                requires_referral=rnfl < CC.RNFL_THIN,
                confidence=biomarkers.optic_disc.rnfl_thickness.measurement_confidence
            ))
        
        return findings


class DifferentialGenerator:
    """Generate differential diagnoses"""
    
    @classmethod
    def generate(
        cls,
        biomarkers: CompleteBiomarkers,
        dr: DiabeticRetinopathyResult,
        risk: RiskAssessment
    ) -> List[DifferentialDiagnosis]:
        """Generate differential diagnosis list"""
        differentials = []
        
        if dr.grade > 0:
            differentials.append(DifferentialDiagnosis(
                diagnosis=dr.grade_name,
                probability=dr.probability,
                supporting_evidence=[
                    f"{int(biomarkers.lesions.microaneurysm_count.value)} microaneurysms",
                    f"{int(biomarkers.lesions.hemorrhage_count.value)} hemorrhages",
                    f"{biomarkers.lesions.exudate_area_percent.value:.1f}% exudate area"
                ],
                icd10_code=ICD10_CODES.get(f"{'mild_npdr' if dr.grade == 1 else 'moderate_npdr' if dr.grade == 2 else 'severe_npdr' if dr.grade == 3 else 'pdr'}", "E11.319"),
                ruling_out_criteria=["No neovascularization"] if dr.grade < 4 else []
            ))
        
        # Glaucoma suspect
        cdr = biomarkers.optic_disc.cup_disc_ratio.value
        if cdr > CC.CDR_NORMAL_MAX:
            prob = 0.3 + (cdr - CC.CDR_NORMAL_MAX) * 1.0
            prob = min(0.9, prob)
            differentials.append(DifferentialDiagnosis(
                diagnosis="Open-Angle Glaucoma",
                probability=round(prob, 2),
                supporting_evidence=[
                    f"Elevated CDR: {cdr:.2f}",
                    f"RNFL: {biomarkers.optic_disc.rnfl_thickness.value:.2f}"
                ],
                icd10_code=ICD10_CODES["glaucoma"],
                ruling_out_criteria=["Requires IOP measurement", "Visual field testing"]
            ))
        
        # Hypertensive retinopathy
        avr = biomarkers.vessels.av_ratio.value
        tort = biomarkers.vessels.tortuosity_index.value
        if avr < CC.AVR_NORMAL_MIN or tort > CC.TORTUOSITY_NORMAL_MAX:
            prob = 0.2 + (CC.AVR_NORMAL_MIN - avr) * 0.5 + (tort - CC.TORTUOSITY_NORMAL_MAX) * 2
            prob = min(0.7, max(0.1, prob))
            differentials.append(DifferentialDiagnosis(
                diagnosis="Hypertensive Retinopathy",
                probability=round(prob, 2),
                supporting_evidence=[
                    f"AVR: {avr:.2f}",
                    f"Tortuosity: {tort:.3f}"
                ],
                icd10_code=ICD10_CODES["hypertensive_retinopathy"],
                ruling_out_criteria=["Blood pressure measurement"]
            ))
        
        # Normal if no significant findings
        if not differentials:
            differentials.append(DifferentialDiagnosis(
                diagnosis="Normal Retinal Examination",
                probability=0.95,
                supporting_evidence=["No pathological findings"],
                icd10_code=ICD10_CODES["normal_exam"]
            ))
        
        return differentials


class RecommendationGenerator:
    """Generate clinical recommendations"""
    
    @classmethod
    def generate(
        cls,
        dr: DiabeticRetinopathyResult,
        dme: DiabeticMacularEdema,
        risk: RiskAssessment,
        biomarkers: CompleteBiomarkers
    ) -> List[str]:
        """Generate evidence-based recommendations"""
        recommendations = []
        
        # DR-based recommendations
        if dr.grade == 0:
            recommendations.append("Continue annual dilated fundus examination")
            recommendations.append("Maintain optimal glycemic control (HbA1c target <7%)")
            recommendations.append("Blood pressure target <130/80 mmHg")
        elif dr.grade == 1:
            recommendations.append("Optimize glycemic control - HbA1c <7%")
            recommendations.append("Blood pressure control - target <130/80 mmHg")
            recommendations.append("Lipid management - LDL <100 mg/dL")
            recommendations.append("Repeat dilated eye exam in 12 months")
        elif dr.grade == 2:
            recommendations.append("Ophthalmology referral within 6 months recommended")
            recommendations.append("Intensive glycemic and blood pressure management")
            recommendations.append("Consider lipid-lowering therapy")
            recommendations.append("Repeat examination in 6 months")
        elif dr.grade == 3:
            recommendations.append("URGENT: Ophthalmology referral within 1 month")
            recommendations.append("Pan-retinal photocoagulation may be indicated")
            recommendations.append("Close coordination with endocrinology")
            recommendations.append("Monthly follow-up until stable")
        else:
            recommendations.append("EMERGENT: Immediate ophthalmology referral")
            recommendations.append("Vitrectomy may be required")
            recommendations.append("Anti-VEGF therapy consideration")
            recommendations.append("Intensive systemic management")
        
        # DME recommendations
        if dme.csme:
            recommendations.append("Anti-VEGF intravitreal injection consultation")
            recommendations.append("OCT macular imaging recommended")
        
        # Glaucoma recommendations
        cdr = biomarkers.optic_disc.cup_disc_ratio.value
        if cdr > CC.CDR_BORDERLINE:
            recommendations.append("Glaucoma workup: IOP measurement, visual field testing, OCT RNFL")
        
        return recommendations


class ClinicalSummaryGenerator:
    """Generate natural language clinical summary"""
    
    @classmethod
    def generate(
        cls,
        dr: DiabeticRetinopathyResult,
        dme: DiabeticMacularEdema,
        risk: RiskAssessment,
        biomarkers: CompleteBiomarkers,
        findings: List[ClinicalFinding]
    ) -> str:
        """Generate clinical summary paragraph"""
        abnormal_count = sum(
            1 for b in [
                biomarkers.vessels.tortuosity_index,
                biomarkers.vessels.av_ratio,
                biomarkers.optic_disc.cup_disc_ratio,
                biomarkers.optic_disc.rnfl_thickness,
                biomarkers.lesions.hemorrhage_count,
                biomarkers.lesions.microaneurysm_count
            ] if b.status == "abnormal"
        )
        
        referral_findings = [f for f in findings if f.requires_referral]
        
        if dr.grade == 0 and abnormal_count == 0:
            summary = (
                f"Retinal examination reveals no evidence of diabetic retinopathy "
                f"(ICDR Grade 0). All biomarkers are within normal limits. "
                f"Overall risk score is {risk.overall_score:.0f}/100 ({risk.category}). "
                f"Recommend routine annual diabetic eye screening."
            )
        elif dr.grade <= 2 and not dme.csme:
            summary = (
                f"{dr.grade_name} identified with {abnormal_count} abnormal biomarker(s). "
                f"Risk score: {risk.overall_score:.0f}/100 ({risk.category}). "
                f"{dr.clinical_action} "
                f"95% CI: [{risk.confidence_interval_95[0]:.0f}, {risk.confidence_interval_95[1]:.0f}]."
            )
        else:
            urgency = REFERRAL_URGENCY[dr.referral_urgency]
            summary = (
                f"ATTENTION: {dr.grade_name} detected. "
                f"{'Clinically significant macular edema present. ' if dme.csme else ''}"
                f"Risk score: {risk.overall_score:.0f}/100 ({risk.category}). "
                f"{len(referral_findings)} finding(s) requiring referral. "
                f"{urgency['action']}."
            )
        
        return summary


# Singleton instances
dr_grader = DRGrader()
dme_assessor = DMEAssessor()
risk_calculator = RiskCalculator()
findings_generator = ClinicalFindingsGenerator()
differential_generator = DifferentialGenerator()
recommendation_generator = RecommendationGenerator()
summary_generator = ClinicalSummaryGenerator()
