"""
Clinical PDF Report Generator for Retinal Analysis

Generates comprehensive PDF clinical reports for retinal assessments:
- Patient demographics (Requirement 7.2)
- Biomarker values with reference ranges (Requirement 7.3)
- Risk assessment section (Requirement 7.4)
- Embedded visualizations (original, annotated, heatmap)
- Plain language interpretation (Requirement 7.5)
- Recommendations for follow-up (Requirement 7.6)
- Clinical literature references (Requirement 7.7)
- Disclaimer (Requirement 7.8)
- Report metadata (Requirement 7.9)
- Provider information (Requirement 7.10)
- Timestamp and report ID (Requirement 7.11)
- Digital signature placeholder (Requirement 7.12)

Author: NeuraLens Team
"""

import io
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether, ListFlowable, ListItem
)
from reportlab.graphics.shapes import Drawing, Line, Rect, Circle
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderPDF

from ..schemas import RetinalAnalysisResponse, RiskAssessment


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    page_size: tuple = letter
    margin_left: float = 0.75 * inch
    margin_right: float = 0.75 * inch
    margin_top: float = 0.5 * inch
    margin_bottom: float = 0.5 * inch
    include_visualizations: bool = True
    language: str = "en"  # Support for multi-language (en, es, zh)


class ReportGenerator:
    """
    Generates comprehensive clinical PDF reports for retinal analysis.
    
    Requirements: 7.1-7.12
    """
    
    # Reference ranges for biomarkers
    REFERENCE_RANGES = {
        "vessel_density": {"min": 4.0, "max": 7.0, "unit": "%", "name": "Vessel Density"},
        "tortuosity_index": {"min": 0.8, "max": 1.3, "unit": "", "name": "Vessel Tortuosity"},
        "avr_ratio": {"min": 0.6, "max": 0.8, "unit": "", "name": "A/V Ratio"},
        "cup_to_disc_ratio": {"min": 0.3, "max": 0.5, "unit": "", "name": "Cup-to-Disc Ratio"},
        "macular_thickness": {"min": 250, "max": 320, "unit": "μm", "name": "Macular Thickness"},
        "amyloid_presence": {"min": 0.0, "max": 0.2, "unit": "", "name": "Amyloid-β Presence"},
    }
    
    # Risk category descriptions
    RISK_DESCRIPTIONS = {
        "minimal": "Your neurological risk indicators are within normal limits. No concerning patterns were detected.",
        "low": "Your results show minor variations from typical ranges. These findings are generally not concerning but warrant monitoring.",
        "moderate": "Your results indicate some patterns that may benefit from closer monitoring. Consider discussing these findings with your healthcare provider.",
        "elevated": "Your results show patterns that warrant attention. We recommend consulting with a neurologist or ophthalmologist for further evaluation.",
        "high": "Your results indicate significant patterns that require prompt medical attention. Please schedule an appointment with a specialist soon.",
        "critical": "Your results show patterns of immediate concern. We strongly recommend urgent consultation with a neurologist.",
    }
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up custom paragraph styles"""
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#1a365d')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#2c5282')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.HexColor('#4a5568')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#718096'),
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='CriticalText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#c53030'),
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalRisk',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#276749'),
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(
        self,
        assessment: RetinalAnalysisResponse,
        patient_name: Optional[str] = None,
        patient_dob: Optional[str] = None,
        provider_name: Optional[str] = None,
        provider_npi: Optional[str] = None,
        original_image_data: Optional[bytes] = None,
        annotated_image_data: Optional[bytes] = None,
        heatmap_image_data: Optional[bytes] = None
    ) -> bytes:
        """
        Generate a comprehensive PDF clinical report.
        
        Args:
            assessment: RetinalAnalysisResponse with analysis results
            patient_name: Optional patient full name
            patient_dob: Optional patient date of birth
            provider_name: Optional healthcare provider name
            provider_npi: Optional provider NPI number
            original_image_data: Optional original image bytes
            annotated_image_data: Optional annotated image bytes
            heatmap_image_data: Optional heatmap image bytes
            
        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.config.page_size,
            leftMargin=self.config.margin_left,
            rightMargin=self.config.margin_right,
            topMargin=self.config.margin_top,
            bottomMargin=self.config.margin_bottom
        )
        
        story = []
        
        # Generate report ID
        report_id = self._generate_report_id(assessment.assessment_id)
        
        # 1. Header Section
        story.extend(self._build_header(assessment, report_id))
        
        # 2. Patient Demographics (Requirement 7.2)
        story.extend(self._build_patient_section(
            assessment, patient_name, patient_dob
        ))
        
        # 3. Executive Summary
        story.extend(self._build_executive_summary(assessment))
        
        # 4. Risk Assessment Section (Requirement 7.4)
        story.extend(self._build_risk_section(assessment))
        
        # 5. Biomarker Results (Requirement 7.3)
        story.extend(self._build_biomarker_section(assessment))
        
        # 6. Visualizations Section (if enabled)
        if self.config.include_visualizations:
            story.extend(self._build_visualization_section(
                original_image_data,
                annotated_image_data,
                heatmap_image_data
            ))
        
        # 7. Clinical Interpretation (Requirement 7.5)
        story.extend(self._build_interpretation_section(assessment))
        
        # 8. Recommendations (Requirement 7.6)
        story.extend(self._build_recommendations_section(assessment))
        
        # 9. References (Requirement 7.7)
        story.extend(self._build_references_section())
        
        # 10. Provider Information (Requirement 7.10)
        story.extend(self._build_provider_section(provider_name, provider_npi))
        
        # 11. Disclaimer (Requirement 7.8)
        story.extend(self._build_disclaimer_section())
        
        # 12. Digital Signature Placeholder (Requirement 7.12)
        story.extend(self._build_signature_section(report_id))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _generate_report_id(self, assessment_id: str) -> str:
        """Generate unique report ID (Requirement 7.11)"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{assessment_id}{timestamp}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8].upper()
        return f"NL-{timestamp}-{short_hash}"
    
    def _build_header(self, assessment: RetinalAnalysisResponse, report_id: str) -> List:
        """Build report header section"""
        story = []
        
        # Title
        story.append(Paragraph(
            "NeuroLens Clinical Assessment Report",
            self.styles['ReportTitle']
        ))
        
        # Subtitle with type
        story.append(Paragraph(
            "Retinal Fundus Analysis for Neurological Risk Assessment",
            self.styles['Normal']
        ))
        
        story.append(Spacer(1, 10))
        
        # Report metadata table
        meta_data = [
            ["Report ID:", report_id, "Assessment ID:", assessment.assessment_id],
            ["Date Generated:", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
             "Model Version:", assessment.model_version],
            ["Processing Time:", f"{assessment.processing_time_ms} ms",
             "Quality Score:", f"{assessment.quality_score:.1f}/100"],
        ]
        
        t = Table(meta_data, colWidths=[90, 150, 90, 150])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#4a5568')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        
        story.append(Spacer(1, 15))
        
        # Horizontal line separator
        story.append(self._create_separator())
        
        return story
    
    def _build_patient_section(
        self,
        assessment: RetinalAnalysisResponse,
        patient_name: Optional[str],
        patient_dob: Optional[str]
    ) -> List:
        """Build patient demographics section (Requirement 7.2)"""
        story = []
        
        story.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        
        data = [
            ["Patient ID:", assessment.patient_id],
        ]
        
        if patient_name:
            data.append(["Patient Name:", patient_name])
        if patient_dob:
            data.append(["Date of Birth:", patient_dob])
        
        data.append(["Analysis Date:", assessment.created_at.strftime("%Y-%m-%d %H:%M")])
        
        t = Table(data, colWidths=[100, 350])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_executive_summary(self, assessment: RetinalAnalysisResponse) -> List:
        """Build executive summary with key findings"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        risk = assessment.risk_assessment
        
        # Risk score highlight box
        risk_color = self._get_risk_color(risk.risk_category)
        
        summary_text = f"""
        This retinal analysis has identified a <b>{risk.risk_category.upper()}</b> 
        neurological risk level with a composite score of <b>{risk.risk_score:.1f}/100</b>
        (95% CI: {risk.confidence_interval[0]:.1f} - {risk.confidence_interval[1]:.1f}).
        """
        
        # Use appropriate style based on risk
        style = self.styles['CriticalText'] if risk.risk_score > 55 else self.styles['NormalRisk']
        story.append(Paragraph(summary_text, style))
        story.append(Spacer(1, 10))
        
        # Brief interpretation
        interpretation = self.RISK_DESCRIPTIONS.get(
            risk.risk_category.lower(),
            "Please consult with your healthcare provider about these results."
        )
        story.append(Paragraph(interpretation, self.styles['BodyText']))
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_risk_section(self, assessment: RetinalAnalysisResponse) -> List:
        """Build risk assessment section (Requirement 7.4)"""
        story = []
        
        story.append(Paragraph("Risk Assessment", self.styles['SectionHeader']))
        
        risk = assessment.risk_assessment
        
        # Risk score table
        data = [
            ["Overall Risk Score", f"{risk.risk_score:.1f} / 100"],
            ["Risk Category", risk.risk_category.upper()],
            ["Confidence Interval", f"{risk.confidence_interval[0]:.1f} - {risk.confidence_interval[1]:.1f}"],
        ]
        
        risk_color = self._get_risk_color(risk.risk_category)
        
        t = Table(data, colWidths=[150, 150])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (1, 1), (1, 1), risk_color),
            ('TEXTCOLOR', (1, 1), (1, 1), colors.white),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))
        
        # Contributing factors
        if risk.contributing_factors:
            story.append(Paragraph("Contributing Factors", self.styles['SubSection']))
            
            factors_data = [["Factor", "Risk Contribution"]]
            for factor, value in risk.contributing_factors.items():
                factor_name = factor.replace("_", " ").title()
                factors_data.append([factor_name, f"{value:.1f}/100"])
            
            t = Table(factors_data, colWidths=[200, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
        
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_biomarker_section(self, assessment: RetinalAnalysisResponse) -> List:
        """Build biomarker results section (Requirement 7.3)"""
        story = []
        
        story.append(Paragraph("Biomarker Analysis", self.styles['SectionHeader']))
        
        bio = assessment.biomarkers
        
        # Biomarker table with reference ranges
        data = [
            ["Biomarker", "Measured Value", "Reference Range", "Status", "Confidence"]
        ]
        
        biomarkers = [
            ("Vessel Density", f"{bio.vessels.density_percentage:.1f}%", "4.0-7.0%", 
             self._get_status(bio.vessels.density_percentage, 4.0, 7.0), f"{bio.vessels.confidence:.0%}"),
            ("Tortuosity Index", f"{bio.vessels.tortuosity_index:.2f}", "0.8-1.3",
             self._get_status(bio.vessels.tortuosity_index, 0.8, 1.3), "-"),
            ("A/V Ratio", f"{bio.vessels.avr_ratio:.2f}", "0.6-0.8",
             self._get_status(bio.vessels.avr_ratio, 0.6, 0.8), "-"),
            ("Cup-to-Disc Ratio", f"{bio.optic_disc.cup_to_disc_ratio:.2f}", "0.3-0.5",
             self._get_status(bio.optic_disc.cup_to_disc_ratio, 0.3, 0.5), f"{bio.optic_disc.confidence:.0%}"),
            ("Disc Area", f"{bio.optic_disc.disc_area_mm2:.2f} mm²", "2.0-3.5 mm²",
             self._get_status(bio.optic_disc.disc_area_mm2, 2.0, 3.5), "-"),
            ("Macular Thickness", f"{bio.macula.thickness_um:.0f} μm", "250-320 μm",
             self._get_status(bio.macula.thickness_um, 250, 320), f"{bio.macula.confidence:.0%}"),
            ("Amyloid-β Presence", f"{bio.amyloid_beta.presence_score:.2f}", "0.0-0.2",
             self._get_status(bio.amyloid_beta.presence_score, 0.0, 0.2), f"{bio.amyloid_beta.confidence:.0%}"),
        ]
        
        for b in biomarkers:
            data.append(list(b))
        
        t = Table(data, colWidths=[120, 90, 90, 70, 70])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
        ]))
        
        # Color-code status column
        for i, row in enumerate(data[1:], start=1):
            status = row[3]
            if status == "Normal":
                t.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#276749'))]))
            elif status == "High" or status == "Low":
                t.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#c53030'))]))
        
        story.append(t)
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_visualization_section(
        self,
        original: Optional[bytes],
        annotated: Optional[bytes],
        heatmap: Optional[bytes]
    ) -> List:
        """Build visualization section with images"""
        story = []
        
        if not any([original, annotated, heatmap]):
            return story
        
        story.append(Paragraph("Retinal Imaging", self.styles['SectionHeader']))
        
        # Note about images
        story.append(Paragraph(
            "The following images show the analyzed retinal fundus with automated annotations.",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 10))
        
        # Add images if provided
        images_data = []
        if original:
            images_data.append(("Original Image", original))
        if annotated:
            images_data.append(("Annotated Analysis", annotated))
        if heatmap:
            images_data.append(("Risk Heatmap", heatmap))
        
        for label, img_data in images_data:
            try:
                img_buffer = io.BytesIO(img_data)
                img = RLImage(img_buffer, width=3*inch, height=3*inch)
                story.append(Paragraph(f"<b>{label}</b>", self.styles['Normal']))
                story.append(img)
                story.append(Spacer(1, 10))
            except Exception as e:
                story.append(Paragraph(f"[{label} - Image could not be loaded]", self.styles['Normal']))
        
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_interpretation_section(self, assessment: RetinalAnalysisResponse) -> List:
        """Build clinical interpretation section (Requirement 7.5)"""
        story = []
        
        story.append(Paragraph("Clinical Interpretation", self.styles['SectionHeader']))
        
        bio = assessment.biomarkers
        risk = assessment.risk_assessment
        
        # Generate interpretation paragraphs
        interpretations = []
        
        # Vessel analysis
        vessel_status = "within normal limits" if 4.0 <= bio.vessels.density_percentage <= 7.0 else "outside normal range"
        interpretations.append(f"""
        <b>Vascular Analysis:</b> Retinal vessel density was measured at {bio.vessels.density_percentage:.1f}%, 
        which is {vessel_status}. The tortuosity index of {bio.vessels.tortuosity_index:.2f} 
        {"suggests normal vessel geometry" if 0.8 <= bio.vessels.tortuosity_index <= 1.3 else "may indicate vascular changes"}.
        The arteriovenous ratio of {bio.vessels.avr_ratio:.2f} 
        {"is within expected parameters" if 0.6 <= bio.vessels.avr_ratio <= 0.8 else "warrants attention"}.
        """)
        
        # Optic disc
        cdr = bio.optic_disc.cup_to_disc_ratio
        cdr_status = "normal" if 0.3 <= cdr <= 0.5 else ("elevated" if cdr > 0.5 else "below typical range")
        interpretations.append(f"""
        <b>Optic Nerve Assessment:</b> The cup-to-disc ratio was measured at {cdr:.2f}, 
        which is {cdr_status}. {"An elevated CDR may indicate increased optic nerve atrophy risk and warrants follow-up." if cdr > 0.5 else ""}
        """)
        
        # Amyloid-beta
        amyloid = bio.amyloid_beta.presence_score
        if amyloid > 0.3:
            interpretations.append(f"""
            <b>Amyloid-β Indicators:</b> Analysis detected potential amyloid-beta deposits with a score of {amyloid:.2f}. 
            The distribution pattern is classified as '{bio.amyloid_beta.distribution_pattern}'. 
            This finding may be relevant for neurodegenerative risk assessment and should be discussed with a specialist.
            """)
        else:
            interpretations.append(f"""
            <b>Amyloid-β Indicators:</b> No significant amyloid-beta indicators were detected (score: {amyloid:.2f}).
            """)
        
        for interp in interpretations:
            story.append(Paragraph(interp, self.styles['BodyText']))
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 10))
        
        return story
    
    def _build_recommendations_section(self, assessment: RetinalAnalysisResponse) -> List:
        """Build recommendations section (Requirement 7.6)"""
        story = []
        
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        
        risk = assessment.risk_assessment
        
        recommendations = []
        
        if risk.risk_score <= 25:
            recommendations = [
                "Continue routine eye health monitoring as recommended by your healthcare provider.",
                "Maintain a healthy lifestyle including regular exercise and balanced nutrition.",
                "Schedule your next retinal screening in 12-24 months.",
            ]
        elif risk.risk_score <= 55:
            recommendations = [
                "Schedule a follow-up retinal examination in 6-12 months.",
                "Consider comprehensive neurological evaluation if symptoms are present.",
                "Review cardiovascular health factors with your primary care provider.",
                "Monitor for any changes in vision or cognitive function.",
            ]
        else:
            recommendations = [
                "Schedule an appointment with an ophthalmologist within 30 days.",
                "Consider referral to a neurologist for comprehensive evaluation.",
                "Discuss additional diagnostic imaging (e.g., MRI, OCT) with your provider.",
                "Monitor closely for any neurological or visual symptoms.",
                "Review family history of neurodegenerative conditions.",
            ]
        
        # Create bullet list
        items = [ListItem(Paragraph(rec, self.styles['BodyText'])) for rec in recommendations]
        story.append(ListFlowable(items, bulletType='bullet', leftIndent=20))
        
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_references_section(self) -> List:
        """Build clinical literature references (Requirement 7.7)"""
        story = []
        
        story.append(Paragraph("Clinical References", self.styles['SectionHeader']))
        
        references = [
            "Cheung CY, et al. Retinal imaging in Alzheimer's disease. J Neurol Neurosurg Psychiatry. 2021.",
            "London A, et al. The retina as a window to the brain. Nat Rev Neurol. 2013.",
            "Patton N, et al. Retinal vascular image analysis as a potential screening tool. Br J Ophthalmol. 2006.",
            "Koronyo Y, et al. Retinal amyloid pathology in Alzheimer's disease. Neurobiol Aging. 2017.",
        ]
        
        for i, ref in enumerate(references, 1):
            story.append(Paragraph(f"[{i}] {ref}", 
                ParagraphStyle('Reference', parent=self.styles['Normal'], fontSize=8, leftIndent=20)))
        
        story.append(Spacer(1, 15))
        
        return story
    
    def _build_provider_section(
        self,
        provider_name: Optional[str],
        provider_npi: Optional[str]
    ) -> List:
        """Build provider information section (Requirement 7.10)"""
        story = []
        
        if provider_name or provider_npi:
            story.append(Paragraph("Provider Information", self.styles['SubSection']))
            
            data = []
            if provider_name:
                data.append(["Provider:", provider_name])
            if provider_npi:
                data.append(["NPI:", provider_npi])
            
            t = Table(data, colWidths=[80, 300])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))
        
        return story
    
    def _build_disclaimer_section(self) -> List:
        """Build disclaimer section (Requirement 7.8)"""
        story = []
        
        story.append(self._create_separator())
        story.append(Spacer(1, 10))
        
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b> This report is generated by NeuroLens AI-powered screening 
        technology and is intended for informational purposes only. It is NOT a definitive medical 
        diagnosis. The analysis utilizes machine learning algorithms trained on clinical data and 
        should be interpreted by a qualified healthcare professional. All findings require clinical 
        correlation and may necessitate additional testing. This report should not replace professional 
        medical advice, diagnosis, or treatment. Always seek the advice of your physician or other 
        qualified health provider with any questions regarding a medical condition.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))
        story.append(Spacer(1, 10))
        
        return story
    
    def _build_signature_section(self, report_id: str) -> List:
        """Build digital signature section (Requirement 7.12)"""
        story = []
        
        story.append(Spacer(1, 20))
        
        # Report hash for verification
        report_hash = hashlib.sha256(report_id.encode()).hexdigest()[:16]
        
        signature_text = f"""
        <b>Digital Verification</b><br/>
        Report ID: {report_id}<br/>
        Verification Hash: {report_hash}<br/>
        Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}<br/>
        System: NeuroLens Clinical Analysis Platform
        """
        
        story.append(Paragraph(signature_text, self.styles['Disclaimer']))
        
        return story
    
    def _create_separator(self) -> Table:
        """Create a horizontal line separator"""
        t = Table([["" * 100]], colWidths=[450])
        t.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#e2e8f0')),
        ]))
        return t
    
    def _get_risk_color(self, category: str) -> colors.Color:
        """Get color for risk category"""
        color_map = {
            "minimal": colors.HexColor('#22c55e'),
            "low": colors.HexColor('#84cc16'),
            "moderate": colors.HexColor('#eab308'),
            "elevated": colors.HexColor('#f97316'),
            "high": colors.HexColor('#ef4444'),
            "critical": colors.HexColor('#991b1b'),
        }
        return color_map.get(category.lower(), colors.HexColor('#6b7280'))
    
    def _get_status(self, value: float, min_val: float, max_val: float) -> str:
        """Determine status based on reference range"""
        if min_val <= value <= max_val:
            return "Normal"
        elif value < min_val:
            return "Low"
        else:
            return "High"


# Singleton instance
report_generator = ReportGenerator()
