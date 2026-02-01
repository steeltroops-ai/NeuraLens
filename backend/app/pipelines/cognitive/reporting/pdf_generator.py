"""
Cognitive Report Generator - Research Grade PDF Export
Generates clinical-quality PDF reports for cognitive assessments.

Features:
- Summary page with risk gauge
- Domain breakdown with percentile charts
- Task details with quality indicators
- Recommendations section
- Clinical disclaimer

Dependencies:
- reportlab (PDF generation)
- matplotlib (charts)
"""

import io
from datetime import datetime
from typing import Optional, Dict, List
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable
    )
    from reportlab.graphics.shapes import Drawing, Circle, Rect
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..schemas import CognitiveResponse, CognitiveRiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


class CognitiveReportGenerator:
    """
    Generate PDF reports for cognitive assessment results.
    """
    
    RISK_COLORS = {
        RiskLevel.LOW: "#22c55e",      # Green
        RiskLevel.MODERATE: "#f59e0b",  # Amber
        RiskLevel.HIGH: "#f97316",      # Orange
        RiskLevel.CRITICAL: "#ef4444"   # Red
    }
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab not available - PDF generation disabled")
    
    def generate_pdf(
        self,
        response: CognitiveResponse,
        patient_info: Optional[Dict] = None,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Generate PDF report from cognitive assessment response.
        
        Args:
            response: CognitiveResponse from analysis
            patient_info: Optional patient demographics
            output_path: Optional file path to save PDF
            
        Returns:
            PDF bytes
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab not installed. Run: pip install reportlab")
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=12,
            textColor=colors.HexColor("#18181b")
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor("#27272a")
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14
        )
        
        # === PAGE 1: SUMMARY ===
        story.append(Paragraph("Cognitive Assessment Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Header info
        header_data = [
            ["Session ID:", response.session_id],
            ["Date:", response.timestamp[:10] if response.timestamp else datetime.now().strftime("%Y-%m-%d")],
            ["Pipeline Version:", response.pipeline_version],
            ["Processing Time:", f"{response.processing_time_ms:.0f} ms"]
        ]
        
        if patient_info:
            if patient_info.get("age"):
                header_data.append(["Age:", str(patient_info["age"])])
            if patient_info.get("education"):
                header_data.append(["Education:", f"{patient_info['education']} years"])
        
        header_table = Table(header_data, colWidths=[1.5*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#52525b")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Summary Box
        if response.risk_assessment:
            ra = response.risk_assessment
            risk_color = self.RISK_COLORS.get(ra.risk_level, "#6b7280")
            
            story.append(Paragraph("Overall Assessment", heading_style))
            
            risk_data = [
                [
                    f"Risk Score: {ra.overall_risk_score*100:.0f}/100",
                    f"Risk Level: {ra.risk_level.value.upper()}",
                    f"Confidence: {ra.confidence_score*100:.0f}%"
                ]
            ]
            
            risk_table = Table(risk_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f4f4f5")),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(risk_color)),
                ('ROUNDEDCORNERS', [5, 5, 5, 5]),
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Confidence interval
            ci_text = f"95% Confidence Interval: [{ra.confidence_interval[0]*100:.0f}, {ra.confidence_interval[1]*100:.0f}]"
            story.append(Paragraph(ci_text, body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Explainability Summary
        if response.explainability:
            story.append(Paragraph("Summary", heading_style))
            story.append(Paragraph(response.explainability.summary, body_style))
            
            if response.explainability.key_factors:
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph("<b>Key Findings:</b>", body_style))
                for factor in response.explainability.key_factors:
                    story.append(Paragraph(f"  - {factor}", body_style))
        
        # === DOMAIN BREAKDOWN ===
        if response.risk_assessment and response.risk_assessment.domain_risks:
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Domain Analysis", heading_style))
            
            domain_data = [["Domain", "Score", "Percentile", "Risk Level", "Confidence"]]
            
            for domain, detail in response.risk_assessment.domain_risks.items():
                domain_name = domain.replace("_", " ").title()
                domain_data.append([
                    domain_name,
                    f"{detail.score*100:.0f}",
                    f"{detail.percentile}th",
                    detail.risk_level.value.title(),
                    f"{detail.confidence*100:.0f}%"
                ])
            
            domain_table = Table(domain_data, colWidths=[1.8*inch, 0.8*inch, 1*inch, 1.2*inch, 1*inch])
            domain_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27272a")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#fafafa")),
            ]))
            story.append(domain_table)
        
        # === PAGE 2: TASK DETAILS ===
        story.append(PageBreak())
        story.append(Paragraph("Task Performance Details", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        if response.features and response.features.raw_metrics:
            for metric in response.features.raw_metrics:
                # Task header
                story.append(Paragraph(f"<b>{metric.task_id.replace('_', ' ').title()}</b>", heading_style))
                
                # Status badge
                status_color = "#22c55e" if metric.validity_flag else "#ef4444"
                status_text = f"Score: {metric.performance_score:.1f}/100 | Status: {metric.completion_status.value}"
                story.append(Paragraph(status_text, body_style))
                
                # Parameters table
                if metric.parameters:
                    param_data = []
                    for key, value in list(metric.parameters.items())[:8]:
                        key_display = key.replace("_", " ").title()
                        if isinstance(value, float):
                            value_display = f"{value:.2f}"
                        else:
                            value_display = str(value)
                        param_data.append([key_display, value_display])
                    
                    if param_data:
                        # Display in two columns
                        mid = len(param_data) // 2
                        left_data = param_data[:mid] if mid > 0 else param_data
                        right_data = param_data[mid:] if mid > 0 else []
                        
                        combined_data = []
                        for i in range(max(len(left_data), len(right_data))):
                            row = []
                            if i < len(left_data):
                                row.extend(left_data[i])
                            else:
                                row.extend(["", ""])
                            if i < len(right_data):
                                row.extend(right_data[i])
                            else:
                                row.extend(["", ""])
                            combined_data.append(row)
                        
                        param_table = Table(combined_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch])
                        param_table.setStyle(TableStyle([
                            ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
                            ('FONTNAME', (2, 0), (2, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 0), (-1, -1), 9),
                            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#52525b")),
                            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor("#52525b")),
                            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
                            ('PADDING', (0, 0), (-1, -1), 4),
                        ]))
                        story.append(param_table)
                
                # Quality warnings
                if metric.quality_warnings:
                    story.append(Spacer(1, 0.1*inch))
                    for warning in metric.quality_warnings:
                        story.append(Paragraph(
                            f"<font color='#f59e0b'>! {warning}</font>",
                            body_style
                        ))
                
                story.append(Spacer(1, 0.2*inch))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e4e4e7")))
        
        # === PAGE 3: RECOMMENDATIONS ===
        if response.recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            for rec in response.recommendations:
                priority_colors = {
                    "low": "#22c55e",
                    "medium": "#f59e0b",
                    "high": "#f97316",
                    "critical": "#ef4444"
                }
                color = priority_colors.get(rec.priority, "#6b7280")
                
                story.append(Paragraph(
                    f"<font color='{color}'><b>[{rec.priority.upper()}]</b></font> "
                    f"<b>{rec.category.title()}</b>",
                    body_style
                ))
                story.append(Paragraph(rec.description, body_style))
                story.append(Spacer(1, 0.1*inch))
        
        # === DISCLAIMER ===
        story.append(Spacer(1, 0.5*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d4d4d8")))
        story.append(Spacer(1, 0.1*inch))
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor("#71717a"),
            spaceAfter=4
        )
        
        disclaimer = """
        <b>CLINICAL DISCLAIMER:</b> This cognitive screening tool is designed for research and wellness 
        monitoring purposes. It is NOT a diagnostic device and should not replace professional 
        neuropsychological evaluation. Results may be affected by fatigue, medication, environmental 
        factors, or device latency. Consult a qualified healthcare professional for clinical interpretation.
        """
        story.append(Paragraph(disclaimer, disclaimer_style))
        
        # Methodology note
        if response.explainability and response.explainability.methodology_note:
            story.append(Paragraph(
                f"<b>Methodology:</b> {response.explainability.methodology_note}",
                disclaimer_style
            ))
        
        # Build PDF
        doc.build(story)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Optionally save to file
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        return pdf_bytes
    
    def generate_risk_gauge_image(
        self,
        risk_score: float,
        risk_level: RiskLevel
    ) -> Optional[bytes]:
        """
        Generate a risk gauge chart as PNG bytes.
        
        Returns None if matplotlib is not available.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(3, 1.5), subplot_kw={'projection': 'polar'})
        
        # Gauge parameters
        theta = np.linspace(np.pi, 0, 100)
        r = np.ones(100)
        
        # Color gradient
        cmap = plt.cm.RdYlGn_r
        colors_list = [cmap(i/100) for i in range(100)]
        
        # Draw gauge background
        for i in range(99):
            ax.fill_between([theta[i], theta[i+1]], 0, 1, 
                          color=colors_list[i], alpha=0.3)
        
        # Draw needle
        needle_angle = np.pi * (1 - risk_score)
        ax.plot([needle_angle, needle_angle], [0, 0.8], 
               color='#18181b', linewidth=2)
        ax.scatter([needle_angle], [0.8], c='#18181b', s=20, zorder=5)
        
        # Center circle
        ax.scatter([0], [0], c='white', s=100, zorder=4)
        ax.scatter([0], [0], c='#18181b', s=50, zorder=5)
        
        # Styling
        ax.set_ylim(0, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        # Convert to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   transparent=True, dpi=150)
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()


# Singleton instance
report_generator = CognitiveReportGenerator()


def generate_cognitive_report(
    response: CognitiveResponse,
    patient_info: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> bytes:
    """
    Convenience function to generate PDF report.
    
    Args:
        response: CognitiveResponse from analysis
        patient_info: Optional dict with 'age', 'education', etc.
        output_path: Optional file path to save PDF
        
    Returns:
        PDF bytes
    """
    return report_generator.generate_pdf(response, patient_info, output_path)
