// NeuroLens-X PDF Report Generator
// Clinical-grade PDF reports for healthcare providers

import type { CompleteAssessmentResult } from '@/lib/ml';

export interface PDFReportOptions {
  includeCharts: boolean;
  includeDetailedFindings: boolean;
  includeRecommendations: boolean;
  includeTechnicalDetails: boolean;
  format: 'clinical' | 'patient' | 'research';
}

export interface PDFReportData {
  patientInfo?: {
    name?: string;
    dateOfBirth?: string;
    patientId?: string;
    gender?: string;
  };
  providerInfo?: {
    name?: string;
    facility?: string;
    contact?: string;
  };
  assessmentResults: CompleteAssessmentResult;
  options: PDFReportOptions;
}

export class PDFReportGenerator {
  private readonly version = '1.0.0';
  private readonly reportTitle = 'NeuroLens-X Neurological Risk Assessment Report';

  /**
   * Generate clinical PDF report
   */
  async generatePDFReport(data: PDFReportData): Promise<Blob> {
    try {
      // In a real implementation, this would use a PDF library like jsPDF or PDFKit
      // For now, we'll create a comprehensive HTML report that can be printed to PDF
      const htmlContent = this.generateHTMLReport(data);
      
      // Convert HTML to PDF (simplified simulation)
      const pdfBlob = await this.htmlToPDF(htmlContent);
      
      return pdfBlob;
    } catch (error) {
      console.error('PDF generation failed:', error);
      throw new Error('Failed to generate PDF report: ' + (error as Error).message);
    }
  }

  /**
   * Generate HTML report content
   */
  private generateHTMLReport(data: PDFReportData): string {
    const { assessmentResults, options, patientInfo, providerInfo } = data;
    const { nriResult, modalityResults, metadata } = assessmentResults;

    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${this.reportTitle}</title>
    <style>
        ${this.getReportStyles()}
    </style>
</head>
<body>
    <div class="report-container">
        ${this.generateReportHeader(patientInfo, providerInfo, metadata)}
        ${this.generateExecutiveSummary(nriResult)}
        ${options.includeCharts ? this.generateScoreVisualization(nriResult) : ''}
        ${this.generateModalityResults(modalityResults, options)}
        ${options.includeRecommendations ? this.generateRecommendations(nriResult) : ''}
        ${options.includeTechnicalDetails ? this.generateTechnicalDetails(assessmentResults) : ''}
        ${this.generateReportFooter(metadata)}
    </div>
</body>
</html>`;
  }

  /**
   * Generate report header
   */
  private generateReportHeader(
    patientInfo?: PDFReportData['patientInfo'],
    providerInfo?: PDFReportData['providerInfo'],
    metadata?: CompleteAssessmentResult['metadata']
  ): string {
    return `
    <header class="report-header">
        <div class="header-content">
            <div class="logo-section">
                <h1>NeuroLens-X</h1>
                <p class="subtitle">Neurological Risk Assessment Platform</p>
            </div>
            <div class="report-info">
                <h2>${this.reportTitle}</h2>
                <p class="report-date">Generated: ${new Date().toLocaleDateString()}</p>
                <p class="report-version">Version: ${this.version}</p>
            </div>
        </div>
        
        ${patientInfo ? `
        <div class="patient-info">
            <h3>Patient Information</h3>
            <div class="info-grid">
                ${patientInfo.name ? `<div><strong>Name:</strong> ${patientInfo.name}</div>` : ''}
                ${patientInfo.dateOfBirth ? `<div><strong>Date of Birth:</strong> ${patientInfo.dateOfBirth}</div>` : ''}
                ${patientInfo.patientId ? `<div><strong>Patient ID:</strong> ${patientInfo.patientId}</div>` : ''}
                ${patientInfo.gender ? `<div><strong>Gender:</strong> ${patientInfo.gender}</div>` : ''}
            </div>
        </div>
        ` : ''}
        
        ${providerInfo ? `
        <div class="provider-info">
            <h3>Healthcare Provider</h3>
            <div class="info-grid">
                ${providerInfo.name ? `<div><strong>Provider:</strong> ${providerInfo.name}</div>` : ''}
                ${providerInfo.facility ? `<div><strong>Facility:</strong> ${providerInfo.facility}</div>` : ''}
                ${providerInfo.contact ? `<div><strong>Contact:</strong> ${providerInfo.contact}</div>` : ''}
            </div>
        </div>
        ` : ''}
    </header>`;
  }

  /**
   * Generate executive summary
   */
  private generateExecutiveSummary(nriResult: CompleteAssessmentResult['nriResult']): string {
    const riskCategoryInfo = this.getRiskCategoryInfo(nriResult.riskCategory);
    
    return `
    <section class="executive-summary">
        <h2>Executive Summary</h2>
        
        <div class="summary-grid">
            <div class="nri-score-box ${riskCategoryInfo.className}">
                <h3>Neuro-Risk Index (NRI)</h3>
                <div class="score-display">
                    <span class="score-number">${nriResult.nriScore}</span>
                    <span class="score-label">/ 100</span>
                </div>
                <div class="risk-category">
                    <span class="risk-badge ${riskCategoryInfo.className}">
                        ${riskCategoryInfo.label}
                    </span>
                </div>
            </div>
            
            <div class="confidence-box">
                <h4>Assessment Confidence</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${nriResult.confidence}%"></div>
                </div>
                <p>${nriResult.confidence}% Confidence</p>
            </div>
            
            <div class="completeness-box">
                <h4>Data Completeness</h4>
                <div class="completeness-bar">
                    <div class="completeness-fill" style="width: ${nriResult.dataCompleteness}%"></div>
                </div>
                <p>${nriResult.dataCompleteness}% Complete</p>
            </div>
        </div>
        
        <div class="key-findings">
            <h3>Key Clinical Findings</h3>
            <ul>
                ${nriResult.clinicalNotes.slice(0, 5).map(note => `<li>${note}</li>`).join('')}
            </ul>
        </div>
        
        ${nriResult.uncertaintyFactors.length > 0 ? `
        <div class="uncertainty-factors">
            <h3>Uncertainty Factors</h3>
            <ul class="uncertainty-list">
                ${nriResult.uncertaintyFactors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
    </section>`;
  }

  /**
   * Generate score visualization
   */
  private generateScoreVisualization(nriResult: CompleteAssessmentResult['nriResult']): string {
    return `
    <section class="score-visualization">
        <h2>Risk Assessment Visualization</h2>
        
        <div class="risk-scale">
            <h3>Risk Scale Position</h3>
            <div class="scale-container">
                <div class="scale-bar">
                    <div class="scale-segments">
                        <div class="segment low">Low (0-25)</div>
                        <div class="segment moderate">Moderate (26-50)</div>
                        <div class="segment high">High (51-75)</div>
                        <div class="segment critical">Critical (76-100)</div>
                    </div>
                    <div class="score-indicator" style="left: ${nriResult.nriScore}%">
                        <div class="indicator-arrow"></div>
                        <div class="indicator-label">${nriResult.nriScore}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="modality-contributions">
            <h3>Assessment Modality Contributions</h3>
            <div class="contribution-chart">
                ${Object.entries(nriResult.modalityContributions).map(([modality, contribution]) => {
                  const percentage = Math.round(contribution * 100);
                  const modalityInfo = this.getModalityInfo(modality);
                  return `
                    <div class="contribution-item">
                        <div class="modality-label">
                            <span class="modality-icon">${modalityInfo.icon}</span>
                            <span class="modality-name">${modalityInfo.name}</span>
                        </div>
                        <div class="contribution-bar">
                            <div class="contribution-fill" style="width: ${percentage}%"></div>
                        </div>
                        <div class="contribution-percentage">${percentage}%</div>
                    </div>
                  `;
                }).join('')}
            </div>
        </div>
    </section>`;
  }

  /**
   * Generate modality results
   */
  private generateModalityResults(
    modalityResults: CompleteAssessmentResult['modalityResults'],
    options: PDFReportOptions
  ): string {
    return `
    <section class="modality-results">
        <h2>Detailed Modality Analysis</h2>
        
        ${modalityResults.speech ? this.generateSpeechResults(modalityResults.speech, options) : ''}
        ${modalityResults.retinal ? this.generateRetinalResults(modalityResults.retinal, options) : ''}
        ${modalityResults.risk ? this.generateRiskResults(modalityResults.risk, options) : ''}
    </section>`;
  }

  /**
   * Generate speech analysis results
   */
  private generateSpeechResults(speechResult: any, options: PDFReportOptions): string {
    return `
    <div class="modality-section">
        <h3>🎤 Speech Analysis</h3>
        <div class="modality-summary">
            <div class="score-box">
                <strong>Risk Score:</strong> ${speechResult.riskScore}/100
            </div>
            <div class="confidence-box">
                <strong>Confidence:</strong> ${speechResult.confidence}%
            </div>
            <div class="quality-box">
                <strong>Audio Quality:</strong> ${Math.round(speechResult.qualityScore * 100)}%
            </div>
        </div>
        
        <div class="findings">
            <h4>Clinical Findings</h4>
            <ul>
                ${speechResult.findings.map((finding: string) => `<li>${finding}</li>`).join('')}
            </ul>
        </div>
        
        ${options.includeTechnicalDetails ? `
        <div class="technical-details">
            <h4>Technical Parameters</h4>
            <div class="parameter-grid">
                <div><strong>Speech Rate:</strong> ${speechResult.features.speechRate} wpm</div>
                <div><strong>Pause Duration:</strong> ${speechResult.features.pauseDuration} ms</div>
                <div><strong>Voice Tremor:</strong> ${speechResult.features.voiceTremor}</div>
                <div><strong>Jitter:</strong> ${speechResult.features.jitter}</div>
            </div>
        </div>
        ` : ''}
    </div>`;
  }

  /**
   * Generate retinal analysis results
   */
  private generateRetinalResults(retinalResult: any, options: PDFReportOptions): string {
    return `
    <div class="modality-section">
        <h3>👁️ Retinal Analysis</h3>
        <div class="modality-summary">
            <div class="score-box">
                <strong>Risk Score:</strong> ${retinalResult.riskScore}/100
            </div>
            <div class="confidence-box">
                <strong>Confidence:</strong> ${retinalResult.confidence}%
            </div>
            <div class="quality-box">
                <strong>Image Quality:</strong> ${Math.round(retinalResult.imageQuality * 100)}%
            </div>
        </div>
        
        <div class="findings">
            <h4>Clinical Findings</h4>
            <ul>
                ${retinalResult.findings.map((finding: string) => `<li>${finding}</li>`).join('')}
            </ul>
        </div>
        
        ${options.includeTechnicalDetails ? `
        <div class="technical-details">
            <h4>Technical Parameters</h4>
            <div class="parameter-grid">
                <div><strong>Vessel Density:</strong> ${retinalResult.features.vesselDensity}%</div>
                <div><strong>Cup-Disc Ratio:</strong> ${retinalResult.features.cupDiscRatio}</div>
                <div><strong>RNFL Thickness:</strong> ${retinalResult.features.retinalNerveLayer} μm</div>
                <div><strong>Vessel Tortuosity:</strong> ${retinalResult.features.vesselTortuosity}</div>
            </div>
        </div>
        ` : ''}
    </div>`;
  }

  /**
   * Generate risk assessment results
   */
  private generateRiskResults(riskResult: any, options: PDFReportOptions): string {
    return `
    <div class="modality-section">
        <h3>📊 Risk Assessment</h3>
        <div class="modality-summary">
            <div class="score-box">
                <strong>Overall Risk:</strong> ${riskResult.overallRisk}/100
            </div>
            <div class="confidence-box">
                <strong>Confidence:</strong> ${riskResult.confidence}%
            </div>
        </div>
        
        <div class="risk-categories">
            <h4>Risk Category Breakdown</h4>
            <div class="category-grid">
                ${Object.entries(riskResult.categoryRisks).map(([category, score]) => `
                  <div class="category-item">
                    <span class="category-name">${category.charAt(0).toUpperCase() + category.slice(1)}:</span>
                    <span class="category-score">${score}/100</span>
                  </div>
                `).join('')}
            </div>
        </div>
        
        <div class="modifiable-factors">
            <h4>Modifiable Risk Factors</h4>
            <ul>
                ${riskResult.modifiableFactors.map((factor: string) => `<li>${factor}</li>`).join('')}
            </ul>
        </div>
    </div>`;
  }

  /**
   * Generate recommendations
   */
  private generateRecommendations(nriResult: CompleteAssessmentResult['nriResult']): string {
    return `
    <section class="recommendations">
        <h2>Clinical Recommendations</h2>
        
        <div class="recommendation-list">
            ${nriResult.recommendations.map((recommendation, index) => `
                <div class="recommendation-item">
                    <div class="recommendation-number">${index + 1}</div>
                    <div class="recommendation-text">${recommendation}</div>
                </div>
            `).join('')}
        </div>
        
        <div class="follow-up">
            <h3>Follow-up Guidelines</h3>
            <p>Based on the ${nriResult.riskCategory} risk category, the following follow-up schedule is recommended:</p>
            <ul>
                ${this.getFollowUpRecommendations(nriResult.riskCategory).map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    </section>`;
  }

  /**
   * Generate technical details
   */
  private generateTechnicalDetails(assessmentResults: CompleteAssessmentResult): string {
    return `
    <section class="technical-details">
        <h2>Technical Details</h2>
        
        <div class="processing-info">
            <h3>Processing Information</h3>
            <div class="info-grid">
                <div><strong>Session ID:</strong> ${assessmentResults.sessionId}</div>
                <div><strong>Processing Time:</strong> ${assessmentResults.metadata.totalProcessingTime}ms</div>
                <div><strong>Platform Version:</strong> ${assessmentResults.metadata.version}</div>
                <div><strong>Data Quality:</strong> ${Math.round(assessmentResults.metadata.dataQuality * 100)}%</div>
            </div>
        </div>
        
        <div class="algorithm-info">
            <h3>Algorithm Information</h3>
            <p>This assessment was performed using NeuroLens-X multi-modal fusion algorithms:</p>
            <ul>
                <li><strong>Speech Analysis:</strong> Voice biomarker detection with tremor analysis</li>
                <li><strong>Retinal Analysis:</strong> Vascular pattern recognition and pathology detection</li>
                <li><strong>Risk Assessment:</strong> Comprehensive demographic and lifestyle factor analysis</li>
                <li><strong>NRI Fusion:</strong> Weighted ensemble with uncertainty quantification</li>
            </ul>
        </div>
    </section>`;
  }

  /**
   * Generate report footer
   */
  private generateReportFooter(metadata: CompleteAssessmentResult['metadata']): string {
    return `
    <footer class="report-footer">
        <div class="disclaimer">
            <h3>Important Medical Disclaimer</h3>
            <p>
                This assessment is a screening tool and not a diagnostic device. Results should not replace 
                professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
                professionals for medical decisions. If you have immediate health concerns, contact your 
                healthcare provider or emergency services.
            </p>
        </div>
        
        <div class="footer-info">
            <div class="generation-info">
                <p>Report generated on ${new Date().toLocaleString()}</p>
                <p>NeuroLens-X Platform Version ${metadata.version}</p>
            </div>
            <div class="contact-info">
                <p>For technical support: support@neurolens-x.com</p>
                <p>For clinical questions: clinical@neurolens-x.com</p>
            </div>
        </div>
    </footer>`;
  }

  /**
   * Get CSS styles for the report
   */
  private getReportStyles(): string {
    return `
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: white;
        }
        
        .report-container {
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: white;
        }
        
        .report-header {
            border-bottom: 3px solid #3B82F6;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        
        .logo-section h1 {
            color: #3B82F6;
            font-size: 28px;
            font-weight: bold;
        }
        
        .subtitle {
            color: #666;
            font-size: 14px;
        }
        
        .report-info {
            text-align: right;
        }
        
        .report-info h2 {
            font-size: 20px;
            color: #333;
            margin-bottom: 5px;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        
        .executive-summary {
            margin-bottom: 40px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .nri-score-box {
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .score-display {
            margin: 15px 0;
        }
        
        .score-number {
            font-size: 48px;
            font-weight: bold;
            color: #3B82F6;
        }
        
        .score-label {
            font-size: 18px;
            color: #666;
        }
        
        .risk-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        
        .risk-badge.low { background: #dcfce7; color: #166534; }
        .risk-badge.moderate { background: #fef3c7; color: #92400e; }
        .risk-badge.high { background: #fed7aa; color: #c2410c; }
        .risk-badge.critical { background: #fecaca; color: #dc2626; }
        
        .confidence-box, .completeness-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
        }
        
        .confidence-bar, .completeness-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: #3B82F6;
            transition: width 0.3s ease;
        }
        
        .completeness-fill {
            height: 100%;
            background: #10b981;
            transition: width 0.3s ease;
        }
        
        .modality-section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
        
        .modality-summary {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 15px 0;
        }
        
        .score-box, .confidence-box, .quality-box {
            background: #f8fafc;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .parameter-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        
        .recommendations {
            margin: 40px 0;
        }
        
        .recommendation-item {
            display: flex;
            align-items: flex-start;
            margin: 15px 0;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
        }
        
        .recommendation-number {
            background: #3B82F6;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }
        
        .report-footer {
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #e2e8f0;
        }
        
        .disclaimer {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .disclaimer h3 {
            color: #92400e;
            margin-bottom: 10px;
        }
        
        .footer-info {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
        }
        
        @media print {
            .report-container {
                margin: 0;
                padding: 15mm;
            }
            
            .modality-section {
                page-break-inside: avoid;
            }
            
            .recommendations {
                page-break-inside: avoid;
            }
        }
        
        h2 {
            color: #1f2937;
            font-size: 24px;
            margin: 30px 0 15px 0;
            border-bottom: 2px solid #3B82F6;
            padding-bottom: 5px;
        }
        
        h3 {
            color: #374151;
            font-size: 18px;
            margin: 20px 0 10px 0;
        }
        
        h4 {
            color: #4b5563;
            font-size: 16px;
            margin: 15px 0 8px 0;
        }
        
        ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        li {
            margin: 5px 0;
        }
    `;
  }

  /**
   * Convert HTML to PDF (simplified simulation)
   */
  private async htmlToPDF(htmlContent: string): Promise<Blob> {
    // In a real implementation, this would use a library like Puppeteer, jsPDF, or PDFKit
    // For now, we'll return the HTML as a blob that can be printed to PDF
    return new Blob([htmlContent], { type: 'text/html' });
  }

  /**
   * Get risk category information
   */
  private getRiskCategoryInfo(category: string) {
    switch (category) {
      case 'low':
        return { label: 'Low Risk', className: 'low' };
      case 'moderate':
        return { label: 'Moderate Risk', className: 'moderate' };
      case 'high':
        return { label: 'High Risk', className: 'high' };
      case 'critical':
        return { label: 'Critical Risk', className: 'critical' };
      default:
        return { label: 'Unknown Risk', className: 'unknown' };
    }
  }

  /**
   * Get modality information
   */
  private getModalityInfo(modality: string) {
    const modalityMap = {
      speech: { name: 'Speech Analysis', icon: '🎤' },
      retinal: { name: 'Retinal Analysis', icon: '👁️' },
      risk: { name: 'Risk Assessment', icon: '📊' },
      motor: { name: 'Motor Assessment', icon: '🤲' },
    };
    return modalityMap[modality as keyof typeof modalityMap] || { name: modality, icon: '❓' };
  }

  /**
   * Get follow-up recommendations based on risk category
   */
  private getFollowUpRecommendations(riskCategory: string): string[] {
    switch (riskCategory) {
      case 'low':
        return [
          'Annual neurological screening',
          'Maintain healthy lifestyle practices',
          'Monitor for any new symptoms',
        ];
      case 'moderate':
        return [
          'Follow-up assessment in 6 months',
          'Regular monitoring with healthcare provider',
          'Implement lifestyle interventions',
          'Annual cognitive screening',
        ];
      case 'high':
        return [
          'Neurological consultation within 3 months',
          'Quarterly monitoring assessments',
          'Comprehensive cognitive evaluation',
          'Aggressive risk factor modification',
        ];
      case 'critical':
        return [
          'Immediate neurological evaluation',
          'Monthly monitoring assessments',
          'Comprehensive diagnostic workup',
          'Urgent intervention planning',
        ];
      default:
        return ['Consult with healthcare provider for appropriate follow-up'];
    }
  }
}

// Export singleton instance
export const pdfReportGenerator = new PDFReportGenerator();
