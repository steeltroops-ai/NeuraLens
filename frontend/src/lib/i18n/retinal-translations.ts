/**
 * Retinal Analysis Translations
 * 
 * Comprehensive translation keys for the retinal analysis module.
 * Supports: English (en), Spanish (es), Chinese (zh)
 * 
 * Requirements: 14.1-14.6
 * 
 * @module lib/i18n/retinal
 */

// ============================================================================
// Types
// ============================================================================

export type SupportedLocale = 'en' | 'es' | 'zh';

export interface RetinalTranslations {
  // Page Headers
  pageTitle: string;
  pageDescription: string;
  modelVersion: string;
  
  // Features
  features: {
    processing: string;
    aiPowered: string;
    biomarkers: string;
    hipaaCompliant: string;
  };
  
  // Upload
  upload: {
    title: string;
    description: string;
    dragDrop: string;
    browse: string;
    acceptedFormats: string;
    maxSize: string;
    selectImage: string;
    uploading: string;
    processing: string;
  };
  
  // Validation
  validation: {
    checking: string;
    success: string;
    failed: string;
    issues: string;
    recommendations: string;
    qualityScore: string;
  };
  
  // Processing
  processing: {
    validating: string;
    uploading: string;
    analyzing: string;
    complete: string;
    failed: string;
    progress: string;
    checkingResolution: string;
    transferring: string;
    extractingBiomarkers: string;
  };
  
  // Results
  results: {
    complete: string;
    processedIn: string;
    qualityBadge: string;
    riskScore: string;
    riskCategory: string;
    contributingFactors: string;
    biomarkerAnalysis: string;
    visualizations: string;
    segmentation: string;
    heatmap: string;
    downloadReport: string;
    saveToHistory: string;
    share: string;
    newAnalysis: string;
    assessmentId: string;
    patient: string;
    created: string;
  };
  
  // Risk Categories
  riskCategories: {
    minimal: string;
    low: string;
    moderate: string;
    elevated: string;
    high: string;
    critical: string;
  };
  
  // Biomarkers
  biomarkers: {
    vessels: {
      title: string;
      density: string;
      tortuosity: string;
      avrRatio: string;
    };
    opticDisc: {
      title: string;
      cupToDiscRatio: string;
      discArea: string;
      rimArea: string;
    };
    macula: {
      title: string;
      thickness: string;
      pigmentDensity: string;
    };
    amyloidBeta: {
      title: string;
      presenceScore: string;
      confidence: string;
      pattern: string;
    };
    referenceRange: string;
    normal: string;
    low: string;
    high: string;
    modelConfidence: string;
  };
  
  // Errors
  errors: {
    analysisFailed: string;
    networkError: string;
    validationFailed: string;
    timeout: string;
    serverError: string;
    tryAgain: string;
    backToUpload: string;
    contactSupport: string;
    recoverySteps: string;
    retryAttempt: string;
    maxRetriesReached: string;
  };
  
  // Accessibility
  a11y: {
    skipToContent: string;
    assessmentProgress: string;
    uploadStep: string;
    validatingStep: string;
    processingStep: string;
    resultsStep: string;
    stepComplete: string;
    stepCurrent: string;
    stepUpcoming: string;
    loadingModule: string;
    processingImage: string;
  };
  
  // Keyboard Shortcuts
  keyboard: {
    shortcuts: string;
    newAnalysis: string;
    help: string;
    cancel: string;
  };
}

// ============================================================================
// English Translations
// ============================================================================

export const en: RetinalTranslations = {
  pageTitle: 'Retinal Analysis',
  pageDescription: 'Advanced fundus image analysis with deep learning for neurological risk assessment',
  modelVersion: 'EfficientNet-B0',
  
  features: {
    processing: 'Processing: <500ms',
    aiPowered: 'AI-Powered Analysis',
    biomarkers: 'Neurological Biomarkers',
    hipaaCompliant: 'HIPAA Compliant',
  },
  
  upload: {
    title: 'Upload Retinal Image',
    description: 'Upload a high-quality fundus photograph for AI-powered neurological risk analysis',
    dragDrop: 'Drag and drop your image here',
    browse: 'Browse files',
    acceptedFormats: 'Accepted formats: JPG, PNG, TIFF',
    maxSize: 'Maximum file size: 10MB',
    selectImage: 'Select Image',
    uploading: 'Uploading...',
    processing: 'Processing...',
  },
  
  validation: {
    checking: 'Checking image quality...',
    success: 'Image validated successfully',
    failed: 'Validation failed',
    issues: 'Issues found',
    recommendations: 'Recommendations',
    qualityScore: 'Quality Score',
  },
  
  processing: {
    validating: 'Validating image quality...',
    uploading: 'Uploading image...',
    analyzing: 'Running AI analysis...',
    complete: 'Analysis complete!',
    failed: 'Analysis failed',
    progress: 'Progress',
    checkingResolution: 'Checking resolution, focus, and anatomical features',
    transferring: 'Securely transferring image to analysis server',
    extractingBiomarkers: 'Extracting biomarkers and calculating risk score',
  },
  
  results: {
    complete: 'Analysis Complete',
    processedIn: 'Processed in',
    qualityBadge: 'Quality',
    riskScore: 'Risk Score',
    riskCategory: 'Risk Category',
    contributingFactors: 'Contributing Factors',
    biomarkerAnalysis: 'Biomarker Analysis',
    visualizations: 'Visualizations',
    segmentation: 'Segmentation',
    heatmap: 'Attention Heatmap',
    downloadReport: 'Download PDF Report',
    saveToHistory: 'Save to History',
    share: 'Share',
    newAnalysis: 'New Analysis',
    assessmentId: 'Assessment ID',
    patient: 'Patient',
    created: 'Created',
  },
  
  riskCategories: {
    minimal: 'Minimal Risk',
    low: 'Low Risk',
    moderate: 'Moderate Risk',
    elevated: 'Elevated Risk',
    high: 'High Risk',
    critical: 'Critical Risk',
  },
  
  biomarkers: {
    vessels: {
      title: 'Vessel Biomarkers',
      density: 'Vessel Density',
      tortuosity: 'Tortuosity Index',
      avrRatio: 'A/V Ratio',
    },
    opticDisc: {
      title: 'Optic Disc Biomarkers',
      cupToDiscRatio: 'Cup-to-Disc Ratio',
      discArea: 'Disc Area',
      rimArea: 'Rim Area',
    },
    macula: {
      title: 'Macular Biomarkers',
      thickness: 'Macular Thickness',
      pigmentDensity: 'Pigment Density',
    },
    amyloidBeta: {
      title: 'Amyloid-β Indicators',
      presenceScore: 'Presence Score',
      confidence: 'Confidence',
      pattern: 'Distribution Pattern',
    },
    referenceRange: 'Reference Range',
    normal: 'Normal',
    low: 'Low',
    high: 'High',
    modelConfidence: 'Model Confidence',
  },
  
  errors: {
    analysisFailed: 'Analysis Failed',
    networkError: 'Network connection error',
    validationFailed: 'Image validation failed',
    timeout: 'Request timed out',
    serverError: 'Server error occurred',
    tryAgain: 'Try Again',
    backToUpload: 'Back to Upload',
    contactSupport: 'Contact Support',
    recoverySteps: 'Try these steps:',
    retryAttempt: 'Retry attempt',
    maxRetriesReached: 'Maximum retries reached. Please contact support.',
  },
  
  a11y: {
    skipToContent: 'Skip to main content',
    assessmentProgress: 'Assessment progress',
    uploadStep: 'Upload Image',
    validatingStep: 'Validating',
    processingStep: 'Processing',
    resultsStep: 'Results',
    stepComplete: 'complete',
    stepCurrent: 'current',
    stepUpcoming: 'upcoming',
    loadingModule: 'Loading retinal analysis interface, please wait...',
    processingImage: 'Analysis in progress',
  },
  
  keyboard: {
    shortcuts: 'Keyboard shortcuts',
    newAnalysis: 'New Analysis',
    help: 'Help',
    cancel: 'Cancel/Reset',
  },
};

// ============================================================================
// Spanish Translations
// ============================================================================

export const es: RetinalTranslations = {
  pageTitle: 'Análisis Retinal',
  pageDescription: 'Análisis avanzado de imágenes de fondo de ojo con aprendizaje profundo para evaluación de riesgo neurológico',
  modelVersion: 'EfficientNet-B0',
  
  features: {
    processing: 'Procesamiento: <500ms',
    aiPowered: 'Análisis con IA',
    biomarkers: 'Biomarcadores Neurológicos',
    hipaaCompliant: 'Compatible HIPAA',
  },
  
  upload: {
    title: 'Subir Imagen Retinal',
    description: 'Suba una fotografía de fondo de ojo de alta calidad para análisis de riesgo neurológico con IA',
    dragDrop: 'Arrastre y suelte su imagen aquí',
    browse: 'Explorar archivos',
    acceptedFormats: 'Formatos aceptados: JPG, PNG, TIFF',
    maxSize: 'Tamaño máximo: 10MB',
    selectImage: 'Seleccionar Imagen',
    uploading: 'Subiendo...',
    processing: 'Procesando...',
  },
  
  validation: {
    checking: 'Verificando calidad de imagen...',
    success: 'Imagen validada exitosamente',
    failed: 'Validación fallida',
    issues: 'Problemas encontrados',
    recommendations: 'Recomendaciones',
    qualityScore: 'Puntuación de Calidad',
  },
  
  processing: {
    validating: 'Validando calidad de imagen...',
    uploading: 'Subiendo imagen...',
    analyzing: 'Ejecutando análisis con IA...',
    complete: '¡Análisis completo!',
    failed: 'Análisis fallido',
    progress: 'Progreso',
    checkingResolution: 'Verificando resolución, enfoque y características anatómicas',
    transferring: 'Transfiriendo imagen de forma segura al servidor de análisis',
    extractingBiomarkers: 'Extrayendo biomarcadores y calculando puntuación de riesgo',
  },
  
  results: {
    complete: 'Análisis Completo',
    processedIn: 'Procesado en',
    qualityBadge: 'Calidad',
    riskScore: 'Puntuación de Riesgo',
    riskCategory: 'Categoría de Riesgo',
    contributingFactors: 'Factores Contribuyentes',
    biomarkerAnalysis: 'Análisis de Biomarcadores',
    visualizations: 'Visualizaciones',
    segmentation: 'Segmentación',
    heatmap: 'Mapa de Calor de Atención',
    downloadReport: 'Descargar Informe PDF',
    saveToHistory: 'Guardar en Historial',
    share: 'Compartir',
    newAnalysis: 'Nuevo Análisis',
    assessmentId: 'ID de Evaluación',
    patient: 'Paciente',
    created: 'Creado',
  },
  
  riskCategories: {
    minimal: 'Riesgo Mínimo',
    low: 'Riesgo Bajo',
    moderate: 'Riesgo Moderado',
    elevated: 'Riesgo Elevado',
    high: 'Riesgo Alto',
    critical: 'Riesgo Crítico',
  },
  
  biomarkers: {
    vessels: {
      title: 'Biomarcadores Vasculares',
      density: 'Densidad Vascular',
      tortuosity: 'Índice de Tortuosidad',
      avrRatio: 'Relación A/V',
    },
    opticDisc: {
      title: 'Biomarcadores del Disco Óptico',
      cupToDiscRatio: 'Relación Copa/Disco',
      discArea: 'Área del Disco',
      rimArea: 'Área del Anillo',
    },
    macula: {
      title: 'Biomarcadores Maculares',
      thickness: 'Espesor Macular',
      pigmentDensity: 'Densidad de Pigmento',
    },
    amyloidBeta: {
      title: 'Indicadores de Amiloide-β',
      presenceScore: 'Puntuación de Presencia',
      confidence: 'Confianza',
      pattern: 'Patrón de Distribución',
    },
    referenceRange: 'Rango de Referencia',
    normal: 'Normal',
    low: 'Bajo',
    high: 'Alto',
    modelConfidence: 'Confianza del Modelo',
  },
  
  errors: {
    analysisFailed: 'Análisis Fallido',
    networkError: 'Error de conexión de red',
    validationFailed: 'Validación de imagen fallida',
    timeout: 'Tiempo de espera agotado',
    serverError: 'Error del servidor',
    tryAgain: 'Intentar de Nuevo',
    backToUpload: 'Volver a Subir',
    contactSupport: 'Contactar Soporte',
    recoverySteps: 'Intente estos pasos:',
    retryAttempt: 'Intento de reintento',
    maxRetriesReached: 'Máximo de reintentos alcanzado. Por favor contacte soporte.',
  },
  
  a11y: {
    skipToContent: 'Saltar al contenido principal',
    assessmentProgress: 'Progreso de evaluación',
    uploadStep: 'Subir Imagen',
    validatingStep: 'Validando',
    processingStep: 'Procesando',
    resultsStep: 'Resultados',
    stepComplete: 'completado',
    stepCurrent: 'actual',
    stepUpcoming: 'próximo',
    loadingModule: 'Cargando interfaz de análisis retinal, por favor espere...',
    processingImage: 'Análisis en progreso',
  },
  
  keyboard: {
    shortcuts: 'Atajos de teclado',
    newAnalysis: 'Nuevo Análisis',
    help: 'Ayuda',
    cancel: 'Cancelar/Reiniciar',
  },
};

// ============================================================================
// Chinese (Simplified) Translations
// ============================================================================

export const zh: RetinalTranslations = {
  pageTitle: '视网膜分析',
  pageDescription: '基于深度学习的高级眼底图像分析，用于神经系统风险评估',
  modelVersion: 'EfficientNet-B0',
  
  features: {
    processing: '处理时间：<500毫秒',
    aiPowered: 'AI驱动分析',
    biomarkers: '神经系统生物标志物',
    hipaaCompliant: '符合HIPAA标准',
  },
  
  upload: {
    title: '上传视网膜图像',
    description: '上传高质量眼底照片，进行AI驱动的神经系统风险分析',
    dragDrop: '将图像拖放到此处',
    browse: '浏览文件',
    acceptedFormats: '接受的格式：JPG、PNG、TIFF',
    maxSize: '最大文件大小：10MB',
    selectImage: '选择图像',
    uploading: '上传中...',
    processing: '处理中...',
  },
  
  validation: {
    checking: '正在检查图像质量...',
    success: '图像验证成功',
    failed: '验证失败',
    issues: '发现问题',
    recommendations: '建议',
    qualityScore: '质量评分',
  },
  
  processing: {
    validating: '正在验证图像质量...',
    uploading: '正在上传图像...',
    analyzing: '正在运行AI分析...',
    complete: '分析完成！',
    failed: '分析失败',
    progress: '进度',
    checkingResolution: '检查分辨率、焦点和解剖特征',
    transferring: '正在安全地将图像传输到分析服务器',
    extractingBiomarkers: '提取生物标志物并计算风险评分',
  },
  
  results: {
    complete: '分析完成',
    processedIn: '处理时间',
    qualityBadge: '质量',
    riskScore: '风险评分',
    riskCategory: '风险类别',
    contributingFactors: '影响因素',
    biomarkerAnalysis: '生物标志物分析',
    visualizations: '可视化',
    segmentation: '分割图',
    heatmap: '注意力热图',
    downloadReport: '下载PDF报告',
    saveToHistory: '保存到历史',
    share: '分享',
    newAnalysis: '新分析',
    assessmentId: '评估ID',
    patient: '患者',
    created: '创建时间',
  },
  
  riskCategories: {
    minimal: '极低风险',
    low: '低风险',
    moderate: '中等风险',
    elevated: '较高风险',
    high: '高风险',
    critical: '危急风险',
  },
  
  biomarkers: {
    vessels: {
      title: '血管生物标志物',
      density: '血管密度',
      tortuosity: '迂曲度指数',
      avrRatio: '动静脉比',
    },
    opticDisc: {
      title: '视盘生物标志物',
      cupToDiscRatio: '杯盘比',
      discArea: '视盘面积',
      rimArea: '边缘面积',
    },
    macula: {
      title: '黄斑生物标志物',
      thickness: '黄斑厚度',
      pigmentDensity: '色素密度',
    },
    amyloidBeta: {
      title: 'β-淀粉样蛋白指标',
      presenceScore: '存在评分',
      confidence: '置信度',
      pattern: '分布模式',
    },
    referenceRange: '参考范围',
    normal: '正常',
    low: '偏低',
    high: '偏高',
    modelConfidence: '模型置信度',
  },
  
  errors: {
    analysisFailed: '分析失败',
    networkError: '网络连接错误',
    validationFailed: '图像验证失败',
    timeout: '请求超时',
    serverError: '服务器错误',
    tryAgain: '重试',
    backToUpload: '返回上传',
    contactSupport: '联系支持',
    recoverySteps: '请尝试以下步骤：',
    retryAttempt: '重试次数',
    maxRetriesReached: '已达到最大重试次数，请联系技术支持。',
  },
  
  a11y: {
    skipToContent: '跳转到主要内容',
    assessmentProgress: '评估进度',
    uploadStep: '上传图像',
    validatingStep: '验证中',
    processingStep: '处理中',
    resultsStep: '结果',
    stepComplete: '已完成',
    stepCurrent: '当前',
    stepUpcoming: '待处理',
    loadingModule: '正在加载视网膜分析界面，请稍候...',
    processingImage: '分析进行中',
  },
  
  keyboard: {
    shortcuts: '键盘快捷键',
    newAnalysis: '新分析',
    help: '帮助',
    cancel: '取消/重置',
  },
};

// ============================================================================
// Translation Utilities
// ============================================================================

const translations: Record<SupportedLocale, RetinalTranslations> = {
  en,
  es,
  zh,
};

/**
 * Get translations for a specific locale
 */
export function getTranslations(locale: SupportedLocale): RetinalTranslations {
  return translations[locale] || translations.en;
}

/**
 * Get a nested translation value by key path
 */
export function t(locale: SupportedLocale, keyPath: string): string {
  const trans = getTranslations(locale);
  const keys = keyPath.split('.');
  let value: any = trans;
  
  for (const key of keys) {
    if (value && typeof value === 'object' && key in value) {
      value = value[key];
    } else {
      return keyPath; // Return key path if not found
    }
  }
  
  return typeof value === 'string' ? value : keyPath;
}

/**
 * Get default locale from browser or localStorage
 */
export function getDefaultLocale(): SupportedLocale {
  if (typeof window === 'undefined') return 'en';
  
  // Check localStorage first
  const stored = localStorage.getItem('neuralens-locale');
  if (stored && ['en', 'es', 'zh'].includes(stored)) {
    return stored as SupportedLocale;
  }
  
  // Check browser language
  const browserLang = navigator.language.split('-')[0] ?? 'en';
  if (['en', 'es', 'zh'].includes(browserLang)) {
    return browserLang as SupportedLocale;
  }
  
  return 'en';
}

/**
 * Save locale preference
 */
export function setLocale(locale: SupportedLocale): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem('neuralens-locale', locale);
  }
}

export default translations;
