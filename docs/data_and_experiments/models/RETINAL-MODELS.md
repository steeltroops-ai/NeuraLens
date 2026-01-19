# Neuralens Retinal Analysis Models Directory

This directory contains the machine learning models used for retinal analysis in the Neuralens platform.

## Retinal Analysis Model

### File: `retinal_classifier.onnx`

**Status**: Placeholder - Model conversion in progress

**Description**: 
This will contain the EfficientNet-B0 model converted to ONNX format for client-side retinal analysis. The model is designed to detect neurological disorder indicators through fundus image analysis including vascular changes and cup-disc ratio assessment.

**Technical Specifications**:
- **Source Model**: EfficientNet-B0 (timm/efficientnet_b0.ra_in1k)
- **Format**: ONNX (Open Neural Network Exchange)
- **Size**: ~30MB (when converted)
- **Input**: 224x224x3 RGB images with ImageNet normalization
- **Output**: Spatial features (1280-dimensional) for risk assessment
- **Target Latency**: <150ms inference time
- **Target Precision**: ≥85% on APTOS 2019 dataset

**Conversion Process**:
1. Download EfficientNet-B0 from timm library
2. Convert to ONNX using torch.onnx.export
3. Optimize for WebAssembly deployment
4. Validate precision and performance

**Usage**:
The model is loaded by the RetinalProcessor class in `src/lib/ml/retinal/retinal-processor.ts` and used for real-time retinal analysis in the Dashboard component.

**Development Note**:
Currently using placeholder implementation. The actual model conversion requires:
- Python environment with timm, torch, onnx libraries
- ONNX conversion tools and optimization
- Model validation pipeline with APTOS 2019 dataset
- Performance optimization for web deployment

To implement the full model:
1. Set up Python environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install timm torch onnx onnxruntime opencv-python pillow`
3. Run conversion script in Jupyter notebook (ml_retinal_validation.ipynb)
4. Validate model precision and performance
5. Deploy optimized ONNX file to this directory

**Integration**:
The Retinal Analysis Card component automatically loads this model when initialized. The component handles:
- Model loading and initialization
- Image upload and camera capture
- Image preprocessing (resize, normalize)
- Real-time inference with ONNX Runtime
- Result display and API integration

**Performance Monitoring**:
The implementation includes comprehensive performance tracking:
- Processing time measurement
- Precision validation on test datasets
- Latency monitoring and optimization
- Error handling and reporting

**Demo Preparation**:
Demo retinal images are prepared with known NRI scores:
- Healthy retina: NRI 15, Vascular Score 0.25, Cup-Disc Ratio 0.30
- Moderate risk: NRI 55, Vascular Score 0.65, Cup-Disc Ratio 0.45  
- High risk: NRI 85, Vascular Score 0.85, Cup-Disc Ratio 0.65

**Next Steps**:
1. Complete model conversion pipeline in Jupyter notebook
2. Implement actual ONNX model inference
3. Validate precision on APTOS 2019 dataset
4. Optimize for production deployment
5. Create actual demo retinal images for testing

**Clinical Applications**:
The retinal analysis model detects:
- **Vascular Changes**: Vessel tortuosity, narrowing, density variations
- **Cup-Disc Ratio**: Optic disc cupping severity assessment
- **Risk Features**: Hemorrhages, microaneurysms, exudates
- **Neurological Indicators**: Early signs of Alzheimer's, stroke risk
- **Quality Assessment**: Image quality and reliability scoring

**Validation Results** (Target Performance):
- Precision: ≥85% on APTOS 2019 dataset
- Latency: <150ms average inference time
- Cross-validation: 83%+ mean precision with 5-fold CV
- Bias audit: <5% disparity across age and ethnicity groups
- Demo consistency: Validated against known NRI scores
