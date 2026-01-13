# Neuralens ML Models Directory

This directory contains the machine learning models used for neurological assessment in the Neuralens platform.

## Speech Analysis Model

### File: `speech_classifier.onnx`

**Status**: Placeholder - Model conversion in progress

**Description**: 
This will contain the Whisper-tiny model converted to ONNX format for client-side speech analysis. The model is designed to detect neurological disorder indicators through voice pattern analysis.

**Technical Specifications**:
- **Source Model**: OpenAI Whisper-tiny
- **Format**: ONNX (Open Neural Network Exchange)
- **Size**: ~200MB (when converted)
- **Input**: MFCC features (13 coefficients)
- **Output**: Fluency predictions and biomarkers
- **Target Latency**: <100ms inference time
- **Target Accuracy**: ≥90% on DementiaBank dataset

**Conversion Process**:
1. Download whisper-tiny from Hugging Face
2. Convert to ONNX using transformers.onnx
3. Optimize for WebAssembly deployment
4. Validate accuracy and performance

**Usage**:
The model is loaded by the SpeechProcessor class in `src/lib/ml/speech-processor.ts` and used for real-time speech analysis in the Dashboard component.

**Development Note**:
Currently using placeholder implementation. The actual model conversion requires:
- Python environment with transformers library
- ONNX conversion tools
- Model validation pipeline
- Performance optimization for web deployment

To implement the full model:
1. Set up Python environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install transformers onnx onnxruntime torch`
3. Run conversion script (to be created in Jupyter notebook)
4. Validate model accuracy and performance
5. Deploy optimized ONNX file to this directory

**Integration**:
The Speech Analysis Card component automatically loads this model when initialized. The component handles:
- Model loading and initialization
- Audio recording via WebRTC
- Feature extraction (MFCC)
- Real-time inference
- Result display and API integration

**Performance Monitoring**:
The implementation includes comprehensive performance tracking:
- Processing time measurement
- Accuracy validation
- Latency monitoring
- Error handling and reporting

**Next Steps**:
1. Complete model conversion pipeline
2. Implement MFCC feature extraction
3. Validate accuracy on test dataset
4. Optimize for production deployment
5. Create demo audio samples for testing
