# Backend Assets (Local Only)

This folder is for local development and testing only.
It is NOT deployed to production (gitignored).

## Structure

```
assets/
|-- test_data/      # Test audio, images for development
|-- models/         # Downloaded ML model weights (cached)
```

## test_data/
Place test files here for local development:
- `.wav` audio files for speech analysis
- `.jpg/.png` images for retinal/radiology testing
- `.json` sample data

## models/
ML model weights are downloaded here at runtime:
- PyTorch models (.pt, .pth)
- ONNX models (.onnx)
- Cached from HuggingFace Hub

## Important
These folders are gitignored to avoid:
1. Large files in git history
2. HuggingFace deployment rejection
3. Slow clone times
