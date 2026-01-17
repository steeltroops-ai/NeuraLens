# MediLens Pre-Built Models Guide

## Quick Decision Matrix

| Pipeline | Recommended Model | Install Command | Why |
|----------|------------------|-----------------|-----|
| **Speech** | Parselmouth + Whisper | `pip install parselmouth openai-whisper` | Industry-standard voice analysis |
| **Retinal** | timm EfficientNet-B4 | `pip install timm` | Best accuracy/speed ratio |
| **Cardiology** | HeartPy + NeuroKit2 | `pip install heartpy neurokit2` | Automatic ECG processing |
| **Radiology** | TorchXRayVision | `pip install torchxrayvision` | 18 conditions, 8 datasets |
| **Cognitive** | NumPy + SciPy | `pip install numpy scipy` | Rule-based scoring |
| **Motor** | SciPy signal | `pip install scipy` | FFT tremor detection |
| **Voice** | ElevenLabs | `pip install elevenlabs` | Best TTS quality |

---

## Complete Installation

```bash
# Core ML
pip install torch torchvision torchaudio

# Speech Pipeline
pip install parselmouth openai-whisper librosa soundfile surfboard

# Retinal/Radiology Pipeline  
pip install timm torchxrayvision pytorch-grad-cam opencv-python pillow

# Cardiology Pipeline
pip install heartpy neurokit2 biosppy

# Signal Processing (Motor)
pip install scipy numpy

# Voice Output
pip install elevenlabs gtts
```

---

## 1. Speech Analysis

### Libraries

| Library | Purpose | Key Features |
|---------|---------|--------------|
| **Parselmouth** | Voice features | Jitter, shimmer, HNR (Praat wrapper) |
| **Whisper** | Transcription | Speech-to-text for pause analysis |
| **Surfboard** | Auto features | 40+ features automatically |
| **librosa** | Audio processing | MFCC, spectral features |

### Usage
```python
import parselmouth
from parselmouth.praat import call

sound = parselmouth.Sound("audio.wav")
jitter = call(sound, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
shimmer = call(sound, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
hnr = call(sound, "Get harmonicity (cc)", 0, 0, 0.01, 4500)
```

---

## 2. Retinal Imaging

### Libraries

| Library | Purpose | Key Features |
|---------|---------|--------------|
| **timm** | Classification | EfficientNet, ResNet pretrained |
| **pytorch-grad-cam** | Explainability | Heatmap generation |
| **OpenCV** | Preprocessing | Image manipulation |

### Pre-trained Models
```python
import timm

# Option 1: EfficientNet (recommended)
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=5)

# Option 2: ResNet
model = timm.create_model('resnet50', pretrained=True, num_classes=5)

# Option 3: DINOv2 for features
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-base")
```

### Heatmap
```python
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
heatmap = cam(input_tensor=image_tensor)
```

---

## 3. Cardiology ECG

### Libraries

| Library | Purpose | Key Features |
|---------|---------|--------------|
| **HeartPy** | ECG analysis | Automatic HR, HRV, R-peaks |
| **NeuroKit2** | Comprehensive | Full ECG processing |
| **biosppy** | Simple | Basic ECG features |

### Usage
```python
import heartpy as hp

# Load and process
data, measures = hp.process(ecg_signal, sample_rate=500)

print(f"Heart Rate: {measures['bpm']}")
print(f"RMSSD: {measures['rmssd']}")
print(f"SDNN: {measures['sdnn']}")
```

---

## 4. Radiology X-Ray

### TorchXRayVision

| Feature | Value |
|---------|-------|
| **Datasets** | 8 merged datasets |
| **Conditions** | 18 pathologies |
| **Architecture** | DenseNet121 |

### Usage
```python
import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# Preprocess
img = xrv.datasets.normalize(img, 255)

# Predict 18 conditions
outputs = model(img.unsqueeze(0))
probs = torch.sigmoid(outputs)
```

---

## 5. Voice Output

### ElevenLabs

| Feature | Value |
|---------|-------|
| **Quality** | Human-like |
| **Latency** | < 1 second |
| **Languages** | Multilingual |

### Usage
```python
from elevenlabs import generate, set_api_key

set_api_key("your_key")

audio = generate(
    text="Your results are ready.",
    voice="Rachel",
    model="eleven_multilingual_v2"
)

# Save or return as base64
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### gTTS Fallback
```python
from gtts import gTTS
import io

tts = gTTS(text="Your results are ready.", lang='en')
audio_buffer = io.BytesIO()
tts.write_to_fp(audio_buffer)
```

---

## Model Download Locations

| Model | Size | Download |
|-------|------|----------|
| Whisper base | 139 MB | Auto on first use |
| EfficientNet-B4 | 75 MB | timm downloads |
| TorchXRayVision | 28 MB | xrv downloads |
| DINOv2 | 350 MB | HuggingFace |

---

## GPU Considerations

### With GPU (CUDA)
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### CPU Only
- Whisper: Use "tiny" or "base" model
- timm: Use EfficientNet-B0 or B2
- All other libs: CPU-optimized

---

## Full requirements.txt

```txt
# FastAPI Backend
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6

# Database
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
asyncpg>=0.29.0

# Core ML
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.11.0
pillow>=10.0.0

# Speech Pipeline
parselmouth>=0.4.3
openai-whisper>=20231117
librosa>=0.10.2
soundfile>=0.12.1
surfboard>=0.2.0

# Retinal/Radiology Pipeline
timm>=0.9.0
torchxrayvision>=1.0.0
pytorch-grad-cam>=1.4.0
opencv-python>=4.8.0

# Cardiology Pipeline
heartpy>=1.2.7
neurokit2>=0.2.0
biosppy>=0.8.0

# Voice
elevenlabs>=0.2.0
gtts>=2.4.0
```
