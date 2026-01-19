# MediLens Data Storage Architecture

## Storage Solutions by Data Type

### 1. User/Application Data (Database)
**Service: Neon PostgreSQL** (Already configured via MCP)
- User profiles, assessment results, medical history
- Free tier: 0.5 GB storage, 1 compute
- Connection via `DATABASE_URL` environment variable

### 2. Binary Files (Audio, Images, Models)
**Service: Cloudflare R2** (Recommended - Free)
- Audio recordings (.wav, .mp3)  
- Medical images (retinal scans, X-rays)
- ML model weights (.pt, .pth, .onnx)

**Free Tier:**
- 10 GB storage
- 10 million Class A operations/month
- 1 million Class B operations/month
- No egress fees

**Setup:**
1. Create Cloudflare account: https://dash.cloudflare.com
2. Go to R2 -> Create bucket -> Name: `medilens-assets`
3. Create API token with R2 read/write access
4. Add to environment variables:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET_NAME`

### 3. Demo/Sample Files for Frontend
**Location: `frontend/public/demo/`**
- Served directly by Vercel/Next.js
- Accessible at: `https://your-site.vercel.app/demo/filename`
- Keep files small (< 5MB each)

### 4. ML Model Weights
**Service: Hugging Face Hub** (Free)
- Store large model files (.pt, .pth, .safetensors)
- Download at runtime using `huggingface_hub`
- No storage limits for public models

---

## Folder Structure

```
NeuraLens/
|-- frontend/
|   |-- public/
|       |-- demo/                    # Demo files for frontend
|           |-- sample_retinal.jpg   # Sample retinal image
|           |-- sample_xray.jpg      # Sample chest X-ray
|           |-- sample_ecg.json      # Sample ECG data
|           |-- sample_speech.mp3    # Sample speech (compressed)
|
|-- backend/
|   |-- app/
|   |-- assets/                      # NOT deployed - local testing only
|       |-- test_data/               # Test files for development
|       |-- models/                  # Downloaded models (gitignored)
|
|-- .gitignore                       # Ignore binary files
```

---

## Environment Variables

Add to your `.env` files:

### Backend (.env)
```env
# Database (Neon)
DATABASE_URL=postgresql://user:pass@host/db

# Cloudflare R2 (Binary Storage)
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=medilens-assets

# HuggingFace (Model Downloads)
HF_TOKEN=hf_your_token
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_DEMO_BASE_URL=/demo
NEXT_PUBLIC_API_URL=https://your-backend.hf.space
```

---

## Code Examples

### Upload to R2 (Backend)
```python
import boto3
from botocore.config import Config

def get_r2_client():
    return boto3.client(
        's3',
        endpoint_url=f'https://{os.getenv("R2_ACCOUNT_ID")}.r2.cloudflarestorage.com',
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version='s3v4')
    )

def upload_file(file_path: str, object_key: str):
    client = get_r2_client()
    client.upload_file(file_path, os.getenv("R2_BUCKET_NAME"), object_key)
    return f"https://pub-xxx.r2.dev/{object_key}"
```

### Download Models from HF (Backend)
```python
from huggingface_hub import hf_hub_download

def download_model(repo_id: str, filename: str):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir="./assets/models"
    )
```

---

## Summary

| Data Type | Service | Free Tier | Use Case |
|-----------|---------|-----------|----------|
| User data | Neon PostgreSQL | 0.5GB | Assessments, profiles |
| Binary files | Cloudflare R2 | 10GB | Audio, images |
| Demo files | Vercel (public/) | Unlimited | Sample data |
| ML models | HuggingFace Hub | Unlimited | Model weights |
