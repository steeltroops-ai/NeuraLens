# MediLens Backend - Hugging Face Spaces Deployment Guide

This guide covers deploying the MediLens backend to Hugging Face Spaces using GitHub Actions.

## Prerequisites

1. **Hugging Face Account** with a write-access token
2. **GitHub Repository** with the MediLens codebase
3. **GitHub Secrets** configured (see below)

## Setup Instructions

### Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Enter space name: `medilens-api` (or your preferred name)
3. Select **Docker** as the SDK
4. Choose **Blank** template
5. Set visibility (public or private)
6. Click **Create Space**

> Note: The first push from GitHub Actions will configure the Space automatically.

### Step 2: Generate Hugging Face Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name: `github-actions-deploy`
4. Type: **Write** (required for pushing to Spaces)
5. Copy the token (starts with `hf_`)

### Step 3: Configure GitHub Secrets

Go to your GitHub repository -> Settings -> Secrets and variables -> Actions

Add the following secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `HF_TOKEN` | Hugging Face write token | `hf_xxxxxxxxxxxxxxxxx` |
| `HF_USERNAME` | Your HF username | `your-username` |
| `HF_SPACE_NAME` | Name of your HF Space | `medilens-api` |
| `API_URL` | Your deployed backend URL | `https://your-username-medilens-api.hf.space` |
| `ELEVENLABS_API_KEY` | (Optional) Voice synthesis | `your-elevenlabs-key` |
| `CEREBRAS_API_KEY` | (Optional) AI explanations | `your-cerebras-key` |

### Step 4: Configure Hugging Face Space Secrets (Optional)

For API keys that the backend needs at runtime:

1. Go to your Space on Hugging Face
2. Click **Settings** tab
3. Scroll to **Repository secrets**
4. Add secrets for: `ELEVENLABS_API_KEY`, `CEREBRAS_API_KEY`, etc.

These will be available as environment variables in your Space.

## Deployment

### Automatic Deployment

The workflow triggers automatically on:
- Push to `main` branch (if backend files changed)
- Manual trigger via GitHub Actions UI

### Manual Deployment

1. Go to **Actions** tab in your GitHub repo
2. Select **Deploy MediLens** workflow
3. Click **Run workflow**
4. Choose deployment target:
   - `all` - Deploy both frontend and backend
   - `backend` - Deploy only backend
   - `frontend` - Deploy only frontend
5. Optionally check **Force deployment**
6. Click **Run workflow**

## Monitoring

### Check Deployment Status

1. **GitHub Actions**: View workflow runs and logs
2. **HF Console**: `https://huggingface.co/spaces/YOUR_USERNAME/medilens-api`
3. **Build Logs**: Click on your Space -> **Logs** tab

### Health Check

```bash
curl https://YOUR_USERNAME-medilens-api.hf.space/health
```

Expected response:
```json
{"status": "ok", "version": "1.0.0"}
```

## Troubleshooting

### Space Not Building

1. Check HF Space logs for errors
2. Verify Dockerfile syntax
3. Ensure all required files are present

### Authentication Failed

1. Verify `HF_TOKEN` has write permissions
2. Check `HF_USERNAME` matches your HF account exactly
3. Ensure Space exists (create it first if needed)

### Space Sleeping (Free Tier)

Free tier Spaces sleep after ~48 hours of inactivity. The workflow includes a keep-alive cron job that pings the Space every 25 minutes.

To manually wake a sleeping Space:
1. Visit the Space URL
2. Or trigger the keep-alive job manually

### Out of Memory

1. Reduce model sizes in `requirements.txt`
2. Use CPU-only PyTorch: Already configured in Dockerfile
3. Consider upgrading to paid HF tier for more resources

## File Structure

```
backend/
|-- Dockerfile           # Docker configuration for HF Spaces
|-- README.md            # HF Space README (with YAML frontmatter)
|-- .dockerignore        # Files to exclude from Docker build
|-- requirements.txt     # Python dependencies
|-- app/
    |-- main.py          # FastAPI application entry point
    |-- config.py        # Configuration settings
    |-- pipelines/       # AI diagnostic pipelines
    |-- routers/         # API route handlers
```

## Resource Limits (Free Tier)

| Resource | Limit |
|----------|-------|
| CPU | 2 vCPUs |
| RAM | 16 GB |
| Disk | 50 GB |
| Build Time | 30 minutes |
| Idle Timeout | ~48 hours |

## Upgrading to Paid Tier

For production use, consider upgrading:
1. Go to your Space -> Settings
2. Click **Hardware** section
3. Select a paid tier (e.g., CPU Upgrade, GPU)

Benefits:
- No sleep timeout
- More CPU/RAM
- GPU support (for faster inference)
- Priority build queue
