# Demo Assets

This folder contains demo/sample files for the MediLens frontend.

Files placed here are served directly by Next.js/Vercel at `/demo/filename`.

## Recommended Files

| File | Description | Max Size |
|------|-------------|----------|
| `sample_retinal.jpg` | Sample retinal fundus image | 500KB |
| `sample_xray.jpg` | Sample chest X-ray | 500KB |
| `sample_ecg.json` | Sample ECG waveform data | 50KB |
| `sample_speech.mp3` | Sample speech recording | 1MB |

## Usage in Frontend

```tsx
// Access demo files
const demoRetinal = '/demo/sample_retinal.jpg';
const demoXray = '/demo/sample_xray.jpg';

<Image src={demoRetinal} alt="Sample retinal scan" />
```

## Note

Keep files small and compressed. Large files (>5MB) should use external storage (Cloudflare R2).
