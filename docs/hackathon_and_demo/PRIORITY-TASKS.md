# MediLens Development - Priority Action Plan

## Current Status (2026-01-17)

### Backend Status: Working
- FastAPI server: Running on port 8000
- Routes loaded: 34
- Pipelines active: Speech, Retinal, Cardiology, Radiology, Cognitive, Motor, NRI, Voice, Explain

### Missing Dependencies (Optional - Install if needed)
```bash
# Only install these when you need the full feature
pip install torchxrayvision  # Radiology X-ray analysis
pip install elevenlabs       # Voice output
pip install cerebras-cloud-sdk  # AI explanations
pip install torch torchvision   # Deep learning models
```

---

## Priority Task List

### P0 - Critical (Complete First)

1. **Frontend API Integration**
   - Connect frontend to backend endpoints
   - File: `frontend/src/lib/api/` 
   - Status: Needs verification

2. **Test All Pipeline Endpoints**
   - Speech: POST /api/speech/analyze
   - Retinal: POST /api/retinal/analyze  
   - Cardiology: POST /api/cardiology/demo
   - NRI: POST /api/nri/calculate

3. **Environment Variables**
   - Create `.env` in backend with:
   ```
   CEREBRAS_API_KEY=csk-d2ry3r6e4rf5nf9h93kj8wed2f642enwjddh644k2xm8hmwt
   ELEVENLABS_API_KEY=your_key
   DATABASE_URL=sqlite:///./medilens.db
   ```

### P1 - High Priority

4. **AI Explanation Panel (Frontend)**
   - Create `ExplanationPanel.tsx` component
   - Integrate with all pipeline result pages
   - Add streaming text display
   - Add voice playback button

5. **Fix Pipeline Analyzers**
   - Some pipeline analyzers may need real implementations
   - Currently using simulation/demo mode

6. **Dashboard Integration**
   - Ensure all pipeline pages connect to API
   - Add loading states
   - Add error handling

### P2 - Medium Priority

7. **Install Optional Dependencies**
   - TorchXRayVision for real X-ray analysis
   - ElevenLabs for voice output
   - OpenAI Whisper for speech-to-text

8. **Styling Refinements**
   - Dark mode consistency
   - Animation polish
   - Mobile responsiveness

### P3 - Nice to Have

9. **Historical Data Tracking**
   - Store assessment results in DB
   - Show trends over time

10. **Export/Share Results**
    - PDF generation
    - Share link creation

---

## Quick Commands

### Start Backend
```bash
cd backend
.venv\Scripts\activate  # Windows
uvicorn app.main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend
bun run dev
```

### Test API
```bash
# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/api/

# Demo cardiology
curl -X POST http://localhost:8000/api/cardiology/demo \
  -H "Content-Type: application/json" \
  -d '{"heart_rate": 72, "duration": 10}'
```

---

## Next Immediate Steps

1. Verify frontend is connecting to backend
2. Test each pipeline endpoint
3. Create ExplanationPanel component
4. Integrate AI explanations into results pages
