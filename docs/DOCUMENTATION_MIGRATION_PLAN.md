# MediLens Documentation Migration Plan

> Generated: 2026-01-19
> **Status: COMPLETE**
> Total Files Migrated: 78 markdown files now in docs/

---

## 1. Discovery Summary

### Total Documentation Files Identified: 68

**Source Locations:**
- Root: 1 file
- `backend/`: 7 files
- `backend/app/pipelines/`: 26 files (across 10 pipeline subdirectories)
- `data/`: 1 file
- `docs/`: 31 files (already in docs/)
- `frontend/public/`: 3 files

---

## 2. File-by-File Migration Plan

### Legend
- `[MOVE]` - File to be relocated
- `[KEEP]` - File already in correct location, may need restructuring
- `[MERGE]` - Content to be merged into another file
- `[SPLIT]` - Large file to be split into smaller focused files

---

### A. Overview Documents

| Status | Original Path | New Path |
|--------|--------------|----------|
| [KEEP] | `README.md` | `README.md` (root remains, serves as project entry point) |
| [MOVE] | `docs/HACKATHON_SUBMISSION.md` | `docs/hackathon_and_demo/SUBMISSION.md` |
| [KEEP] | `docs/hackathon/00-PIPELINE-OVERVIEW.md` | `docs/overview/PIPELINE-OVERVIEW.md` |
| [MOVE] | `docs/hackathon/00-COMPETITIVE-ANALYSIS.md` | `docs/research_and_design/COMPETITIVE-ANALYSIS.md` |

---

### B. Backend Documentation

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/README.md` | `docs/backend/README.md` |
| [MOVE] | `backend/DEPLOYMENT.md` | `docs/infrastructure/DEPLOYMENT.md` |
| [MOVE] | `backend/AI_EXPLANATION_FIX_PLAN.md` | `docs/backend/AI-EXPLANATION-FIX-PLAN.md` |
| [MOVE] | `backend/SPEECH_PIPELINE.md` | `docs/pipelines/speech/ARCHITECTURE.md` |
| [MOVE] | `backend/SPEECH_FRONTEND_INTEGRATION.md` | `docs/pipelines/speech/FRONTEND-INTEGRATION.md` |
| [MOVE] | `backend/RETINAL_PIPELINE.md` | `docs/pipelines/retinal/ARCHITECTURE.md` |
| [MOVE] | `backend/assets/README.md` | `docs/backend/ASSETS.md` |

---

### C. Pipeline Documentation - Retinal

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/retinal/ARCHITECTURE.md` | `docs/pipelines/retinal/00-OVERVIEW.md` |
| [MOVE] | `backend/app/pipelines/retinal/FILE_STRUCTURE.md` | `docs/pipelines/retinal/FILE-STRUCTURE.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/00-INDEX.md` | `docs/pipelines/retinal/INDEX.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/01-INPUT-VALIDATION-SPEC.md` | `docs/pipelines/retinal/01-INPUT-VALIDATION.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/02-PREPROCESSING-SPEC.md` | `docs/pipelines/retinal/02-PREPROCESSING.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/03-ANATOMICAL-DETECTION-SPEC.md` | `docs/pipelines/retinal/03-ANATOMICAL-DETECTION.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/04-PATHOLOGY-ANALYSIS-SPEC.md` | `docs/pipelines/retinal/04-PATHOLOGY-ANALYSIS.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/05-MODELS-INFERENCE-SPEC.md` | `docs/pipelines/retinal/05-MODELS-INFERENCE.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/06-POST-PROCESSING-SPEC.md` | `docs/pipelines/retinal/06-POSTPROCESSING.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/07-ORCHESTRATION-SPEC.md` | `docs/pipelines/retinal/07-ORCHESTRATION.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/08-ERROR-HANDLING-SPEC.md` | `docs/pipelines/retinal/08-ERROR-HANDLING.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/09-RESPONSE-CONTRACT-SPEC.md` | `docs/pipelines/retinal/09-RESPONSE-CONTRACT.md` |
| [MOVE] | `backend/app/pipelines/retinal/docs/10-SAFETY-COMPLIANCE-SPEC.md` | `docs/pipelines/retinal/10-SAFETY-COMPLIANCE.md` |
| [MERGE] | `docs/hackathon/02-RETINAL-PIPELINE-PRD.md` | `docs/pipelines/retinal/PRD.md` |

---

### D. Pipeline Documentation - Cardiology

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/cardiology/ARCHITECTURE.md` | `docs/pipelines/cardiology/ARCHITECTURE.md` |
| [MOVE] | `backend/app/pipelines/cardiology/FILE_STRUCTURE.md` | `docs/pipelines/cardiology/FILE-STRUCTURE.md` |
| [MOVE] | `backend/app/pipelines/cardiology/docs/INDEX.md` | `docs/pipelines/cardiology/INDEX.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/00-OVERVIEW.md` | `docs/pipelines/cardiology/00-OVERVIEW.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/01-INPUT-VALIDATION.md` | `docs/pipelines/cardiology/01-INPUT-VALIDATION.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/02-PREPROCESSING.md` | `docs/pipelines/cardiology/02-PREPROCESSING.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/03-ANATOMICAL-DETECTION.md` | `docs/pipelines/cardiology/03-ANATOMICAL-DETECTION.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/04-FUNCTIONAL-ANALYSIS.md` | `docs/pipelines/cardiology/04-FUNCTIONAL-ANALYSIS.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/05-MODELS-INFERENCE.md` | `docs/pipelines/cardiology/05-MODELS-INFERENCE.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/06-POSTPROCESSING.md` | `docs/pipelines/cardiology/06-POSTPROCESSING.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/07-ORCHESTRATION.md` | `docs/pipelines/cardiology/07-ORCHESTRATION.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/08-ERROR-HANDLING.md` | `docs/pipelines/cardiology/08-ERROR-HANDLING.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/09-RESPONSE-CONTRACT.md` | `docs/pipelines/cardiology/09-RESPONSE-CONTRACT.md` |
| [KEEP] | `docs/hackathon/pipeline/cardiology/10-SAFETY-COMPLIANCE.md` | `docs/pipelines/cardiology/10-SAFETY-COMPLIANCE.md` |
| [MOVE] | `docs/hackathon/09-CARDIOLOGY-PIPELINE-PRD.md` | `docs/pipelines/cardiology/PRD.md` |

---

### E. Pipeline Documentation - Radiology

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/radiology/ARCHITECTURE.md` | `docs/pipelines/radiology/ARCHITECTURE.md` |
| [KEEP] | `docs/hackathon/pipeline/radiology/00-OVERVIEW.md` | `docs/pipelines/radiology/00-OVERVIEW.md` |
| [KEEP] | `docs/hackathon/pipeline/radiology/01-INPUT-VALIDATION.md` | `docs/pipelines/radiology/01-INPUT-VALIDATION.md` |
| [MOVE] | `docs/hackathon/10-RADIOLOGY-PIPELINE-PRD.md` | `docs/pipelines/radiology/PRD.md` |

---

### F. Pipeline Documentation - Speech

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/speech/ARCHITECTURE.md` | `docs/pipelines/speech/00-OVERVIEW.md` |
| [MOVE] | `docs/hackathon/01-SPEECH-PIPELINE-PRD.md` | `docs/pipelines/speech/PRD.md` |

---

### G. Pipeline Documentation - Cognitive

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/cognitive/ARCHITECTURE.md` | `docs/pipelines/cognitive/ARCHITECTURE.md` |
| [MOVE] | `docs/hackathon/04-COGNITIVE-PIPELINE-PRD.md` | `docs/pipelines/cognitive/PRD.md` |

---

### H. Pipeline Documentation - Motor

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/motor/ARCHITECTURE.md` | `docs/pipelines/motor/ARCHITECTURE.md` |
| [MOVE] | `docs/hackathon/05-MOTOR-PIPELINE-PRD.md` | `docs/pipelines/motor/PRD.md` |

---

### I. Pipeline Documentation - NRI Fusion

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/nri/ARCHITECTURE.md` | `docs/pipelines/nri/ARCHITECTURE.md` |
| [MOVE] | `docs/hackathon/03-NRI-FUSION-PRD.md` | `docs/pipelines/nri/PRD.md` |

---

### J. Pipeline Documentation - Voice

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/voice/ARCHITECTURE.md` | `docs/pipelines/voice/ARCHITECTURE.md` |
| [MOVE] | `docs/hackathon/11-VOICE-ASSISTANT-PRD.md` | `docs/pipelines/voice/PRD.md` |

---

### K. Pipeline Documentation - AI Explanation

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/explain/ARCHITECTURE.md` | `docs/pipelines/explain/ARCHITECTURE.md` |
| [MOVE] | `docs/hackathon/13-AI-EXPLANATION-PIPELINE-PRD.md` | `docs/pipelines/explain/PRD.md` |

---

### L. Pipeline Documentation - Chatbot

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `backend/app/pipelines/chatbot/ARCHITECTURE.md` | `docs/pipelines/chatbot/ARCHITECTURE.md` |

---

### M. Frontend/Models Documentation

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `frontend/public/models/retinal/README.md` | `docs/data_and_experiments/models/RETINAL-MODELS.md` |
| [MOVE] | `frontend/public/models/speech/README.md` | `docs/data_and_experiments/models/SPEECH-MODELS.md` |
| [MOVE] | `frontend/public/demo/README.md` | `docs/hackathon_and_demo/DEMO-ASSETS.md` |

---

### N. Data Documentation

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `data/DATA_STORAGE.md` | `docs/data_and_experiments/DATA-STORAGE.md` |

---

### O. Hackathon/Demo Materials

| Status | Original Path | New Path |
|--------|--------------|----------|
| [MOVE] | `docs/hackathon/PRIORITY-TASKS.md` | `docs/hackathon_and_demo/PRIORITY-TASKS.md` |
| [MOVE] | `docs/hackathon/06-DEPLOYMENT.md` | `docs/infrastructure/HACKATHON-DEPLOYMENT.md` |
| [MOVE] | `docs/hackathon/07-SUBMISSION-CHECKLIST.md` | `docs/hackathon_and_demo/SUBMISSION-CHECKLIST.md` |
| [MOVE] | `docs/hackathon/08-AGENT-WORKFLOWS.md` | `docs/hackathon_and_demo/AGENT-WORKFLOWS.md` |
| [MOVE] | `docs/hackathon/12-PREBUILT-MODELS-GUIDE.md` | `docs/data_and_experiments/models/PREBUILT-MODELS-GUIDE.md` |

---

## 3. Documents to Split

No documents identified for splitting. All documents are appropriately focused.

---

## 4. Link Updates Required

After migration, the following internal links must be updated:

| File | Link Pattern | Update To |
|------|-------------|-----------|
| `README.md` | `docs/` | Verify paths remain valid |
| All pipeline docs | `../backend/app/pipelines/...` | `../pipelines/...` |
| All hackathon docs | `backend/DEPLOYMENT.md` | `infrastructure/DEPLOYMENT.md` |
| Backend README | `SPEECH_PIPELINE.md` | `../docs/pipelines/speech/ARCHITECTURE.md` |
| Backend README | `RETINAL_PIPELINE.md` | `../docs/pipelines/retinal/ARCHITECTURE.md` |

---

## 5. Confirmation Checklist

Upon completion:
- [ ] All 68 markdown files accounted for
- [ ] No documentation remains in source folders (except root README)
- [ ] All internal links updated
- [ ] Navigation hub (docs/README.md) created
- [ ] Git history preserved where possible

---
