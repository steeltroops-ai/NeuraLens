# Cardiology Pipeline - Documentation

This directory contains detailed specifications for each component of the cardiology pipeline.

## Document Index

| Document | Description |
|----------|-------------|
| [00-OVERVIEW.md](../../../../docs/hackathon/pipeline/cardiology/00-OVERVIEW.md) | Pipeline architecture overview |
| [01-INPUT-VALIDATION.md](../../../../docs/hackathon/pipeline/cardiology/01-INPUT-VALIDATION.md) | Input validation specifications |
| [02-PREPROCESSING.md](../../../../docs/hackathon/pipeline/cardiology/02-PREPROCESSING.md) | Signal/image preprocessing |
| [03-ANATOMICAL-DETECTION.md](../../../../docs/hackathon/pipeline/cardiology/03-ANATOMICAL-DETECTION.md) | Echo structure detection |
| [04-FUNCTIONAL-ANALYSIS.md](../../../../docs/hackathon/pipeline/cardiology/04-FUNCTIONAL-ANALYSIS.md) | EF, wall motion, HRV analysis |
| [05-MODELS-INFERENCE.md](../../../../docs/hackathon/pipeline/cardiology/05-MODELS-INFERENCE.md) | Model architectures and inference |
| [06-POSTPROCESSING.md](../../../../docs/hackathon/pipeline/cardiology/06-POSTPROCESSING.md) | Aggregation and risk scoring |
| [07-ORCHESTRATION.md](../../../../docs/hackathon/pipeline/cardiology/07-ORCHESTRATION.md) | Pipeline state machine |
| [08-ERROR-HANDLING.md](../../../../docs/hackathon/pipeline/cardiology/08-ERROR-HANDLING.md) | Error taxonomy and handling |
| [09-RESPONSE-CONTRACT.md](../../../../docs/hackathon/pipeline/cardiology/09-RESPONSE-CONTRACT.md) | API response schemas |
| [10-SAFETY-COMPLIANCE.md](../../../../docs/hackathon/pipeline/cardiology/10-SAFETY-COMPLIANCE.md) | Safety and regulatory compliance |

## Quick Links

- **Full Documentation**: `docs/hackathon/pipeline/cardiology/`
- **Architecture**: See `ARCHITECTURE.md` in parent directory
- **File Structure**: See `FILE_STRUCTURE.md` in parent directory

## Navigation

```
backend/app/pipelines/cardiology/
|-- docs/                    <- You are here
|   |-- INDEX.md            <- This file
|
|-- (Implementation files)
|
|-- ARCHITECTURE.md          <- High-level architecture
|-- FILE_STRUCTURE.md        <- Complete folder guide
```

## Related Documentation

- Product Requirements: `docs/hackathon/09-CARDIOLOGY-PIPELINE-PRD.md`
- API Documentation: `docs/api/cardiology.md`
