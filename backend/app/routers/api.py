"""
MediLens API - Main Router
Combines all pipeline routers
"""

from fastapi import APIRouter

router = APIRouter()

# Import pipelines with fallback
pipelines = {}

try:
    from app.pipelines.speech.router import router as speech
    router.include_router(speech, prefix="/speech", tags=["Speech"])
    pipelines["speech"] = True
except Exception as e:
    print(f"[SKIP] Speech: {e}")

try:
    from app.pipelines.retinal.router import router as retinal
    router.include_router(retinal, prefix="/retinal", tags=["Retinal"])
    pipelines["retinal"] = True
except Exception as e:
    print(f"[SKIP] Retinal: {e}")

try:
    from app.pipelines.cardiology import router as cardio
    router.include_router(cardio, tags=["Cardiology"])
    pipelines["cardiology"] = True
except Exception as e:
    print(f"[SKIP] Cardiology: {e}")

try:
    from app.pipelines.radiology.router import router as radio
    router.include_router(radio, tags=["Radiology"])
    pipelines["radiology"] = True
except Exception as e:
    print(f"[SKIP] Radiology: {e}")

try:
    from app.pipelines.cognitive.router import router as cognitive
    router.include_router(cognitive, prefix="/cognitive", tags=["Cognitive"])
    pipelines["cognitive"] = True
except Exception as e:
    print(f"[SKIP] Cognitive: {e}")

try:
    from app.pipelines.motor.router import router as motor
    router.include_router(motor, prefix="/motor", tags=["Motor"])
    pipelines["motor"] = True
except Exception as e:
    print(f"[SKIP] Motor: {e}")

try:
    from app.pipelines.nri.router import router as nri
    router.include_router(nri, prefix="/nri", tags=["NRI Fusion"])
    pipelines["nri"] = True
except Exception as e:
    print(f"[SKIP] NRI: {e}")

try:
    from app.pipelines.voice.router import router as voice
    router.include_router(voice, tags=["Voice"])
    pipelines["voice"] = True
except Exception as e:
    print(f"[SKIP] Voice: {e}")

try:
    from app.pipelines.explain.router import router as explain
    router.include_router(explain, prefix="/explain", tags=["AI Explanation"])
    pipelines["explain"] = True
except Exception as e:
    print(f"[SKIP] Explain: {e}")

try:
    from app.pipelines.chatbot.router import router as chatbot
    router.include_router(chatbot, prefix="/chatbot", tags=["Medical Chatbot"])
    pipelines["chatbot"] = True
except Exception as e:
    print(f"[SKIP] Chatbot: {e}")


@router.get("/")
async def api_info():
    return {
        "name": "MediLens API",
        "pipelines": list(pipelines.keys()),
        "count": len(pipelines)
    }
