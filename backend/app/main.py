"""
MediLens Backend - FastAPI Application
AI-Powered Medical Diagnostics Platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db
from app.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[START] {settings.APP_NAME} v{settings.VERSION}")
    await init_db()
    yield
    print("[STOP] Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-Powered Multi-Modal Medical Diagnostics",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}


# Include all pipeline routes
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
