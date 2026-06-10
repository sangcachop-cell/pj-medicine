"""
Drug-Pred AI — FastAPI Backend
Hệ thống dự đoán nhóm thuốc từ mô tả bệnh án tiếng Việt
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: load ML model, connect DB, etc.
    print("🚀 Drug-Pred AI Backend starting...")
    yield
    # Shutdown: cleanup
    print("👋 Drug-Pred AI Backend shutting down...")


app = FastAPI(
    title="Drug-Pred AI",
    description="API hỗ trợ dự đoán nhóm thuốc từ mô tả bệnh án tiếng Việt",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — cho phép Frontend truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check ---
@app.get("/api/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "drug-pred-ai",
        "version": "0.1.0",
    }


# --- Import routers here khi có ---
# from app.api import auth, patients, records, predictions
# app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
# app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
# app.include_router(records.router, prefix="/api/records", tags=["Medical Records"])
# app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
