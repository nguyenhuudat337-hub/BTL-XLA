"""
Main FastAPI application for the vehicle counting system.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.api import routes


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{settings.log_dir}/vehicle_counting.log")
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Vehicle Counting System")
    logger.info(f"Configuration: {settings.app_name} v{settings.version}")
    
    # Ensure required directories exist
    import os
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs(settings.log_dir, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Vehicle Counting System")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Professional vehicle counting system using YOLOv8 and DeepSORT",
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler for unhandled exceptions.
    
    Args:
        request: HTTP request
        exc: Exception instance
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "path": str(request.url)
        }
    )


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(
    routes.router,
    prefix="/api/v1",
    tags=["Vehicle Counting API"]
)


# Root endpoint - redirect to web interface
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with basic information.
    
    Returns:
        Basic application information
    """
    return {
        "service": settings.app_name,
        "version": settings.version,
        "description": "Professional vehicle counting system using YOLOv8 and DeepSORT",
        "web_interface": "/static/index.html",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )