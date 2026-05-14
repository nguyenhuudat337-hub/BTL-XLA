"""
Core configuration settings for the vehicle counting application.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Configuration
    app_name: str = Field(default="Vehicle Counting System", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Model Configuration
    yolo_model_path: str = Field(default="yolov8n.pt", description="Path to YOLO model")
    confidence_threshold: float = Field(default=0.5, description="Detection confidence threshold")
    
    # Tracking Configuration
    max_age: int = Field(default=30, description="Maximum age for tracks")
    min_hits: int = Field(default=3, description="Minimum hits for track confirmation")
    iou_threshold: float = Field(default=0.3, description="IoU threshold for tracking")
    
    # Vehicle Classes (COCO dataset classes)
    vehicle_classes: List[int] = Field(
        default=[2, 3, 5, 7],  # car, motorbike, bus, truck
        description="COCO class IDs for vehicles"
    )
    
    # Video Processing
    max_video_size_mb: int = Field(default=100, description="Maximum video file size in MB")
    output_fps: int = Field(default=30, description="Output video FPS")
    
    # Paths
    upload_dir: str = Field(default="data/uploads", description="Upload directory")
    output_dir: str = Field(default="data/outputs", description="Output directory")
    log_dir: str = Field(default="logs", description="Log directory")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()