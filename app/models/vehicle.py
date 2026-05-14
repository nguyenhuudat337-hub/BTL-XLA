"""
Data models for vehicle detection and counting.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class VehicleType(str, Enum):
    """Enumeration of vehicle types."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    
    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple[float, float]:
        """Calculate bounding box center."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detection(BaseModel):
    """Vehicle detection model."""
    id: Optional[int] = Field(None, description="Detection ID")
    bbox: BoundingBox = Field(..., description="Bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class ID from model")
    vehicle_type: VehicleType = Field(default=VehicleType.UNKNOWN, description="Vehicle type")


class Track(BaseModel):
    """Vehicle tracking model."""
    track_id: int = Field(..., description="Unique track ID")
    bbox: BoundingBox = Field(..., description="Current bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Track confidence")
    class_id: int = Field(..., description="Class ID from model")
    vehicle_type: VehicleType = Field(default=VehicleType.UNKNOWN, description="Vehicle type")
    center_point: Tuple[float, float] = Field(..., description="Center point of track")
    history: List[Tuple[float, float]] = Field(default_factory=list, description="Position history")
    age: int = Field(default=0, description="Track age in frames")
    hits: int = Field(default=0, description="Number of detection hits")
    is_confirmed: bool = Field(default=True, description="Track confirmation status")


class CountingLine(BaseModel):
    """Counting line configuration."""
    name: str = Field(..., description="Line name/identifier")
    start_point: Tuple[float, float] = Field(..., description="Line start point (x, y)")
    end_point: Tuple[float, float] = Field(..., description="Line end point (x, y)")
    direction: str = Field(default="horizontal", description="Line orientation (horizontal/vertical)")


class VehicleCount(BaseModel):
    """Vehicle counting results."""
    total_vehicles: int = Field(default=0, description="Total vehicle count")
    vehicles_in: int = Field(default=0, description="Vehicles entering")
    vehicles_out: int = Field(default=0, description="Vehicles exiting")
    count_by_type: Dict[VehicleType, int] = Field(
        default_factory=dict, description="Count by vehicle type"
    )
    frame_number: int = Field(..., description="Current frame number")
    timestamp: float = Field(..., description="Timestamp in seconds")


class CountingStats(BaseModel):
    """Comprehensive counting statistics."""
    total_in: int = Field(default=0, description="Total vehicles entering")
    total_out: int = Field(default=0, description="Total vehicles exiting")
    total_vehicles: int = Field(default=0, description="Total vehicle count")
    net_count: int = Field(default=0, description="Net vehicle count (in - out)")
    counts_by_type: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Counts by vehicle type and direction"
    )
    session_duration: float = Field(default=0.0, description="Session duration in seconds")
    frames_processed: int = Field(default=0, description="Number of frames processed")
    active_tracks: int = Field(default=0, description="Number of active tracks")
    counting_lines: int = Field(default=0, description="Number of counting lines")


class VehicleCount(BaseModel):
    """Vehicle counting results."""
    total_vehicles: int = Field(default=0, description="Total vehicle count")
    vehicles_in: int = Field(default=0, description="Vehicles entering")
    vehicles_out: int = Field(default=0, description="Vehicles exiting")
    count_by_type: Dict[VehicleType, int] = Field(
        default_factory=dict, description="Count by vehicle type"
    )
    frame_number: int = Field(..., description="Current frame number")
    timestamp: float = Field(..., description="Timestamp in seconds")


class ProcessingResult(BaseModel):
    """Video processing result."""
    video_id: str = Field(..., description="Unique video identifier")
    total_frames: int = Field(..., description="Total number of frames processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    final_count: VehicleCount = Field(..., description="Final counting results")
    output_video_path: Optional[str] = Field(None, description="Path to output video")
    success: bool = Field(default=True, description="Processing success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class VideoProcessRequest(BaseModel):
    """Request model for video processing."""
    video_path: str = Field(..., description="Path to input video file")
    counting_lines: Optional[List[CountingLine]] = Field(
        None, description="Custom counting lines"
    )
    save_output_video: bool = Field(default=True, description="Save processed video")
    detection_classes: Optional[List[int]] = Field(
        None, description="Custom detection classes"
    )