"""
Main video processing service integrating detection, tracking, and counting.
"""

import logging
import cv2
import numpy as np
from typing import List, Optional, Generator, Tuple
import time
from pathlib import Path

from app.services.detection_service import DetectionService
from app.services.tracking_service import TrackingService
from app.services.counting_service import CountingService
from app.models.vehicle import (
    CountingLine, CountingStats, Track, Detection, VehicleType
)
from app.core.config import settings


logger = logging.getLogger(__name__)


class VideoProcessingService:
    """Main service for processing video streams with vehicle counting."""
    
    def __init__(self):
        """Initialize the video processing service."""
        self.detection_service = DetectionService()
        self.tracking_service = TrackingService()
        self.counting_service = CountingService()
        
        # Processing state
        self.is_processing = False
        self.current_frame = None
        self.frame_count = 0
        self.fps = 0.0
        
        # Default counting line (center of frame)
        self.default_counting_lines = []
        
        logger.info("Video processing service initialized")
    
    def set_counting_lines(self, lines: List[CountingLine]) -> None:
        """
        Set counting lines for the service.
        
        Args:
            lines: List of counting lines
        """
        self.counting_service.set_counting_lines(lines)
        logger.info(f"Set {len(lines)} counting lines")
    
    def create_default_counting_line(self, frame_width: int, frame_height: int) -> CountingLine:
        """
        Create a default counting line in the center of the frame.
        
        Args:
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Default counting line
        """
        return CountingLine(
            name="center_line",
            start_point=(0, frame_height // 2),
            end_point=(frame_width, frame_height // 2),
            direction="horizontal"
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, CountingStats]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, counting_stats)
        """
        start_time = time.time()
        
        # Detect vehicles
        detections = self.detection_service.detect_vehicles(frame)
        
        # Update tracks
        tracks = self.tracking_service.update_tracks(detections, frame)
        
        # Process counting
        frame_counts = self.counting_service.process_tracks(tracks)
        
        # Draw results on frame
        result_frame = self._draw_results(frame, tracks, detections)
        
        # Get current statistics
        stats = self.counting_service.get_current_stats()
        
        # Update frame count and FPS
        self.frame_count += 1
        processing_time = time.time() - start_time
        self.fps = 1.0 / processing_time if processing_time > 0 else 0.0
        
        # Cleanup old tracks periodically
        if self.frame_count % 30 == 0:
            active_track_ids = {track.track_id for track in tracks}
            self.counting_service.cleanup_old_tracks(active_track_ids)
            self.tracking_service.clear_old_tracks()
        
        return result_frame, stats
    
    def process_video_file(self, video_path: str, 
                          output_path: Optional[str] = None) -> Generator[Tuple[np.ndarray, CountingStats], None, None]:
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            
        Yields:
            Tuple of (frame, stats) for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Create default counting line if none set
        if not self.counting_service.counting_lines:
            default_line = self.create_default_counting_line(frame_width, frame_height)
            self.set_counting_lines([default_line])
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        self.is_processing = True
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, stats = self.process_frame(frame)
                
                # Write to output video if writer is available
                if writer:
                    writer.write(processed_frame)
                
                yield processed_frame, stats
                
        finally:
            cap.release()
            if writer:
                writer.release()
            self.is_processing = False
            
            logger.info(f"Video processing completed: {self.frame_count} frames")
    
    def process_webcam(self, camera_index: int = 0) -> Generator[Tuple[np.ndarray, CountingStats], None, None]:
        """
        Process webcam stream.
        
        Args:
            camera_index: Camera index (default 0)
            
        Yields:
            Tuple of (frame, stats) for each processed frame
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Starting webcam processing: {frame_width}x{frame_height}")
        
        # Create default counting line if none set
        if not self.counting_service.counting_lines:
            default_line = self.create_default_counting_line(frame_width, frame_height)
            self.set_counting_lines([default_line])
        
        self.is_processing = True
        self.frame_count = 0
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                processed_frame, stats = self.process_frame(frame)
                
                yield processed_frame, stats
                
        finally:
            cap.release()
            logger.info("Webcam processing stopped")
    
    def _draw_results(self, frame: np.ndarray, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """
        Draw detection and tracking results on frame.
        
        Args:
            frame: Input frame
            tracks: Current tracks
            detections: Current detections
            
        Returns:
            Frame with results drawn
        """
        result_frame = frame.copy()
        
        # Draw counting lines
        result_frame = self.counting_service.draw_counting_lines(result_frame)
        
        # Draw tracks
        for track in tracks:
            bbox = track.bbox
            track_id = track.track_id
            vehicle_type = track.vehicle_type.value
            
            # Draw bounding box
            cv2.rectangle(
                result_frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 255),  # Yellow for tracks
                2
            )
            
            # Draw track ID and vehicle type
            label = f"ID:{track_id} {vehicle_type}"
            cv2.putText(
                result_frame,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Draw track history
            if len(track.history) > 1:
                points = [(int(x), int(y)) for x, y in track.history[-10:]]  # Last 10 points
                for i in range(1, len(points)):
                    cv2.line(result_frame, points[i-1], points[i], (255, 0, 0), 2)
        
        # Draw counting statistics
        stats = self.counting_service.get_current_stats()
        self._draw_stats_overlay(result_frame, stats)
        
        return result_frame
    
    def _draw_stats_overlay(self, frame: np.ndarray, stats: CountingStats) -> None:
        """
        Draw statistics overlay on frame.
        
        Args:
            frame: Frame to draw on
            stats: Counting statistics
        """
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        y_offset = 30
        text_lines = [
            f"Vehicles In: {stats.total_in}",
            f"Vehicles Out: {stats.total_out}",
            f"Net Count: {stats.net_count}",
            f"Total: {stats.total_vehicles}",
            f"Active Tracks: {stats.active_tracks}",
            f"FPS: {self.fps:.1f}"
        ]
        
        for line in text_lines:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 20
    
    def stop_processing(self) -> None:
        """Stop video processing."""
        self.is_processing = False
        logger.info("Processing stop requested")
    
    def reset_counts(self) -> None:
        """Reset all counting statistics."""
        self.counting_service.reset_counts()
        self.frame_count = 0
        logger.info("Counts reset")
    
    def get_current_stats(self) -> CountingStats:
        """
        Get current counting statistics.
        
        Returns:
            Current statistics
        """
        return self.counting_service.get_current_stats()
    
    def get_service_info(self) -> dict:
        """
        Get information about service status.
        
        Returns:
            Service information
        """
        return {
            "is_processing": self.is_processing,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "detection_service": self.detection_service.get_model_info(),
            "tracking_service": self.tracking_service.get_tracker_info(),
            "counting_lines": len(self.counting_service.counting_lines)
        }
