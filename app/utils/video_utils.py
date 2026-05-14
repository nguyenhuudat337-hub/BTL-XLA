"""
Video processing utilities for the vehicle counting system.
"""

import logging
import os
import uuid
from typing import Generator, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path

from app.core.config import settings


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self):
        """Initialize the video processor."""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate if the video file exists and is supported.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > settings.max_video_size_mb:
            logger.error(f"Video file too large: {file_size_mb}MB > {settings.max_video_size_mb}MB")
            return False
        
        # Check file extension
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported video format: {file_ext}")
            return False
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            cap.release()
            return False
        
        cap.release()
        return True
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video file information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not self.validate_video_file(video_path):
            return {}
        
        cap = cv2.VideoCapture(video_path)
        
        info = {
            "file_path": video_path,
            "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": 0
        }
        
        if info["fps"] > 0:
            info["duration_seconds"] = round(info["total_frames"] / info["fps"], 2)
        
        cap.release()
        return info
    
    def read_frames(
        self, 
        video_path: str, 
        start_frame: int = 0, 
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator to read frames from video file.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            max_frames: Maximum number of frames to read
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        if not self.validate_video_file(video_path):
            return
        
        cap = cv2.VideoCapture(video_path)
        
        # Set starting frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        current_frame = start_frame
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                yield current_frame, frame
                
                frame_count += 1
                current_frame += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                    
        except Exception as e:
            logger.error(f"Error reading frames: {e}")
        finally:
            cap.release()
    
    def create_video_writer(
        self, 
        output_path: str, 
        width: int, 
        height: int, 
        fps: float = None
    ) -> cv2.VideoWriter:
        """
        Create a video writer for output video.
        
        Args:
            output_path: Output video file path
            width: Video width
            height: Video height
            fps: Frames per second (uses default if None)
            
        Returns:
            OpenCV VideoWriter object
        """
        if fps is None:
            fps = settings.output_fps
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")
        
        return writer
    
    def resize_frame(
        self, 
        frame: np.ndarray, 
        target_width: int, 
        target_height: int, 
        maintain_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Resize frame to target dimensions.
        
        Args:
            frame: Input frame
            target_width: Target width
            target_height: Target height
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        if maintain_aspect_ratio:
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            
            if target_width / target_height > aspect_ratio:
                # Height is the constraining dimension
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            else:
                # Width is the constraining dimension
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            
            resized = cv2.resize(frame, (new_width, new_height))
            
            # Create canvas with target dimensions
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Center the resized frame
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
        else:
            return cv2.resize(frame, (target_width, target_height))
    
    def generate_output_path(self, input_path: str, suffix: str = "_processed") -> str:
        """
        Generate output file path based on input path.
        
        Args:
            input_path: Input video file path
            suffix: Suffix to add to filename
            
        Returns:
            Output file path
        """
        input_file = Path(input_path)
        timestamp = uuid.uuid4().hex[:8]
        output_filename = f"{input_file.stem}{suffix}_{timestamp}.mp4"
        
        # Ensure output directory exists
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir / output_filename)
    
    def extract_frame_at_time(self, video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at specified time.
        
        Args:
            video_path: Path to video file
            time_seconds: Time in seconds
            
        Returns:
            Frame array or None if failed
        """
        if not self.validate_video_file(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        # Set position to specified time
        cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def create_thumbnail(
        self, 
        video_path: str, 
        thumbnail_path: str, 
        time_seconds: float = 5.0,
        width: int = 320,
        height: int = 240
    ) -> bool:
        """
        Create a thumbnail image from video.
        
        Args:
            video_path: Path to video file
            thumbnail_path: Output thumbnail path
            time_seconds: Time in seconds to extract frame
            width: Thumbnail width
            height: Thumbnail height
            
        Returns:
            True if successful, False otherwise
        """
        frame = self.extract_frame_at_time(video_path, time_seconds)
        
        if frame is None:
            return False
        
        # Resize frame
        thumbnail = self.resize_frame(frame, width, height)
        
        # Save thumbnail
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        success = cv2.imwrite(thumbnail_path, thumbnail)
        
        return success
    
    def get_frame_at_position(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """
        Get frame at specific frame number.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract
            
        Returns:
            Frame array or None if failed
        """
        if not self.validate_video_file(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None


# Utility functions
def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_video_codec_info(video_path: str) -> dict:
    """
    Get video codec information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with codec information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}
    
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    
    info = {
        "codec": codec,
        "fourcc": int(fourcc),
        "backend": cap.getBackendName()
    }
    
    cap.release()
    return info