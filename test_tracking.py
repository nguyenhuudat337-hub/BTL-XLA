"""
Test script for the tracking service to verify the fix for the confidence validation error.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import the app modules
sys.path.append(str(Path(__file__).parent))

from app.services.detection_service import DetectionService
from app.services.tracking_service import TrackingService
from app.services.counting_service import CountingService
from app.models.vehicle import CountingLine


def main():
    """Test the tracking service with a sample video or image."""
    print("Testing tracking service...")
    
    # Initialize services
    detection_service = DetectionService()
    tracking_service = TrackingService()
    counting_service = CountingService()
    
    # We found a sample video, let's use that
    video_path = "/home/dekiru/Desktop/Onschool/image_processing/vehicle_counting/data/uploads/a514f56a-5b03-457b-8421-0c6b5892d969_flow.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return False
    
    print(f"Opened video: {video_path}")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video")
        cap.release()
        return False
    
    print(f"Read first frame with shape {frame.shape}")
    
    try:
        # Run detection
        print("Running detection...")
        detections = detection_service.detect_vehicles(image)
        print(f"Found {len(detections)} vehicles")
        
        # Run tracking
        print("Running tracking...")
        tracks = tracking_service.update_tracks(detections, image)
        print(f"Created {len(tracks)} tracks")
        
        # Verify confidence values
        print("Verifying confidence values...")
        for i, track in enumerate(tracks):
            print(f"Track {i}: ID={track.track_id}, Confidence={track.confidence}, Class={track.class_id}, Type={track.vehicle_type}")
        
        # Set up a counting line
        height, width = image.shape[:2]
        counting_line = CountingLine(
            name="test_line",
            start_point=(0, height // 2),
            end_point=(width, height // 2),
            direction="horizontal"
        )
        counting_service.set_counting_lines([counting_line])
        
        # Process the tracks through counting
        print("Testing counting service...")
        counts = counting_service.process_tracks(tracks)
        stats = counting_service.get_current_stats()
        print(f"Counting statistics: {stats}")
        
        print("Test completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
