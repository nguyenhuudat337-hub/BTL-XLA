#!/usr/bin/env python3
"""
Test script to verify vehicle counting system components.
"""

import sys
import os
sys.path.append('/home/dekiru/Desktop/Onschool/image_processing/vehicle_counting')

import numpy as np
import cv2
from app.services.detection_service import DetectionService
from app.services.tracking_service import TrackingService
from app.services.counting_service import CountingService
from app.services.video_processing_service import VideoProcessingService
from app.models.vehicle import CountingLine

def test_detection_service():
    """Test the detection service."""
    print("🔍 Testing Detection Service...")
    try:
        detection_service = DetectionService()
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detection_service.detect_vehicles(test_image)
        print(f"✅ Detection service working. Found {len(detections)} detections on test image.")
        
        # Get model info
        model_info = detection_service.get_model_info()
        print(f"📊 Model info: {model_info}")
        
        return True
    except Exception as e:
        print(f"❌ Detection service failed: {e}")
        return False

def test_tracking_service():
    """Test the tracking service."""
    print("\n🎯 Testing Tracking Service...")
    try:
        tracking_service = TrackingService()
        
        # Test with empty detections
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = tracking_service.update_tracks([], test_image)
        print(f"✅ Tracking service working. Got {len(tracks)} tracks.")
        
        # Get tracker info
        tracker_info = tracking_service.get_tracker_info()
        print(f"📊 Tracker info: {tracker_info}")
        
        return True
    except Exception as e:
        print(f"❌ Tracking service failed: {e}")
        return False

def test_counting_service():
    """Test the counting service."""
    print("\n📊 Testing Counting Service...")
    try:
        # Create test counting line
        counting_line = CountingLine(
            name="test_line",
            start_point=(0, 240),
            end_point=(640, 240),
            direction="horizontal"
        )
        
        counting_service = CountingService([counting_line])
        
        # Test processing empty tracks
        counts = counting_service.process_tracks([])
        print(f"✅ Counting service working. Current counts: {counts}")
        
        # Get current stats
        stats = counting_service.get_current_stats()
        print(f"📊 Current stats: Total In: {stats.total_in}, Total Out: {stats.total_out}")
        
        return True
    except Exception as e:
        print(f"❌ Counting service failed: {e}")
        return False

def test_video_processing_service():
    """Test the video processing service."""
    print("\n🎬 Testing Video Processing Service...")
    try:
        # Initialize video processing service
        video_service = VideoProcessingService()
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
        
        # Process test frame
        result_frame, stats = video_service.process_frame(test_image)
        print(f"✅ Video processing service working. Processed test frame.")
        print(f"📊 Stats: Total In: {stats.total_in}, Total Out: {stats.total_out}")
        
        # Get service info
        service_info = video_service.get_service_info()
        print(f"📊 Service info: {service_info}")
        
        return True
    except Exception as e:
        print(f"❌ Video processing service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all dependencies are installed."""
    print("📦 Testing Dependencies...")
    
    dependencies = [
        ('ultralytics', 'YOLO'),
        ('cv2', 'OpenCV'),
        ('fastapi', 'FastAPI'),
        ('deep_sort_realtime', 'DeepSORT'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} not found")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("🚗 Vehicle Counting System - Component Tests")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\n❌ Some dependencies are missing. Please install them first.")
        return False
    
    # Test services
    detection_ok = test_detection_service()
    tracking_ok = test_tracking_service()
    counting_ok = test_counting_service()
    video_processing_ok = test_video_processing_service()
    
    print("\n" + "=" * 50)
    if detection_ok and tracking_ok and counting_ok and video_processing_ok:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the system, run:")
        print("   ./start.sh")
        print("\n🌐 Then open: http://localhost:8000/static/index.html")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    main()
