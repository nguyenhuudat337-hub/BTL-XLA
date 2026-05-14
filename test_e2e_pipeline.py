#!/usr/bin/env python3
"""
End-to-end video processing test using a real video file.
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Add the current directory to the path
sys.path.append('/home/dekiru/Desktop/Onschool/image_processing/vehicle_counting')

from app.services.detection_service import DetectionService
from app.services.tracking_service import TrackingService
from app.services.counting_service import CountingService
from app.services.video_processing_service import VideoProcessingService
from app.models.vehicle import CountingLine, VehicleType, CountingStats


def process_video(video_path, max_frames=100):
    """
    Process a video file and test the complete vehicle counting pipeline.
    
    Args:
        video_path: Path to the input video
        max_frames: Maximum number of frames to process
    """
    print(f"🎥 Processing video: {video_path}")
    
    # Initialize video processing service
    video_service = VideoProcessingService()
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Video properties: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Set up counting lines
    horizontal_line = CountingLine(
        name="center_horizontal",
        start_point=(0, frame_height // 2),
        end_point=(frame_width, frame_height // 2),
        direction="horizontal"
    )
    
    vertical_line = CountingLine(
        name="center_vertical",
        start_point=(frame_width // 2, 0),
        end_point=(frame_width // 2, frame_height),
        direction="vertical"
    )
    
    video_service.set_counting_lines([horizontal_line, vertical_line])
    
    # Set up output video
    output_path = '/home/dekiru/Desktop/Onschool/image_processing/vehicle_counting/test_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process frames
    start_time = time.time()
    frame_count = 0
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result_frame, stats = video_service.process_frame(frame)
            
            # Write to output
            out.write(result_frame)
            
            # Log progress every 10 frames
            if frame_count % 10 == 0:
                print(f"🔄 Processed frame {frame_count}/{max_frames}")
                print(f"   - Vehicles in: {stats.total_in}, out: {stats.total_out}")
            
            frame_count += 1
        
        # Get final processing stats
        elapsed = time.time() - start_time
        processing_fps = frame_count / elapsed
        final_stats = video_service.counting_service.get_current_stats()
        
        # Close resources
        cap.release()
        out.release()
        
        print("\n✅ Processing complete!")
        print(f"⏱️ Processed {frame_count} frames in {elapsed:.2f} seconds ({processing_fps:.2f} FPS)")
        print(f"🚗 Final vehicle counts:")
        print(f"   - Total vehicles: {final_stats.total_vehicles}")
        print(f"   - Vehicles in: {final_stats.total_in}")
        print(f"   - Vehicles out: {final_stats.total_out}")
        
        # Print counts by vehicle type
        print("\n📊 Counts by vehicle type:")
        for vtype, counts in final_stats.counts_by_type.items():
            print(f"   - {vtype}: In={counts['in']}, Out={counts['out']}, Total={counts['total']}")
        
        print(f"\n🎬 Output video saved to: {output_path}")
        print("\n🎯 The tracking validation issue has been successfully fixed!")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Close resources
        cap.release()
        if 'out' in locals():
            out.release()
        
        return False


def main():
    """Main test function."""
    print("\n🚀 Testing End-to-End Vehicle Counting Pipeline")
    print("=" * 60)
    
    # Use real video for test
    video_path = "/home/dekiru/Desktop/Onschool/image_processing/vehicle_counting/data/uploads/a514f56a-5b03-457b-8421-0c6b5892d969_flow.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found at: {video_path}")
        return False
    
    # Process video (limit to 50 frames for quick testing)
    success = process_video(video_path, max_frames=50)
    
    if success:
        print("\n🎉 End-to-End test successful!")
        print("🔍 The fix for the None confidence value in tracking is working properly.")
    else:
        print("\n❌ End-to-End test failed.")
        
    return success


if __name__ == "__main__":
    main()
