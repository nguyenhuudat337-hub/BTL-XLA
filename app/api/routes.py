"""
API routes for the vehicle counting system.
"""

import logging
import os
import time
import uuid
import asyncio
import json
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64

from app.core.config import settings
from app.models.vehicle import (
    ProcessingResult, VehicleCount, CountingLine, VideoProcessRequest, CountingStats
)
from app.services.video_processing_service import VideoProcessingService


logger = logging.getLogger(__name__)
router = APIRouter()

# Global video processing service
video_service = VideoProcessingService()

# Store processing results temporarily
processing_results = {}
active_websockets = set()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version,
        "timestamp": time.time(),
        "detection_service": video_service.detection_service.get_model_info(),
        "tracking_service": video_service.tracking_service.get_tracker_info()
    }


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        return video_service.get_service_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_current_stats():
    """Get current counting statistics."""
    try:
        stats = video_service.get_current_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-counts")
async def reset_counts():
    """Reset all counting statistics."""
    try:
        video_service.reset_counts()
        return {"message": "Counts reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting counts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counting-lines")
async def set_counting_lines(lines: List[CountingLine]):
    """Set counting lines for vehicle counting."""
    try:
        video_service.set_counting_lines(lines)
        return {"message": f"Set {len(lines)} counting lines successfully"}
    except Exception as e:
        logger.error(f"Error setting counting lines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-video")
async def process_video_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a video file."""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        upload_path = os.path.join(settings.upload_dir, f"{job_id}_{file.filename}")
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add background task for processing
        output_path = os.path.join(settings.output_dir, f"{job_id}_processed.mp4")
        background_tasks.add_task(process_video_task, job_id, upload_path, output_path)
        
        return {
            "job_id": job_id,
            "message": "Video uploaded and processing started",
            "status_url": f"/api/v1/job-status/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Error processing video upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_task(job_id: str, input_path: str, output_path: str):
    """Background task for video processing."""
    try:
        processing_results[job_id] = {
            "status": "processing",
            "progress": 0,
            "start_time": time.time()
        }
        
        os.makedirs(settings.output_dir, exist_ok=True)
        
        # Process video
        frame_count = 0
        for processed_frame, stats in video_service.process_video_file(input_path, output_path):
            frame_count += 1
            processing_results[job_id]["progress"] = frame_count
            
            # Update every 30 frames to avoid excessive updates
            if frame_count % 30 == 0:
                processing_results[job_id]["current_stats"] = stats.dict()
        
        # Final results
        final_stats = video_service.get_current_stats()
        processing_results[job_id].update({
            "status": "completed",
            "progress": frame_count,
            "final_stats": final_stats.dict(),
            "output_path": output_path,
            "processing_time": time.time() - processing_results[job_id]["start_time"]
        })
        
        logger.info(f"Video processing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {e}")
        processing_results[job_id] = {
            "status": "failed",
            "error": str(e),
            "processing_time": time.time() - processing_results[job_id]["start_time"]
        }
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)


@router.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a video processing job."""
    if job_id not in processing_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_results[job_id]


@router.get("/download-result/{job_id}")
async def download_result(job_id: str):
    """Download processed video result."""
    if job_id not in processing_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = processing_results[job_id]
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = result.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )


@router.websocket("/webcam-stream")
async def webcam_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time webcam processing."""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        logger.info("Starting webcam stream")
        
        for processed_frame, stats in video_service.process_webcam():
            try:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame and stats
                message = {
                    "type": "frame",
                    "frame": frame_base64,
                    "stats": stats.dict(),
                    "timestamp": time.time()
                }
                
                await websocket.send_text(json.dumps(message))
                
                # Check for client messages (e.g., stop command)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    command = json.loads(data)
                    if command.get("action") == "stop":
                        break
                except asyncio.TimeoutError:
                    pass
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in webcam stream: {e}")
                break
                
    except Exception as e:
        logger.error(f"Webcam stream error: {e}")
    finally:
        active_websockets.discard(websocket)
        video_service.stop_processing()
        logger.info("Webcam stream ended")


@router.websocket("/live-stats")
async def live_stats(websocket: WebSocket):
    """WebSocket endpoint for live statistics updates."""
    await websocket.accept()
    
    try:
        while True:
            stats = video_service.get_current_stats()
            await websocket.send_text(json.dumps({
                "type": "stats",
                "data": stats.dict(),
                "timestamp": time.time()
            }))
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Live stats WebSocket error: {e}")


@router.post("/camera/start")
async def start_camera_processing():
    """Start camera processing without WebSocket."""
    try:
        if video_service.is_processing:
            return {"message": "Camera processing already running"}
        
        # Start processing in background
        asyncio.create_task(camera_processing_task())
        
        return {"message": "Camera processing started"}
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/camera/stop")
async def stop_camera_processing():
    """Stop camera processing."""
    try:
        video_service.stop_processing()
        return {"message": "Camera processing stopped"}
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def camera_processing_task():
    """Background task for camera processing."""
    try:
        for processed_frame, stats in video_service.process_webcam():
            # Just process frames, stats are available via /stats endpoint
            pass
    except Exception as e:
        logger.error(f"Camera processing task error: {e}")


@router.get("/test-detection")
async def test_detection():
    """Test endpoint to verify detection service is working."""
    try:
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = video_service.detection_service.detect_vehicles(test_image)
        
        return {
            "message": "Detection service is working",
            "detections_found": len(detections),
            "model_info": video_service.detection_service.get_model_info()
        }
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing.
    
    Args:
        file: Uploaded video file
        
    Returns:
        Video upload information
    """
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique filename
    file_id = uuid.uuid4().hex
    file_extension = os.path.splitext(file.filename)[1]
    upload_path = os.path.join(settings.upload_dir, f"{file_id}{file_extension}")
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    try:
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate video file
        if not video_processor.validate_video_file(upload_path):
            os.remove(upload_path)
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Get video information
        video_info = video_processor.get_video_info(upload_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "upload_path": upload_path,
            "video_info": video_info
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail="Failed to upload video")


@router.post("/process-video", response_model=ProcessingResult)
async def process_video(
    background_tasks: BackgroundTasks,
    request: VideoProcessRequest
):
    """
    Process a video for vehicle counting.
    
    Args:
        background_tasks: FastAPI background tasks
        request: Video processing request
        
    Returns:
        Processing result
    """
    try:
        # Validate video file
        if not video_processor.validate_video_file(request.video_path):
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Generate processing ID
        processing_id = uuid.uuid4().hex
        
        # Start background processing
        background_tasks.add_task(
            _process_video_background,
            processing_id,
            request
        )
        
        return ProcessingResult(
            video_id=processing_id,
            total_frames=0,
            processing_time=0,
            final_count=VehicleCount(
                total_vehicles=0,
                vehicles_in=0,
                vehicles_out=0,
                count_by_type={},
                frame_number=0,
                timestamp=0
            ),
            success=True,
            error_message=None
        )
        
    except Exception as e:
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start video processing")


async def _process_video_background(processing_id: str, request: VideoProcessRequest):
    """
    Background task for video processing.
    
    Args:
        processing_id: Unique processing identifier
        request: Video processing request
    """
    start_time = time.time()
    
    try:
        # Initialize services
        tracking_service = TrackingService()
        counting_service = CountingService(request.counting_lines)
        
        # Get video info
        video_info = video_processor.get_video_info(request.video_path)
        total_frames = video_info.get("total_frames", 0)
        
        # Set default counting line if none provided
        if not request.counting_lines:
            counting_service.set_default_counting_line(
                video_info["width"], video_info["height"]
            )
        
        # Initialize video writer if needed
        video_writer = None
        output_path = None
        
        if request.save_output_video:
            output_path = video_processor.generate_output_path(request.video_path)
            video_writer = video_processor.create_video_writer(
                output_path,
                video_info["width"],
                video_info["height"],
                video_info["fps"]
            )
        
        # Process frames
        frame_count = 0
        final_count = None
        
        for frame_number, frame in video_processor.read_frames(request.video_path):
            # Detect vehicles
            detections = detection_service.detect_vehicles(frame)
            
            # Update tracking
            tracks = tracking_service.update_tracks(detections)
            
            # Update counting
            current_time = frame_number / video_info.get("fps", 30)
            final_count = counting_service.update_counts(tracks, frame_number, current_time)
            
            # Visualize if saving output
            if video_writer:
                # Draw detections
                frame_vis = detection_service.visualize_detections(frame, detections)
                
                # Draw counting lines
                frame_vis = counting_service.visualize_counting_lines(frame_vis)
                
                # Draw counts
                frame_vis = counting_service.visualize_counts(frame_vis, final_count)
                
                # Write frame
                video_writer.write(frame_vis)
            
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Processing {processing_id}: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        if video_writer:
            video_writer.release()
        
        processing_time = time.time() - start_time
        
        # Store result
        result = ProcessingResult(
            video_id=processing_id,
            total_frames=frame_count,
            processing_time=processing_time,
            final_count=final_count or VehicleCount(
                total_vehicles=0,
                vehicles_in=0,
                vehicles_out=0,
                count_by_type={},
                frame_number=frame_count,
                timestamp=processing_time
            ),
            output_video_path=output_path,
            success=True
        )
        
        processing_results[processing_id] = result
        logger.info(f"Processing {processing_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing video {processing_id}: {e}")
        processing_results[processing_id] = ProcessingResult(
            video_id=processing_id,
            total_frames=0,
            processing_time=time.time() - start_time,
            final_count=VehicleCount(
                total_vehicles=0,
                vehicles_in=0,
                vehicles_out=0,
                count_by_type={},
                frame_number=0,
                timestamp=0
            ),
            success=False,
            error_message=str(e)
        )


@router.get("/processing-status/{processing_id}")
async def get_processing_status(processing_id: str):
    """
    Get the status of a video processing task.
    
    Args:
        processing_id: Processing task identifier
        
    Returns:
        Processing status and result
    """
    if processing_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing task not found")
    
    result = processing_results[processing_id]
    return result


@router.get("/download-result/{processing_id}")
async def download_result(processing_id: str):
    """
    Download the processed video result.
    
    Args:
        processing_id: Processing task identifier
        
    Returns:
        Processed video file
    """
    if processing_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing task not found")
    
    result = processing_results[processing_id]
    
    if not result.success or not result.output_video_path:
        raise HTTPException(status_code=400, detail="No output video available")
    
    if not os.path.exists(result.output_video_path):
        raise HTTPException(status_code=404, detail="Output video file not found")
    
    return FileResponse(
        path=result.output_video_path,
        media_type='video/mp4',
        filename=f"processed_{processing_id}.mp4"
    )


@router.post("/process-frame")
async def process_single_frame(file: UploadFile = File(...)):
    """
    Process a single image frame for vehicle detection.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detection results
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect vehicles
        detections = detection_service.detect_vehicles(frame)
        
        # Convert detections to response format
        detection_results = []
        for detection in detections:
            detection_results.append({
                "bbox": {
                    "x1": detection.bbox.x1,
                    "y1": detection.bbox.y1,
                    "x2": detection.bbox.x2,
                    "y2": detection.bbox.y2
                },
                "confidence": detection.confidence,
                "vehicle_type": detection.vehicle_type.value,
                "class_id": detection.class_id
            })
        
        return {
            "detections": detection_results,
            "total_vehicles": len(detection_results),
            "image_shape": frame.shape
        }
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail="Failed to process frame")


@router.delete("/cleanup/{processing_id}")
async def cleanup_processing_data(processing_id: str):
    """
    Clean up processing data and files.
    
    Args:
        processing_id: Processing task identifier
        
    Returns:
        Cleanup status
    """
    if processing_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing task not found")
    
    try:
        result = processing_results[processing_id]
        
        # Remove output video file if exists
        if result.output_video_path and os.path.exists(result.output_video_path):
            os.remove(result.output_video_path)
        
        # Remove from memory
        del processing_results[processing_id]
        
        return {"status": "cleaned", "processing_id": processing_id}
        
    except Exception as e:
        logger.error(f"Error cleaning up processing data: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup processing data")


@router.get("/statistics")
async def get_system_statistics():
    """
    Get system statistics and performance metrics.
    
    Returns:
        System statistics
    """
    try:
        model_info = detection_service.get_model_info()
        
        return {
            "model_status": model_info.get("status", "unknown"),
            "active_processings": len(processing_results),
            "supported_formats": video_processor.supported_formats,
            "max_video_size_mb": settings.max_video_size_mb,
            "output_fps": settings.output_fps,
            "confidence_threshold": settings.confidence_threshold
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")