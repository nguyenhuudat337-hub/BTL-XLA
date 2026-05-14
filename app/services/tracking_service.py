"""
Vehicle tracking service using DeepSORT.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from app.core.config import settings
from app.models.vehicle import Detection, Track, BoundingBox, VehicleType


logger = logging.getLogger(__name__)


class TrackingService:
    """Service for vehicle tracking using DeepSORT."""
    
    def __init__(self):
        """Initialize the tracking service."""
        self.max_age = settings.max_age
        self.min_hits = settings.min_hits
        self.iou_threshold = settings.iou_threshold
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            max_iou_distance=self.iou_threshold,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Track history for analysis
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        
        logger.info("DeepSORT tracker initialized")
    
    def update_tracks(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of vehicle detections
            frame: Current frame for feature extraction
            
        Returns:
            List of updated tracks
        """
        try:
            # Convert detections to DeepSORT format
            detection_list = []
            for detection in detections:
                bbox = [
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2 - detection.bbox.x1,  # width
                    detection.bbox.y2 - detection.bbox.y1   # height
                ]
                detection_list.append((bbox, detection.confidence, detection.class_id))
            
            # Update tracker
            tracks = self.tracker.update_tracks(detection_list, frame=frame)
            
            # Convert to our Track objects
            track_objects = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltwh = track.to_ltwh()
                
                # Create bounding box
                bbox = BoundingBox(
                    x1=float(ltwh[0]),
                    y1=float(ltwh[1]),
                    x2=float(ltwh[0] + ltwh[2]),
                    y2=float(ltwh[1] + ltwh[3])
                )
                
                # Get center point for history
                center_x = bbox.x1 + (bbox.x2 - bbox.x1) / 2
                center_y = bbox.y1 + (bbox.y2 - bbox.y1) / 2
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((center_x, center_y))
                
                # Limit history length
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id] = self.track_history[track_id][-30:]
                
                # Create track object
                # Get detection confidence, ensuring it's not None
                conf = 0.0
                if hasattr(track, 'get_det_conf'):
                    conf_value = track.get_det_conf()
                    if conf_value is not None:
                        conf = conf_value
                    else:
                        logger.warning(f"Track {track_id} has None confidence, using default 0.0")
                
                # Get detection class, ensuring it's not None
                class_id = 0
                if hasattr(track, 'get_det_class'):
                    class_value = track.get_det_class()
                    if class_value is not None:
                        class_id = class_value
                    
                track_obj = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    vehicle_type=self._get_vehicle_type(class_id),
                    center_point=(center_x, center_y),
                    history=self.track_history[track_id].copy()
                )
                
                track_objects.append(track_obj)
            
            logger.debug(f"Updated {len(track_objects)} tracks")
            return track_objects
            
        except Exception as e:
            logger.error(f"Track update failed: {e}")
            return []
    
    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get movement history for a specific track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            List of (x, y) center points
        """
        return self.track_history.get(track_id, [])
    
    def clear_old_tracks(self) -> None:
        """Remove old track histories to prevent memory issues."""
        # Get active track IDs from current tracker state
        active_tracks = set()
        try:
            for track in self.tracker.tracks:
                if track.is_confirmed():
                    active_tracks.add(track.track_id)
        except:
            pass
        
        # Remove inactive tracks
        inactive_tracks = set(self.track_history.keys()) - active_tracks
        for track_id in inactive_tracks:
            del self.track_history[track_id]
        
        logger.debug(f"Cleared {len(inactive_tracks)} inactive track histories")
    
    def _get_vehicle_type(self, class_id: int) -> VehicleType:
        """
        Map class ID to vehicle type.
        
        Args:
            class_id: YOLO class ID
            
        Returns:
            Vehicle type enum
        """
        vehicle_type_mapping = {
            2: VehicleType.CAR,
            3: VehicleType.MOTORCYCLE,
            5: VehicleType.BUS,
            7: VehicleType.TRUCK,
        }
        
        return vehicle_type_mapping.get(class_id, VehicleType.CAR)
    
    def get_tracker_info(self) -> dict:
        """
        Get information about the tracker state.
        
        Returns:
            Tracker information dictionary
        """
        active_tracks = 0
        try:
            active_tracks = len([t for t in self.tracker.tracks if t.is_confirmed()])
        except:
            pass
        
        return {
            "active_tracks": active_tracks,
            "total_track_histories": len(self.track_history),
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "iou_threshold": self.iou_threshold
        }