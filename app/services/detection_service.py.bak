"""
Vehicle detection service using YOLOv8.
"""

import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
from ultralytics import YOLO
import cv2

from app.core.config import settings
from app.models.vehicle import Detection, BoundingBox, VehicleType


logger = logging.getLogger(__name__)


class DetectionService:
    """Service for vehicle detection using YOLOv8."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detection service.
        
        Args:
            model_path: Path to YOLO model file. If None, uses default from settings.
        """
        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = settings.confidence_threshold
        self.vehicle_classes = settings.vehicle_classes
        self.model: Optional[YOLO] = None
        
        # COCO dataset vehicle class mappings
        self.vehicle_type_mapping = {
            2: VehicleType.CAR,      # car
            3: VehicleType.MOTORCYCLE,  # motorcycle
            5: VehicleType.BUS,      # bus
            7: VehicleType.TRUCK,    # truck
        }
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the YOLO model."""
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not initialize YOLO model: {e}")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input image frame as numpy array
            
        Returns:
            List of vehicle detections
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Filter for vehicle classes only
                        if class_id in self.vehicle_classes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Create detection object
                            bbox = BoundingBox(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2)
                            )
                            
                            detection = Detection(
                                bbox=bbox,
                                confidence=confidence,
                                class_id=class_id,
                                vehicle_type=self._get_vehicle_type(class_id)
                            )
                            
                            detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} vehicles")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _get_vehicle_type(self, class_id: int) -> VehicleType:
        """
        Map class ID to vehicle type.
        
        Args:
            class_id: YOLO class ID
            
        Returns:
            Vehicle type enum
        """
        return self.vehicle_type_mapping.get(class_id, VehicleType.UNKNOWN)
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Visualize detections on frame.
        
        Args:
            frame: Input frame
            detections: List of detections to visualize
            
        Returns:
            Frame with visualized detections
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            bbox = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(
                frame_copy,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),  # Green color
                2
            )
            
            # Draw label
            label = f"{detection.vehicle_type.value}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(
                frame_copy,
                (int(bbox.x1), int(bbox.y1) - label_size[1] - 10),
                (int(bbox.x1) + label_size[0], int(bbox.y1)),
                (0, 255, 0),
                -1
            )
            
            cv2.putText(
                frame_copy,
                label,
                (int(bbox.x1), int(bbox.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame_copy
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Get list of supported vehicle types
        vehicle_types = [vt.value for vt in set(self.vehicle_type_mapping.values())]
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "vehicle_classes": self.vehicle_classes,
            "model_size": getattr(self.model, 'model_size', 'unknown'),
            "supported_vehicle_types": vehicle_types
        }
