"""
Vehicle counting service with crossing line detection.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
import math
import cv2
import numpy as np
from datetime import datetime

from app.models.vehicle import (
    Track, CountingLine, VehicleCount, VehicleType, CountingStats
)


logger = logging.getLogger(__name__)


class CountingService:
    """Service for counting vehicles crossing defined lines."""
    
    def __init__(self, counting_lines: Optional[List[CountingLine]] = None):
        """
        Initialize the counting service.
        
        Args:
            counting_lines: List of counting lines. If None, uses default center line.
        """
        self.counting_lines = counting_lines or []
        self.vehicle_counts: Dict[str, VehicleCount] = {}
        self.crossed_tracks: Dict[int, Set[str]] = {}  # track_id -> set of crossed line directions
        self.track_positions: Dict[int, List[Tuple[float, float]]] = {}
        
        # Global counters
        self.total_vehicles_in = 0
        self.total_vehicles_out = 0
        self.count_by_type: Dict[VehicleType, Dict[str, int]] = {
            vehicle_type: {"in": 0, "out": 0} 
            for vehicle_type in VehicleType
        }
        
        # Session tracking
        self.session_start = datetime.now()
        self.frame_count = 0
        
        logger.info(f"Counting service initialized with {len(self.counting_lines)} lines")
    
    def set_counting_lines(self, counting_lines: List[CountingLine]) -> None:
        """
        Set or update counting lines.
        
        Args:
            counting_lines: List of counting lines
        """
        self.counting_lines = counting_lines
        logger.info(f"Updated counting lines: {len(counting_lines)} lines")
    
    def process_tracks(self, tracks: List[Track]) -> Dict[str, int]:
        """
        Process tracks and count vehicles crossing lines.
        
        Args:
            tracks: List of current tracks
            
        Returns:
            Dictionary with current frame counts
        """
        self.frame_count += 1
        current_counts = {"in": 0, "out": 0}
        
        for track in tracks:
            track_id = track.track_id
            center_x, center_y = track.center_point
            
            # Update track position history
            if track_id not in self.track_positions:
                self.track_positions[track_id] = []
            
            self.track_positions[track_id].append((center_x, center_y))
            
            # Limit position history
            if len(self.track_positions[track_id]) > 10:
                self.track_positions[track_id] = self.track_positions[track_id][-10:]
            
            # Check line crossings for each counting line
            for line in self.counting_lines:
                crossing_result = self._check_line_crossing(track_id, track, line)
                if crossing_result:
                    direction, vehicle_type = crossing_result
                    self._update_counts(direction, vehicle_type)
                    current_counts[direction] += 1
                    
                    logger.info(f"Vehicle {track_id} ({vehicle_type.value}) crossed {line.name} - {direction}")
        
        return current_counts
    
    def _check_line_crossing(self, track_id: int, track: Track, line: CountingLine) -> Optional[Tuple[str, VehicleType]]:
        """
        Check if a track has crossed a counting line.
        
        Args:
            track_id: Track identifier
            track: Track object
            line: Counting line to check
            
        Returns:
            Tuple of (direction, vehicle_type) if crossed, None otherwise
        """
        if track_id not in self.track_positions or len(self.track_positions[track_id]) < 2:
            return None
        
        positions = self.track_positions[track_id]
        current_pos = positions[-1]
        previous_pos = positions[-2]
        
        # Check if line was crossed between previous and current position
        crossed = self._line_intersection(previous_pos, current_pos, line.start_point, line.end_point)
        
        if crossed:
            # Initialize crossed tracks set for this track if needed
            if track_id not in self.crossed_tracks:
                self.crossed_tracks[track_id] = set()
            
            # Determine direction based on line orientation and crossing direction
            direction = self._determine_crossing_direction(previous_pos, current_pos, line)
            direction_key = f"{line.name}_{direction}"
            
            # Only count if this track hasn't crossed this line in this direction before
            if direction_key not in self.crossed_tracks[track_id]:
                self.crossed_tracks[track_id].add(direction_key)
                return direction, track.vehicle_type
        
        return None
    
    def _line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """
        Check if two line segments intersect.
        
        Args:
            p1, p2: Points defining first line segment (track movement)
            p3, p4: Points defining second line segment (counting line)
            
        Returns:
            True if lines intersect, False otherwise
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _determine_crossing_direction(self, prev_pos: Tuple[float, float], 
                                    curr_pos: Tuple[float, float], 
                                    line: CountingLine) -> str:
        """
        Determine the direction of crossing (in/out) based on line orientation.
        
        Args:
            prev_pos: Previous position
            curr_pos: Current position
            line: Counting line
            
        Returns:
            Direction string ("in" or "out")
        """
        # Calculate the cross product to determine which side of the line the movement is
        line_vector = (line.end_point[0] - line.start_point[0], line.end_point[1] - line.start_point[1])
        movement_vector = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        
        # Cross product
        cross_product = line_vector[0] * movement_vector[1] - line_vector[1] * movement_vector[0]
        
        # Determine direction based on cross product sign and line direction
        if line.direction == "horizontal":
            return "in" if cross_product > 0 else "out"
        else:  # vertical
            return "in" if cross_product < 0 else "out"
    
    def _update_counts(self, direction: str, vehicle_type: VehicleType) -> None:
        """
        Update global vehicle counts.
        
        Args:
            direction: Direction of crossing ("in" or "out")
            vehicle_type: Type of vehicle
        """
        if direction == "in":
            self.total_vehicles_in += 1
        else:
            self.total_vehicles_out += 1
        
        self.count_by_type[vehicle_type][direction] += 1
    
    def get_current_stats(self) -> CountingStats:
        """
        Get current counting statistics.
        
        Returns:
            Current counting statistics
        """
        # Calculate total by type
        total_by_type = {}
        for vehicle_type in VehicleType:
            total_by_type[vehicle_type.value] = {
                "in": self.count_by_type[vehicle_type]["in"],
                "out": self.count_by_type[vehicle_type]["out"],
                "total": self.count_by_type[vehicle_type]["in"] + self.count_by_type[vehicle_type]["out"]
            }
        
        # Calculate session duration
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return CountingStats(
            total_in=self.total_vehicles_in,
            total_out=self.total_vehicles_out,
            total_vehicles=self.total_vehicles_in + self.total_vehicles_out,
            net_count=self.total_vehicles_in - self.total_vehicles_out,
            counts_by_type=total_by_type,
            session_duration=session_duration,
            frames_processed=self.frame_count,
            active_tracks=len(self.track_positions),
            counting_lines=len(self.counting_lines)
        )
    
    def reset_counts(self) -> None:
        """Reset all counting statistics."""
        self.total_vehicles_in = 0
        self.total_vehicles_out = 0
        self.count_by_type = {
            vehicle_type: {"in": 0, "out": 0} 
            for vehicle_type in VehicleType
        }
        self.crossed_tracks.clear()
        self.track_positions.clear()
        self.session_start = datetime.now()
        self.frame_count = 0
        
        logger.info("Vehicle counts reset")
    
    def cleanup_old_tracks(self, active_track_ids: Set[int]) -> None:
        """
        Clean up data for tracks that are no longer active.
        
        Args:
            active_track_ids: Set of currently active track IDs
        """
        # Remove old track positions
        old_tracks = set(self.track_positions.keys()) - active_track_ids
        for track_id in old_tracks:
            if track_id in self.track_positions:
                del self.track_positions[track_id]
            if track_id in self.crossed_tracks:
                del self.crossed_tracks[track_id]
        
        if old_tracks:
            logger.debug(f"Cleaned up {len(old_tracks)} old tracks")
    
    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw counting lines on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with counting lines drawn
        """
        result_frame = frame.copy()
        
        for line in self.counting_lines:
            # Draw line
            cv2.line(
                result_frame,
                tuple(map(int, line.start_point)),
                tuple(map(int, line.end_point)),
                (0, 255, 0),  # Green color
                3
            )
            
            # Draw line name
            text_pos = (
                int((line.start_point[0] + line.end_point[0]) / 2),
                int((line.start_point[1] + line.end_point[1]) / 2) - 10
            )
            cv2.putText(
                result_frame,
                line.name,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return result_frame
        self.reset_counts()
        logger.info(f"Updated counting lines: {len(counting_lines)} lines")
    
    def set_default_counting_line(self, frame_width: int, frame_height: int) -> None:
        """
        Set a default vertical counting line in the center of the frame.
        
        Args:
            frame_width: Frame width
            frame_height: Frame height
        """
        center_x = frame_width // 2
        default_line = CountingLine(
            start_point=(center_x, 0),
            end_point=(center_x, frame_height),
            direction="vertical"
        )
        self.counting_lines = [default_line]
        self.reset_counts()
        logger.info(f"Set default counting line at x={center_x}")
    
    def update_counts(
        self, 
        tracks: List[Track], 
        frame_number: int, 
        timestamp: float
    ) -> VehicleCount:
        """
        Update vehicle counts based on track positions.
        
        Args:
            tracks: List of current tracks
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            Updated vehicle count
        """
        if not self.counting_lines:
            logger.warning("No counting lines defined")
            return self._create_vehicle_count(frame_number, timestamp)
        
        for track in tracks:
            if not track.is_confirmed or track.current_detection is None:
                continue
            
            track_id = track.track_id
            current_position = track.current_detection.bbox.center
            
            # Initialize track position history
            if track_id not in self.track_positions:
                self.track_positions[track_id] = []
                self.crossed_tracks[track_id] = set()
            
            # Add current position to history
            self.track_positions[track_id].append(current_position)
            
            # Keep only recent positions (last 10 frames)
            if len(self.track_positions[track_id]) > 10:
                self.track_positions[track_id] = self.track_positions[track_id][-10:]
            
            # Check line crossings
            self._check_line_crossings(track, frame_number)
        
        return self._create_vehicle_count(frame_number, timestamp)
    
    def _check_line_crossings(self, track: Track, frame_number: int) -> None:
        """
        Check if a track has crossed any counting lines.
        
        Args:
            track: Track to check
            frame_number: Current frame number
        """
        track_id = track.track_id
        positions = self.track_positions.get(track_id, [])
        
        if len(positions) < 2:
            return
        
        # Get last two positions
        prev_pos = positions[-2]
        curr_pos = positions[-1]
        
        for i, line in enumerate(self.counting_lines):
            line_key = f"line_{i}_{line.direction}"
            
            # Skip if already crossed this line
            if line_key in self.crossed_tracks[track_id]:
                continue
            
            # Check if the track crossed the line
            crossed, direction = self._check_line_intersection(
                prev_pos, curr_pos, line
            )
            
            if crossed:
                self.crossed_tracks[track_id].add(line_key)
                vehicle_type = track.current_detection.vehicle_type
                
                # Update counters
                if direction == "in":
                    self.total_vehicles_in += 1
                    self.count_by_type[vehicle_type]["in"] += 1
                elif direction == "out":
                    self.total_vehicles_out += 1
                    self.count_by_type[vehicle_type]["out"] += 1
                
                logger.info(
                    f"Track {track_id} ({vehicle_type.value}) crossed {line_key} "
                    f"direction: {direction} at frame {frame_number}"
                )
    
    def _check_line_intersection(
        self, 
        pos1: Tuple[float, float], 
        pos2: Tuple[float, float], 
        line: CountingLine
    ) -> Tuple[bool, str]:
        """
        Check if a track segment intersects with a counting line.
        
        Args:
            pos1: Previous position (x, y)
            pos2: Current position (x, y)
            line: Counting line
            
        Returns:
            Tuple of (intersected: bool, direction: str)
        """
        # Line segment from track movement
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Counting line
        x3, y3 = line.start_point
        x4, y4 = line.end_point
        
        # Check for intersection using line segment intersection formula
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Lines are parallel
            return False, ""
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection point is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Determine direction based on line orientation and crossing direction
            direction = self._determine_crossing_direction(pos1, pos2, line)
            return True, direction
        
        return False, ""
    
    def _determine_crossing_direction(
        self, 
        pos1: Tuple[float, float], 
        pos2: Tuple[float, float], 
        line: CountingLine
    ) -> str:
        """
        Determine the crossing direction (in/out) based on movement and line orientation.
        
        Args:
            pos1: Previous position
            pos2: Current position
            line: Counting line
            
        Returns:
            Direction string ("in" or "out")
        """
        x1, y1 = pos1
        x2, y2 = pos2
        line_x1, line_y1 = line.start_point
        line_x2, line_y2 = line.end_point
        
        # Calculate line vector and movement vector
        line_vec = (line_x2 - line_x1, line_y2 - line_y1)
        movement_vec = (x2 - x1, y2 - y1)
        
        # Cross product to determine which side of the line the movement is going
        cross_product = line_vec[0] * movement_vec[1] - line_vec[1] * movement_vec[0]
        
        # Direction mapping based on line direction setting
        if line.direction == "vertical":
            return "in" if cross_product > 0 else "out"
        elif line.direction == "horizontal":
            return "in" if cross_product < 0 else "out"
        else:
            # For custom directions, use the line's direction field
            return line.direction if cross_product > 0 else ("out" if line.direction == "in" else "in")
    
    def _create_vehicle_count(self, frame_number: int, timestamp: float) -> VehicleCount:
        """
        Create a VehicleCount object with current counts.
        
        Args:
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            VehicleCount object
        """
        total_by_type = {}
        for vehicle_type, counts in self.count_by_type.items():
            total_count = counts["in"] + counts["out"]
            if total_count > 0:
                total_by_type[vehicle_type] = total_count
        
        return VehicleCount(
            total_vehicles=self.total_vehicles_in + self.total_vehicles_out,
            vehicles_in=self.total_vehicles_in,
            vehicles_out=self.total_vehicles_out,
            count_by_type=total_by_type,
            frame_number=frame_number,
            timestamp=timestamp
        )
    
    def visualize_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw counting lines on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with counting lines drawn
        """
        frame_copy = frame.copy()
        
        for i, line in enumerate(self.counting_lines):
            start_point = (int(line.start_point[0]), int(line.start_point[1]))
            end_point = (int(line.end_point[0]), int(line.end_point[1]))
            
            # Draw line
            cv2.line(frame_copy, start_point, end_point, (0, 0, 255), 3)  # Red line
            
            # Draw direction label
            mid_x = (start_point[0] + end_point[0]) // 2
            mid_y = (start_point[1] + end_point[1]) // 2
            
            cv2.putText(
                frame_copy,
                f"Line {i+1}: {line.direction}",
                (mid_x - 50, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        return frame_copy
    
    def visualize_counts(self, frame: np.ndarray, current_count: VehicleCount) -> np.ndarray:
        """
        Draw count information on the frame.
        
        Args:
            frame: Input frame
            current_count: Current vehicle count
            
        Returns:
            Frame with count information
        """
        frame_copy = frame.copy()
        
        # Background for text
        cv2.rectangle(frame_copy, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (10, 10), (400, 150), (255, 255, 255), 2)
        
        # Count text
        y_offset = 35
        cv2.putText(frame_copy, f"Total: {current_count.total_vehicles}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame_copy, f"In: {current_count.vehicles_in}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame_copy, f"Out: {current_count.vehicles_out}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Count by type
        y_offset += 30
        for vehicle_type, count in current_count.count_by_type.items():
            if count > 0:
                cv2.putText(frame_copy, f"{vehicle_type.value}: {count}", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
        
        return frame_copy
    
    def reset_counts(self) -> None:
        """Reset all counts and tracking data."""
        self.total_vehicles_in = 0
        self.total_vehicles_out = 0
        self.count_by_type = {
            vehicle_type: {"in": 0, "out": 0} 
            for vehicle_type in VehicleType
        }
        self.crossed_tracks.clear()
        self.track_positions.clear()
        logger.info("Counts reset")
    
    def get_final_counts(self) -> Dict[str, int]:
        """
        Get final counting statistics.
        
        Returns:
            Dictionary with final counts
        """
        total_by_type = {}
        for vehicle_type, counts in self.count_by_type.items():
            total_count = counts["in"] + counts["out"]
            if total_count > 0:
                total_by_type[vehicle_type.value] = {
                    "total": total_count,
                    "in": counts["in"],
                    "out": counts["out"]
                }
        
        return {
            "total_vehicles": self.total_vehicles_in + self.total_vehicles_out,
            "vehicles_in": self.total_vehicles_in,
            "vehicles_out": self.total_vehicles_out,
            "by_type": total_by_type
        }