import cv2
import numpy as np
from collections import deque
import math


class AdvancedBallTracker:
    def __init__(self, history_length=10, min_radius=3, max_radius=25):
        self.history_length = history_length
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Ball position history for trajectory prediction
        self.position_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length-1)
        
        # Kalman filter for ball tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self._initialize_kalman()
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # HSV range for tennis ball (yellow-green)
        self.ball_color_lower = np.array([20, 100, 100])
        self.ball_color_upper = np.array([35, 255, 255])
        
        self.tracking_confidence = 0.0
        self.lost_frames = 0
        self.max_lost_frames = 10
        
    def _initialize_kalman(self):
        # State: [x, y, vx, vy]
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        
    def track_ball(self, frame):
        candidates = self._detect_ball_candidates(frame)
        
        if not candidates:
            return self._handle_lost_tracking()
        
        # Select best candidate based on multiple criteria
        best_candidate = self._select_best_candidate(candidates, frame)
        
        if best_candidate:
            self._update_tracking(best_candidate)
            self.lost_frames = 0
            return best_candidate
        else:
            return self._handle_lost_tracking()
    
    def _detect_ball_candidates(self, frame):
        candidates = []
        
        # Method 1: Color-based detection
        color_candidates = self._detect_by_color(frame)
        candidates.extend(color_candidates)
        
        # Method 2: Circular Hough transform
        circle_candidates = self._detect_by_circles(frame)
        candidates.extend(circle_candidates)
        
        # Method 3: Motion-based detection
        motion_candidates = self._detect_by_motion(frame)
        candidates.extend(motion_candidates)
        
        # Remove duplicates (candidates too close to each other)
        candidates = self._remove_duplicate_candidates(candidates)
        
        return candidates
    
    def _detect_by_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Filter by area
                # Get bounding circle
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                
                if self.min_radius <= radius <= self.max_radius:
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    if circularity > 0.5:  # Circular enough
                        candidates.append({
                            'position': (int(x), int(y)),
                            'radius': radius,
                            'confidence': circularity * 0.8,  # Color-based confidence
                            'method': 'color'
                        })
        
        return candidates
    
    def _detect_by_circles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        candidates = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Validate the circle by checking edge strength
                roi = gray[max(0, y-r):min(gray.shape[0], y+r+1),
                          max(0, x-r):min(gray.shape[1], x+r+1)]
                
                if roi.size > 0:
                    edge_strength = cv2.Laplacian(roi, cv2.CV_64F).var()
                    confidence = min(edge_strength / 1000, 1.0) * 0.7
                    
                    candidates.append({
                        'position': (x, y),
                        'radius': r,
                        'confidence': confidence,
                        'method': 'circle'
                    })
        
        return candidates
    
    def _detect_by_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 300:  # Small moving objects
                # Get bounding circle
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                
                if self.min_radius <= radius <= self.max_radius:
                    # Calculate aspect ratio of bounding rectangle
                    rect = cv2.boundingRect(contour)
                    aspect_ratio = rect[3] / rect[2] if rect[2] > 0 else 0
                    
                    if 0.7 <= aspect_ratio <= 1.3:  # Nearly square
                        confidence = (1.0 - abs(1.0 - aspect_ratio)) * 0.6
                        
                        candidates.append({
                            'position': (int(x), int(y)),
                            'radius': radius,
                            'confidence': confidence,
                            'method': 'motion'
                        })
        
        return candidates
    
    def _remove_duplicate_candidates(self, candidates, min_distance=20):
        if len(candidates) <= 1:
            return candidates
        
        unique_candidates = []
        
        for candidate in candidates:
            is_duplicate = False
            for unique in unique_candidates:
                distance = math.sqrt(
                    (candidate['position'][0] - unique['position'][0])**2 +
                    (candidate['position'][1] - unique['position'][1])**2
                )
                
                if distance < min_distance:
                    # Keep the one with higher confidence
                    if candidate['confidence'] > unique['confidence']:
                        unique_candidates.remove(unique)
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _select_best_candidate(self, candidates, frame):
        if not candidates:
            return None
        
        # If we have tracking history, use prediction
        if len(self.position_history) >= 2:
            predicted_pos = self._predict_next_position()
            
            # Score candidates based on distance from prediction
            for candidate in candidates:
                distance = math.sqrt(
                    (candidate['position'][0] - predicted_pos[0])**2 +
                    (candidate['position'][1] - predicted_pos[1])**2
                )
                
                # Boost confidence for candidates close to prediction
                distance_score = max(0, 1.0 - distance / 100.0)
                candidate['confidence'] += distance_score * 0.3
        
        # Select candidate with highest confidence
        best_candidate = max(candidates, key=lambda c: c['confidence'])
        
        # Only accept if confidence is above threshold
        if best_candidate['confidence'] > 0.3:
            return best_candidate
        
        return None
    
    def _predict_next_position(self):
        if len(self.position_history) < 2:
            return self.position_history[-1] if self.position_history else (0, 0)
        
        # Simple linear prediction based on last velocity
        last_pos = self.position_history[-1]
        if self.velocity_history:
            last_vel = self.velocity_history[-1]
            predicted_x = last_pos[0] + last_vel[0]
            predicted_y = last_pos[1] + last_vel[1]
            return (predicted_x, predicted_y)
        
        return last_pos
    
    def _update_tracking(self, candidate):
        position = candidate['position']
        
        # Update position history
        self.position_history.append(position)
        
        # Calculate velocity if we have enough history
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            velocity = (position[0] - prev_pos[0], position[1] - prev_pos[1])
            self.velocity_history.append(velocity)
        
        # Update Kalman filter
        measurement = np.array([[np.float32(position[0])], [np.float32(position[1])]])
        
        if self.lost_frames == 0:  # Continuous tracking
            self.kalman.correct(measurement)
        else:  # Re-initialize after lost tracking
            self.kalman.statePre = np.array([position[0], position[1], 0, 0], dtype=np.float32).reshape(-1, 1)
            self.kalman.statePost = self.kalman.statePre.copy()
        
        # Update confidence
        self.tracking_confidence = candidate['confidence']
    
    def _handle_lost_tracking(self):
        self.lost_frames += 1
        
        if self.lost_frames <= self.max_lost_frames and len(self.position_history) >= 2:
            # Try to predict position during lost tracking
            predicted_pos = self._predict_next_position()
            self.tracking_confidence *= 0.8  # Reduce confidence
            
            return {
                'position': predicted_pos,
                'radius': self.max_radius // 2,
                'confidence': self.tracking_confidence,
                'method': 'predicted'
            }
        
        # Lost tracking completely
        if self.lost_frames > self.max_lost_frames:
            self.tracking_confidence = 0.0
            self.position_history.clear()
            self.velocity_history.clear()
        
        return None
    
    def get_trajectory(self):
        return list(self.position_history)
    
    def get_velocity(self):
        if self.velocity_history:
            return self.velocity_history[-1]
        return (0, 0)
    
    def get_speed(self):
        velocity = self.get_velocity()
        return math.sqrt(velocity[0]**2 + velocity[1]**2)
    
    def is_tracking(self):
        return self.tracking_confidence > 0.3 and self.lost_frames < self.max_lost_frames
    
    def reset(self):
        self.position_history.clear()
        self.velocity_history.clear()
        self.tracking_confidence = 0.0
        self.lost_frames = 0
        self._initialize_kalman()


class BallTrajectoryAnalyzer:
    def __init__(self):
        self.trajectories = []
        self.current_trajectory = []
        self.min_trajectory_length = 5
    
    def add_detection(self, detection, frame_number, timestamp):
        if detection is None:
            self._finalize_current_trajectory()
            return
        
        self.current_trajectory.append({
            'frame': frame_number,
            'timestamp': timestamp,
            'position': detection['position'],
            'confidence': detection['confidence']
        })
    
    def _finalize_current_trajectory(self):
        if len(self.current_trajectory) >= self.min_trajectory_length:
            trajectory_analysis = self._analyze_trajectory(self.current_trajectory)
            self.trajectories.append({
                'points': self.current_trajectory.copy(),
                'analysis': trajectory_analysis
            })
        
        self.current_trajectory.clear()
    
    def _analyze_trajectory(self, trajectory):
        if len(trajectory) < 2:
            return {}
        
        positions = [point['position'] for point in trajectory]
        timestamps = [point['timestamp'] for point in trajectory]
        
        # Calculate speeds and accelerations
        speeds = []
        accelerations = []
        
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                speed = math.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
        
        for i in range(1, len(speeds)):
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                acceleration = (speeds[i] - speeds[i-1]) / dt
                accelerations.append(acceleration)
        
        return {
            'duration': timestamps[-1] - timestamps[0],
            'length': len(trajectory),
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0,
            'start_position': positions[0],
            'end_position': positions[-1],
            'total_distance': sum(math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                          (positions[i][1] - positions[i-1][1])**2) 
                                for i in range(1, len(positions)))
        }
    
    def get_all_trajectories(self):
        # Finalize current trajectory if it exists
        if self.current_trajectory:
            self._finalize_current_trajectory()
        
        return self.trajectories
    
    def get_statistics(self):
        if not self.trajectories:
            return {}
        
        all_speeds = []
        all_durations = []
        
        for trajectory in self.trajectories:
            analysis = trajectory['analysis']
            if 'avg_speed' in analysis:
                all_speeds.append(analysis['avg_speed'])
            if 'duration' in analysis:
                all_durations.append(analysis['duration'])
        
        return {
            'total_trajectories': len(self.trajectories),
            'avg_speed_overall': np.mean(all_speeds) if all_speeds else 0,
            'max_speed_overall': np.max(all_speeds) if all_speeds else 0,
            'avg_trajectory_duration': np.mean(all_durations) if all_durations else 0,
            'total_tracking_time': sum(all_durations) if all_durations else 0
        }