import cv2
import numpy as np
from collections import deque
import math


class PlayerDetector:
    def __init__(self):
        # Initialize YOLO or use OpenCV's DNN module for person detection
        self.net = None
        self.output_layers = None
        self._initialize_detector()
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Player tracking
        self.trackers = []
        self.next_player_id = 0
        self.max_trackers = 4  # Typically 2 players in tennis
        
        # Player detection parameters
        self.min_person_area = 1000
        self.max_person_area = 50000
        self.min_aspect_ratio = 1.2  # Height/Width ratio for standing person
        self.max_aspect_ratio = 4.0
        
    def _initialize_detector(self):
        # Try to load a pre-trained model for person detection
        # This is a placeholder - in practice, you'd load actual model weights
        try:
            # Example: Load YOLO weights (you need to download these)
            # self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
            # self.output_layers = self.net.getUnconnectedOutLayersNames()
            pass
        except:
            # Fall back to background subtraction method
            self.net = None
    
    def detect_players(self, frame):
        players = []
        
        if self.net is not None:
            # Use deep learning model for detection
            players = self._detect_with_dnn(frame)
        else:
            # Use traditional computer vision methods
            players = self._detect_with_background_subtraction(frame)
        
        # Update trackers with detected players
        self._update_trackers(frame, players)
        
        # Return tracked players with IDs
        return self._get_tracked_players()
    
    def _detect_with_dnn(self, frame):
        # Placeholder for DNN-based detection
        # In practice, you would:
        # 1. Preprocess frame for the model
        # 2. Run inference
        # 3. Post-process results to get bounding boxes
        # 4. Filter for person class
        
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class (class_id = 0 in COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        players = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                players.append({
                    'bbox': [x, y, w, h],
                    'center': [x + w//2, y + h//2],
                    'confidence': confidences[i],
                    'area': w * h
                })
        
        return players
    
    def _detect_with_background_subtraction(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        players = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_person_area < area < self.max_person_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Filter based on aspect ratio (standing person should be taller than wide)
                if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                    # Additional validation: check if the shape looks person-like
                    confidence = self._calculate_person_confidence(contour, frame[y:y+h, x:x+w])
                    
                    if confidence > 0.3:
                        players.append({
                            'bbox': [x, y, w, h],
                            'center': [x + w//2, y + h//2],
                            'confidence': confidence,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        return players
    
    def _calculate_person_confidence(self, contour, roi):
        # Calculate various features to determine if this looks like a person
        confidence = 0.0
        
        # Feature 1: Convexity (persons are roughly convex)
        hull = cv2.convexHull(contour)
        convexity = cv2.contourArea(contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        confidence += convexity * 0.3
        
        # Feature 2: Solidity (filled vs outline)
        solidity = cv2.contourArea(contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        confidence += solidity * 0.2
        
        # Feature 3: Color analysis (skin tone detection)
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Simple skin color range
            skin_lower = np.array([0, 20, 70])
            skin_upper = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv_roi, skin_lower, skin_upper)
            skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size
            confidence += min(skin_ratio * 2, 0.3)  # Cap at 0.3
        
        # Feature 4: Edge density (persons have distinctive edges)
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            confidence += min(edge_density * 2, 0.2)  # Cap at 0.2
        
        return min(confidence, 1.0)
    
    def _update_trackers(self, frame, detections):
        # Remove failed trackers
        self.trackers = [tracker for tracker in self.trackers if tracker['active']]
        
        # Match detections to existing trackers
        matched_trackers = set()
        unmatched_detections = []
        
        for detection in detections:
            best_match = None
            best_distance = float('inf')
            
            for i, tracker in enumerate(self.trackers):
                if i in matched_trackers:
                    continue
                
                # Calculate distance between detection and tracker prediction
                pred_center = tracker['predictor'].predict()
                det_center = detection['center']
                distance = math.sqrt((pred_center[0] - det_center[0])**2 + 
                                   (pred_center[1] - det_center[1])**2)
                
                if distance < best_distance and distance < 100:  # Max distance threshold
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update existing tracker
                self.trackers[best_match]['predictor'].update(detection['center'])
                self.trackers[best_match]['bbox'] = detection['bbox']
                self.trackers[best_match]['confidence'] = detection['confidence']
                self.trackers[best_match]['last_seen'] = 0
                matched_trackers.add(best_match)
            else:
                unmatched_detections.append(detection)
        
        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            if len(self.trackers) < self.max_trackers:
                tracker = {
                    'id': self.next_player_id,
                    'predictor': PlayerPositionPredictor(detection['center']),
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'active': True,
                    'last_seen': 0
                }
                self.trackers.append(tracker)
                self.next_player_id += 1
        
        # Update unmatched trackers (predict position, increase last_seen)
        for i, tracker in enumerate(self.trackers):
            if i not in matched_trackers:
                tracker['last_seen'] += 1
                if tracker['last_seen'] > 30:  # Deactivate after 30 frames
                    tracker['active'] = False
                else:
                    # Predict next position
                    predicted_pos = tracker['predictor'].predict()
                    # Update bbox based on prediction (keep same size)
                    bbox = tracker['bbox']
                    new_x = predicted_pos[0] - bbox[2] // 2
                    new_y = predicted_pos[1] - bbox[3] // 2
                    tracker['bbox'] = [new_x, new_y, bbox[2], bbox[3]]
                    tracker['confidence'] *= 0.9  # Reduce confidence
    
    def _get_tracked_players(self):
        players = []
        for tracker in self.trackers:
            if tracker['active'] and tracker['confidence'] > 0.1:
                bbox = tracker['bbox']
                players.append({
                    'id': tracker['id'],
                    'bbox': bbox,
                    'center': [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2],
                    'confidence': tracker['confidence'],
                    'predicted': tracker['last_seen'] > 0
                })
        return players
    
    def get_player_statistics(self):
        stats = {
            'total_players_detected': len(self.trackers),
            'active_players': len([t for t in self.trackers if t['active']]),
            'player_details': []
        }
        
        for tracker in self.trackers:
            if tracker['active']:
                positions = tracker['predictor'].get_position_history()
                if len(positions) > 1:
                    # Calculate movement statistics
                    distances = []
                    for i in range(1, len(positions)):
                        dist = math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                       (positions[i][1] - positions[i-1][1])**2)
                        distances.append(dist)
                    
                    stats['player_details'].append({
                        'id': tracker['id'],
                        'total_distance': sum(distances),
                        'avg_speed': np.mean(distances) if distances else 0,
                        'positions_tracked': len(positions),
                        'current_confidence': tracker['confidence']
                    })
        
        return stats


class PlayerPositionPredictor:
    def __init__(self, initial_position, history_length=10):
        self.position_history = deque([initial_position], maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length-1)
        
    def update(self, new_position):
        if self.position_history:
            last_pos = self.position_history[-1]
            velocity = (new_position[0] - last_pos[0], new_position[1] - last_pos[1])
            self.velocity_history.append(velocity)
        
        self.position_history.append(new_position)
    
    def predict(self):
        if not self.position_history:
            return (0, 0)
        
        current_pos = self.position_history[-1]
        
        if not self.velocity_history:
            return current_pos
        
        # Simple linear prediction based on average velocity
        avg_velocity = (
            sum(v[0] for v in self.velocity_history) / len(self.velocity_history),
            sum(v[1] for v in self.velocity_history) / len(self.velocity_history)
        )
        
        predicted_x = current_pos[0] + avg_velocity[0]
        predicted_y = current_pos[1] + avg_velocity[1]
        
        return (predicted_x, predicted_y)
    
    def get_position_history(self):
        return list(self.position_history)
    
    def get_velocity_history(self):
        return list(self.velocity_history)


class PlayerActivityAnalyzer:
    def __init__(self):
        self.player_activities = {}
        self.court_zones = self._define_court_zones()
    
    def _define_court_zones(self):
        # Define tennis court zones for activity analysis
        # This is a simplified version - in practice, you'd use court detection
        return {
            'baseline_left': {'x_range': (0, 0.3), 'y_range': (0, 1)},
            'baseline_right': {'x_range': (0.7, 1), 'y_range': (0, 1)},
            'net_area': {'x_range': (0.4, 0.6), 'y_range': (0, 1)},
            'service_box_left': {'x_range': (0.3, 0.5), 'y_range': (0, 0.5)},
            'service_box_right': {'x_range': (0.5, 0.7), 'y_range': (0.5, 1)}
        }
    
    def analyze_player_activity(self, player_data, frame_dimensions):
        player_id = player_data['id']
        position = player_data['center']
        
        # Normalize position to court coordinates (0-1 range)
        normalized_pos = (
            position[0] / frame_dimensions[1],  # width
            position[1] / frame_dimensions[0]   # height
        )
        
        if player_id not in self.player_activities:
            self.player_activities[player_id] = {
                'positions': [],
                'zone_time': {zone: 0 for zone in self.court_zones},
                'movement_patterns': [],
                'activity_level': 0
            }
        
        # Add position to history
        self.player_activities[player_id]['positions'].append(normalized_pos)
        
        # Determine current zone
        current_zone = self._get_zone(normalized_pos)
        if current_zone:
            self.player_activities[player_id]['zone_time'][current_zone] += 1
        
        # Calculate activity level (based on movement)
        if len(self.player_activities[player_id]['positions']) >= 2:
            last_pos = self.player_activities[player_id]['positions'][-2]
            movement = math.sqrt(
                (normalized_pos[0] - last_pos[0])**2 + 
                (normalized_pos[1] - last_pos[1])**2
            )
            self.player_activities[player_id]['activity_level'] = \
                0.9 * self.player_activities[player_id]['activity_level'] + 0.1 * movement
    
    def _get_zone(self, position):
        for zone_name, zone_def in self.court_zones.items():
            if (zone_def['x_range'][0] <= position[0] <= zone_def['x_range'][1] and
                zone_def['y_range'][0] <= position[1] <= zone_def['y_range'][1]):
                return zone_name
        return None
    
    def get_player_summary(self, player_id):
        if player_id not in self.player_activities:
            return None
        
        activity = self.player_activities[player_id]
        total_positions = len(activity['positions'])
        
        if total_positions == 0:
            return None
        
        return {
            'player_id': player_id,
            'total_tracked_frames': total_positions,
            'zone_distribution': {zone: time/total_positions 
                                for zone, time in activity['zone_time'].items()},
            'average_activity_level': activity['activity_level'],
            'court_coverage': len([zone for zone, time in activity['zone_time'].items() if time > 0])
        }