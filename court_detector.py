import cv2
import numpy as np
import math
from scipy import optimize


class TennisCourtDetector:
    def __init__(self):
        # Standard tennis court dimensions (in meters)
        self.court_width = 23.77  # 78 feet
        self.court_length = 10.97  # 36 feet for doubles
        
        # Court line detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10
        
        # Court template points (normalized coordinates)
        self.template_points = self._get_court_template()
        
        # Detected court lines and corners
        self.detected_lines = []
        self.court_corners = None
        self.homography_matrix = None
        
    def _get_court_template(self):
        # Define standard tennis court template in normalized coordinates (0-1)
        return {
            'outer_boundary': [
                (0, 0), (1, 0), (1, 1), (0, 1)  # Outer court rectangle
            ],
            'service_lines': [
                (0, 0.186), (1, 0.186),  # Service line (far)
                (0, 0.814), (1, 0.814)   # Service line (near)
            ],
            'center_line': [
                (0.5, 0.186), (0.5, 0.814)  # Center service line
            ],
            'baseline': [
                (0, 0), (1, 0),    # Far baseline
                (0, 1), (1, 1)     # Near baseline
            ],
            'sidelines': [
                (0, 0), (0, 1),    # Left sideline
                (1, 0), (1, 1)     # Right sideline
            ],
            'net_line': [
                (0, 0.5), (1, 0.5)  # Net position
            ]
        }
    
    def detect_court(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high, apertureSize=3)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return None
        
        # Filter and classify lines
        self.detected_lines = self._filter_and_classify_lines(lines, frame.shape)
        
        # Find court corners
        self.court_corners = self._find_court_corners(self.detected_lines, frame.shape)
        
        if self.court_corners is not None:
            # Calculate homography matrix
            self.homography_matrix = self._calculate_homography(self.court_corners)
            
            return {
                'corners': self.court_corners,
                'lines': self.detected_lines,
                'homography': self.homography_matrix,
                'court_detected': True
            }
        
        return {
            'corners': None,
            'lines': self.detected_lines,
            'homography': None,
            'court_detected': False
        }
    
    def _filter_and_classify_lines(self, lines, frame_shape):
        height, width = frame_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            # Calculate line length
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filter short lines
            if length < 30:
                continue
            
            # Classify as horizontal or vertical based on angle
            if abs(angle) < 20 or abs(angle) > 160:  # Horizontal-ish
                horizontal_lines.append({
                    'points': [(x1, y1), (x2, y2)],
                    'angle': angle,
                    'length': length,
                    'type': 'horizontal'
                })
            elif 70 < abs(angle) < 110:  # Vertical-ish
                vertical_lines.append({
                    'points': [(x1, y1), (x2, y2)],
                    'angle': angle,
                    'length': length,
                    'type': 'vertical'
                })
        
        # Merge similar lines
        horizontal_lines = self._merge_similar_lines(horizontal_lines, width)
        vertical_lines = self._merge_similar_lines(vertical_lines, height)
        
        return {
            'horizontal': horizontal_lines,
            'vertical': vertical_lines,
            'all': horizontal_lines + vertical_lines
        }
    
    def _merge_similar_lines(self, lines, reference_dimension):
        if not lines:
            return []
        
        merged_lines = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            # Start with current line
            merged_line = line1.copy()
            used.add(i)
            
            # Find similar lines to merge
            for j, line2 in enumerate(lines):
                if j in used or j <= i:
                    continue
                
                if self._are_lines_similar(line1, line2, reference_dimension):
                    # Merge lines by extending endpoints
                    merged_line = self._merge_two_lines(merged_line, line2)
                    used.add(j)
            
            merged_lines.append(merged_line)
        
        return merged_lines
    
    def _are_lines_similar(self, line1, line2, reference_dimension):
        # Check if two lines are similar (parallel and close)
        angle_diff = abs(line1['angle'] - line2['angle'])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Lines should be roughly parallel
        if angle_diff > 15:
            return False
        
        # Lines should be close to each other
        if line1['type'] == 'horizontal':
            # Compare y-coordinates
            y1_avg = (line1['points'][0][1] + line1['points'][1][1]) / 2
            y2_avg = (line2['points'][0][1] + line2['points'][1][1]) / 2
            distance = abs(y1_avg - y2_avg)
        else:
            # Compare x-coordinates
            x1_avg = (line1['points'][0][0] + line1['points'][1][0]) / 2
            x2_avg = (line2['points'][0][0] + line2['points'][1][0]) / 2
            distance = abs(x1_avg - x2_avg)
        
        return distance < reference_dimension * 0.05  # 5% of reference dimension
    
    def _merge_two_lines(self, line1, line2):
        # Merge two similar lines by extending endpoints
        all_points = line1['points'] + line2['points']
        
        if line1['type'] == 'horizontal':
            # Sort by x-coordinate
            all_points.sort(key=lambda p: p[0])
            merged_points = [all_points[0], all_points[-1]]
        else:
            # Sort by y-coordinate
            all_points.sort(key=lambda p: p[1])
            merged_points = [all_points[0], all_points[-1]]
        
        # Calculate new length
        new_length = math.sqrt(
            (merged_points[1][0] - merged_points[0][0])**2 +
            (merged_points[1][1] - merged_points[0][1])**2
        )
        
        return {
            'points': merged_points,
            'angle': (line1['angle'] + line2['angle']) / 2,
            'length': new_length,
            'type': line1['type']
        }
    
    def _find_court_corners(self, lines, frame_shape):
        if not lines['horizontal'] or not lines['vertical']:
            return None
        
        height, width = frame_shape[:2]
        
        # Find intersections between horizontal and vertical lines
        intersections = []
        
        for h_line in lines['horizontal']:
            for v_line in lines['vertical']:
                intersection = self._line_intersection(h_line, v_line)
                if intersection and self._is_point_in_frame(intersection, width, height):
                    intersections.append(intersection)
        
        if len(intersections) < 4:
            return None
        
        # Find the four corners that form the largest rectangle
        corners = self._find_court_rectangle(intersections, width, height)
        
        return corners
    
    def _line_intersection(self, line1, line2):
        # Find intersection point of two lines
        (x1, y1), (x2, y2) = line1['points']
        (x3, y3), (x4, y4) = line2['points']
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        
        return (intersection_x, intersection_y)
    
    def _is_point_in_frame(self, point, width, height, margin=50):
        x, y = point
        return margin <= x <= width - margin and margin <= y <= height - margin
    
    def _find_court_rectangle(self, intersections, width, height):
        # Find four points that form the best rectangle representing the court
        if len(intersections) < 4:
            return None
        
        # Convert to numpy array for easier processing
        points = np.array(intersections)
        
        # Find convex hull to get outer boundary points
        hull = cv2.convexHull(points.astype(np.float32))
        hull_points = hull.reshape(-1, 2)
        
        if len(hull_points) < 4:
            return None
        
        # If we have exactly 4 points, use them
        if len(hull_points) == 4:
            return self._order_court_corners(hull_points)
        
        # If we have more than 4 points, find the best 4 that form a rectangle
        best_corners = self._find_best_rectangle(hull_points, width, height)
        
        return self._order_court_corners(best_corners) if best_corners is not None else None
    
    def _find_best_rectangle(self, points, width, height):
        # Find the best 4 points that form a rectangle
        # This is a simplified approach - in practice, you might use more sophisticated methods
        
        # Try all combinations of 4 points
        from itertools import combinations
        
        best_score = -1
        best_corners = None
        
        for four_points in combinations(points, 4):
            corners = np.array(four_points)
            score = self._score_rectangle(corners, width, height)
            
            if score > best_score:
                best_score = score
                best_corners = corners
        
        return best_corners if best_score > 0.5 else None
    
    def _score_rectangle(self, corners, width, height):
        # Score how well four points form a rectangle
        if len(corners) != 4:
            return 0
        
        # Calculate side lengths
        ordered_corners = self._order_court_corners(corners)
        if ordered_corners is None:
            return 0
        
        # Calculate distances between adjacent corners
        distances = []
        for i in range(4):
            j = (i + 1) % 4
            dist = np.linalg.norm(ordered_corners[i] - ordered_corners[j])
            distances.append(dist)
        
        # Check if opposite sides are similar in length
        side1_ratio = min(distances[0], distances[2]) / max(distances[0], distances[2])
        side2_ratio = min(distances[1], distances[3]) / max(distances[1], distances[3])
        
        # Check aspect ratio (tennis court is roughly 2.36:1)
        width_court = max(distances[0], distances[2])
        height_court = max(distances[1], distances[3])
        aspect_ratio = width_court / height_court if height_court > 0 else 0
        
        ideal_aspect_ratio = self.court_width / self.court_length  # ~2.17
        aspect_score = 1 - abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio
        
        # Check if corners are reasonably spaced
        area = cv2.contourArea(ordered_corners.astype(np.float32))
        frame_area = width * height
        area_ratio = area / frame_area
        
        area_score = 1 if 0.1 <= area_ratio <= 0.8 else 0
        
        # Combined score
        score = (side1_ratio + side2_ratio + aspect_score + area_score) / 4
        
        return max(0, min(1, score))
    
    def _order_court_corners(self, corners):
        # Order corners as: top-left, top-right, bottom-right, bottom-left
        if len(corners) != 4:
            return None
        
        corners = np.array(corners)
        
        # Find centroid
        centroid = np.mean(corners, axis=0)
        
        # Sort by angle from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder to start from top-left (smallest x+y sum)
        sums = corners[:, 0] + corners[:, 1]
        top_left_idx = np.argmin(sums)
        
        # Find the correct starting index in sorted order
        start_idx = np.where(sorted_indices == top_left_idx)[0][0]
        
        # Reorder starting from top-left and going clockwise
        ordered_indices = np.roll(sorted_indices, -start_idx)
        
        return corners[ordered_indices]
    
    def _calculate_homography(self, court_corners):
        # Calculate homography matrix to transform court to standard view
        if court_corners is None or len(court_corners) != 4:
            return None
        
        # Define destination points (standard court view)
        dst_points = np.array([
            [0, 0],                    # top-left
            [self.court_width, 0],     # top-right
            [self.court_width, self.court_length],  # bottom-right
            [0, self.court_length]     # bottom-left
        ], dtype=np.float32)
        
        src_points = court_corners.astype(np.float32)
        
        # Calculate homography
        homography, _ = cv2.findHomography(src_points, dst_points)
        
        return homography
    
    def transform_point_to_court(self, point):
        # Transform a point from image coordinates to court coordinates
        if self.homography_matrix is None:
            return None
        
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.homography_matrix)
        
        return tuple(transformed[0][0])
    
    def transform_point_to_image(self, court_point):
        # Transform a point from court coordinates to image coordinates
        if self.homography_matrix is None:
            return None
        
        # Use inverse homography
        inv_homography = np.linalg.inv(self.homography_matrix)
        point_array = np.array([[court_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, inv_homography)
        
        return tuple(transformed[0][0])
    
    def draw_court_overlay(self, frame, color=(0, 255, 0), thickness=2):
        # Draw detected court lines and corners on the frame
        if self.court_corners is None:
            return frame
        
        overlay = frame.copy()
        
        # Draw court corners
        for corner in self.court_corners:
            cv2.circle(overlay, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
        
        # Draw court boundary
        corners_int = self.court_corners.astype(int)
        cv2.polylines(overlay, [corners_int], True, color, thickness)
        
        # Draw detected lines
        for line_type in ['horizontal', 'vertical']:
            for line in self.detected_lines.get(line_type, []):
                pt1 = tuple(map(int, line['points'][0]))
                pt2 = tuple(map(int, line['points'][1]))
                cv2.line(overlay, pt1, pt2, (255, 0, 0), 1)
        
        return overlay
    
    def get_court_info(self):
        # Return information about the detected court
        return {
            'corners': self.court_corners.tolist() if self.court_corners is not None else None,
            'homography_available': self.homography_matrix is not None,
            'lines_detected': {
                'horizontal': len(self.detected_lines.get('horizontal', [])),
                'vertical': len(self.detected_lines.get('vertical', []))
            },
            'court_template': self.template_points
        }