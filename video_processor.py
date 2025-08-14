import cv2
import numpy as np
from pathlib import Path
import json


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
    def get_video_info(self):
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration
        }
    
    def extract_frames(self, start_time=0, end_time=None, step=1):
        if end_time is None:
            end_time = self.duration
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        frames = []
        frame_indices = []
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if (current_frame - start_frame) % step == 0:
                frames.append(frame.copy())
                frame_indices.append(current_frame)
            
            current_frame += 1
        
        return frames, frame_indices
    
    def extract_frame_at_time(self, timestamp):
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def process_video_in_chunks(self, chunk_size=100, callback=None):
        results = []
        
        for start_frame in range(0, self.frame_count, chunk_size):
            end_frame = min(start_frame + chunk_size, self.frame_count)
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            chunk_frames = []
            for frame_idx in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                chunk_frames.append((frame_idx, frame))
            
            if callback:
                chunk_result = callback(chunk_frames)
                results.append(chunk_result)
        
        return results
    
    def save_frame(self, frame, output_path, timestamp=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if timestamp is not None:
            # Add timestamp overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Time: {timestamp:.2f}s"
            cv2.putText(frame, text, (10, 30), font, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(str(output_path), frame)
    
    def create_highlight_video(self, highlights, output_path, padding=1.0):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        for start_time, end_time in highlights:
            # Add padding
            start_time = max(0, start_time - padding)
            end_time = min(self.duration, end_time + padding)
            
            start_frame = int(start_time * self.fps)
            end_frame = int(end_time * self.fps)
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                out.write(frame)
        
        out.release()
    
    def apply_filters(self, frame, filters=None):
        if filters is None:
            return frame
        
        processed_frame = frame.copy()
        
        for filter_name, params in filters.items():
            if filter_name == "blur":
                ksize = params.get("ksize", 5)
                processed_frame = cv2.GaussianBlur(processed_frame, (ksize, ksize), 0)
            
            elif filter_name == "brightness":
                alpha = params.get("alpha", 1.0)  # Contrast
                beta = params.get("beta", 0)      # Brightness
                processed_frame = cv2.convertScaleAbs(processed_frame, alpha=alpha, beta=beta)
            
            elif filter_name == "grayscale":
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            
            elif filter_name == "edge_detection":
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, params.get("low", 50), params.get("high", 150))
                processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return processed_frame
    
    def release(self):
        if self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FrameAnalyzer:
    def __init__(self):
        self.frame_cache = {}
    
    def analyze_frame(self, frame, frame_idx=None):
        analysis = {
            "frame_index": frame_idx,
            "shape": frame.shape,
            "mean_intensity": np.mean(frame),
            "motion_vectors": self._calculate_motion(frame),
            "dominant_colors": self._get_dominant_colors(frame),
            "sharpness": self._calculate_sharpness(frame)
        }
        
        return analysis
    
    def _calculate_motion(self, frame):
        # Simple motion detection using frame differencing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if hasattr(self, 'prev_frame'):
            diff = cv2.absdiff(self.prev_frame, gray)
            motion_magnitude = np.mean(diff)
            self.prev_frame = gray.copy()
            return motion_magnitude
        else:
            self.prev_frame = gray.copy()
            return 0
    
    def _get_dominant_colors(self, frame, k=3):
        # Reshape frame to be a list of pixels
        pixels = frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        return centers.tolist()
    
    def _calculate_sharpness(self, frame):
        # Calculate sharpness using Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()


def extract_key_frames(video_path, output_dir, threshold=0.3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with VideoProcessor(video_path) as processor:
        analyzer = FrameAnalyzer()
        key_frames = []
        
        def process_chunk(chunk_frames):
            chunk_key_frames = []
            prev_analysis = None
            
            for frame_idx, frame in chunk_frames:
                analysis = analyzer.analyze_frame(frame, frame_idx)
                
                if prev_analysis is not None:
                    # Calculate difference between consecutive frames
                    motion_diff = abs(analysis["motion_vectors"] - prev_analysis["motion_vectors"])
                    intensity_diff = abs(analysis["mean_intensity"] - prev_analysis["mean_intensity"])
                    
                    # Determine if this is a key frame
                    if motion_diff > threshold or intensity_diff > 20:
                        chunk_key_frames.append((frame_idx, frame, analysis))
                
                prev_analysis = analysis
            
            return chunk_key_frames
        
        chunk_results = processor.process_video_in_chunks(callback=process_chunk)
        
        # Flatten results and save key frames
        frame_count = 0
        for chunk_result in chunk_results:
            for frame_idx, frame, analysis in chunk_result:
                timestamp = frame_idx / processor.fps
                output_path = output_dir / f"keyframe_{frame_count:04d}_{frame_idx:06d}.jpg"
                processor.save_frame(frame, output_path, timestamp)
                
                key_frames.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "file_path": str(output_path),
                    "analysis": analysis
                })
                frame_count += 1
        
        # Save key frame metadata
        metadata_path = output_dir / "keyframes_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(key_frames, f, indent=2, default=str)
        
        return key_frames