import cv2
import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime

from video_processor import VideoProcessor
from ball_tracker import AdvancedBallTracker, BallTrajectoryAnalyzer
from court_detector import TennisCourtDetector
from player_detector import PlayerDetector, PlayerActivityAnalyzer
from analysis_visualizer import TennisAnalysisVisualizer, create_analysis_video


class TennisVideoAnalyzer:
    def __init__(self, video_path, output_dir="analysis_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize video processor
        self.video_processor = VideoProcessor(video_path)
        video_info = self.video_processor.get_video_info()
        
        # Initialize analysis components
        self.ball_tracker = AdvancedBallTracker()
        self.ball_trajectory_analyzer = BallTrajectoryAnalyzer()
        self.court_detector = TennisCourtDetector()
        self.player_detector = PlayerDetector()
        self.player_activity_analyzer = PlayerActivityAnalyzer()
        self.visualizer = TennisAnalysisVisualizer(self.output_dir / "visualizations")
        
        # Analysis results structure
        self.analysis_results = {
            "video_info": video_info,
            "ball_tracking": [],
            "player_positions": [],
            "court_lines": None,
            "court_detection": None,
            "rallies": [],
            "statistics": {},
            "player_statistics": {},
            "ball_statistics": {}
        }
    
    def analyze_video(self, progress_callback=None):
        video_info = self.analysis_results["video_info"]
        print(f"Analyzing video: {self.video_path}")
        print(f"Duration: {video_info['duration']:.2f}s, FPS: {video_info['fps']:.2f}")
        
        # Extract frames for analysis
        frames, frame_indices = self.video_processor.extract_frames()
        total_frames = len(frames)
        
        # Detect court in the first frame
        if frames:
            court_result = self.court_detector.detect_court(frames[0])
            self.analysis_results["court_detection"] = court_result
            
        # Process each frame
        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            timestamp = frame_idx / video_info['fps']
            
            # Track ball
            ball_detection = self.ball_tracker.track_ball(frame)
            self.ball_trajectory_analyzer.add_detection(ball_detection, frame_idx, timestamp)
            
            if ball_detection:
                self.analysis_results["ball_tracking"].append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "position": ball_detection['position'],
                    "confidence": ball_detection['confidence'],
                    "method": ball_detection['method']
                })
            
            # Detect players
            players = self.player_detector.detect_players(frame)
            if players:
                self.analysis_results["player_positions"].append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "players": players
                })
                
                # Analyze player activity
                for player in players:
                    self.player_activity_analyzer.analyze_player_activity(
                        player, (video_info['height'], video_info['width'])
                    )
            
            # Progress reporting
            if i % 100 == 0 or i == total_frames - 1:
                progress = (i + 1) / total_frames * 100
                print(f"Processed {i + 1}/{total_frames} frames ({progress:.1f}%)")
                
                if progress_callback:
                    progress_callback(progress)
        
        # Finalize analysis
        self._finalize_analysis()
        self._save_results()
        self._generate_visualizations()
        
        print("Analysis complete!")
    
    def _finalize_analysis(self):
        # Get ball trajectory statistics
        ball_trajectories = self.ball_trajectory_analyzer.get_all_trajectories()
        ball_stats = self.ball_trajectory_analyzer.get_statistics()
        self.analysis_results["ball_statistics"] = ball_stats
        
        # Get player statistics
        player_stats = self.player_detector.get_player_statistics()
        self.analysis_results["player_statistics"] = player_stats
        
        # Calculate overall statistics
        self.analysis_results["statistics"] = {
            "total_ball_detections": len(self.analysis_results["ball_tracking"]),
            "total_player_detections": len(self.analysis_results["player_positions"]),
            "ball_tracking_confidence": np.mean([entry["confidence"] for entry in self.analysis_results["ball_tracking"]]) if self.analysis_results["ball_tracking"] else 0,
            "court_detected": self.analysis_results["court_detection"] is not None and self.analysis_results["court_detection"].get("court_detected", False),
            "analysis_duration": self.analysis_results["video_info"]["duration"],
            "trajectories_found": len(ball_trajectories)
        }
        
        # Get player activity summaries
        player_summaries = {}
        for player_id in range(self.player_detector.next_player_id):
            summary = self.player_activity_analyzer.get_player_summary(player_id)
            if summary:
                player_summaries[player_id] = summary
        
        self.analysis_results["player_activity"] = player_summaries
    
    def _save_results(self):
        # Save analysis results in multiple formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save CSV data
        self.visualizer.save_analysis_data(self.analysis_results, format='csv')
        
        # Create text report
        report_file = self.visualizer.create_video_summary_report(self.analysis_results)
        
        print(f"Results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")
    
    def _generate_visualizations(self):
        print("Generating visualizations...")
        
        # Ball trajectory plot
        if self.analysis_results["ball_tracking"]:
            self.visualizer.create_ball_trajectory_plot(
                self.analysis_results["ball_tracking"],
                "Ball Tracking Analysis"
            )
        
        # Player heatmap
        if self.analysis_results["player_positions"]:
            player_data = self._extract_player_positions()
            video_info = self.analysis_results["video_info"]
            self.visualizer.create_player_heatmap(
                player_data,
                (video_info["height"], video_info["width"]),
                "Player Movement Analysis"
            )
        
        # Court analysis
        self.visualizer.create_court_analysis_plot(
            self.analysis_results["court_detection"],
            self.analysis_results["ball_tracking"],
            self._extract_player_positions() if self.analysis_results["player_positions"] else None
        )
        
        # Statistics dashboard
        self.visualizer.create_statistics_dashboard(self.analysis_results)
        
        print("Visualizations generated in:", self.output_dir / "visualizations")
    
    def _extract_player_positions(self):
        player_data = {}
        for entry in self.analysis_results["player_positions"]:
            for player in entry["players"]:
                player_id = player.get("id", "unknown")
                if player_id not in player_data:
                    player_data[player_id] = []
                player_data[player_id].append(player["center"])
        return player_data
    
    def create_annotated_video(self, output_path=None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"annotated_video_{timestamp}.mp4"
        
        create_analysis_video(
            str(self.video_path),
            self.analysis_results,
            str(output_path)
        )
        
        print(f"Annotated video saved to: {output_path}")
        return output_path




def main():
    parser = argparse.ArgumentParser(description="Advanced Tennis Video Analyzer")
    parser.add_argument("video_path", help="Path to the tennis video file")
    parser.add_argument("--output", "-o", default="analysis_output", 
                       help="Output directory for analysis results")
    parser.add_argument("--create-video", action="store_true",
                       help="Create annotated video with analysis overlays")
    parser.add_argument("--extract-keyframes", action="store_true",
                       help="Extract key frames from the video")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    try:
        # Initialize analyzer
        analyzer = TennisVideoAnalyzer(args.video_path, args.output)
        
        # Run analysis
        print("Starting tennis video analysis...")
        analyzer.analyze_video()
        
        # Create annotated video if requested
        if args.create_video:
            print("Creating annotated video...")
            analyzer.create_annotated_video()
        
        # Extract key frames if requested
        if args.extract_keyframes:
            print("Extracting key frames...")
            from video_processor import extract_key_frames
            keyframes_dir = Path(args.output) / "keyframes"
            extract_key_frames(args.video_path, keyframes_dir)
        
        print(f"\nAnalysis complete! Results saved to: {args.output}")
        
        # Print summary statistics
        stats = analyzer.analysis_results.get("statistics", {})
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Ball detections: {stats.get('total_ball_detections', 0)}")
        print(f"Player detections: {stats.get('total_player_detections', 0)}")
        print(f"Court detected: {'Yes' if stats.get('court_detected', False) else 'No'}")
        print(f"Ball tracking confidence: {stats.get('ball_tracking_confidence', 0):.2f}")
        print(f"Trajectories found: {stats.get('trajectories_found', 0)}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()