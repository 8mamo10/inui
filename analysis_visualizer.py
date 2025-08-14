import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import json
from pathlib import Path
import seaborn as sns
from datetime import datetime
import pandas as pd


class TennisAnalysisVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Tennis court dimensions for visualization
        self.court_width = 23.77  # meters
        self.court_length = 10.97  # meters
        
    def create_ball_trajectory_plot(self, ball_data, title="Ball Trajectory Analysis"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        if not ball_data:
            ax1.text(0.5, 0.5, 'No ball tracking data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No ball tracking data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            plt.tight_layout()
            return fig
        
        # Extract positions and timestamps
        positions = np.array([entry['position'] for entry in ball_data])
        timestamps = np.array([entry['timestamp'] for entry in ball_data])
        
        # Plot 1: Ball trajectory in 2D space
        ax1.scatter(positions[:, 0], positions[:, 1], 
                   c=timestamps, cmap='viridis', s=20, alpha=0.7)
        ax1.plot(positions[:, 0], positions[:, 1], alpha=0.3, linewidth=1)
        
        # Add colorbar for time
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Time (seconds)')
        
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title('Ball Trajectory Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Invert Y axis to match image coordinates
        
        # Plot 2: Ball speed over time
        if len(positions) > 1:
            speeds = []
            speed_times = []
            
            for i in range(1, len(positions)):
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    speeds.append(speed)
                    speed_times.append(timestamps[i])
            
            if speeds:
                ax2.plot(speed_times, speeds, linewidth=2, marker='o', markersize=3)
                ax2.fill_between(speed_times, speeds, alpha=0.3)
                
                # Add statistics
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                ax2.axhline(y=avg_speed, color='red', linestyle='--', 
                           label=f'Average Speed: {avg_speed:.1f} px/s')
                ax2.axhline(y=max_speed, color='orange', linestyle='--', 
                           label=f'Max Speed: {max_speed:.1f} px/s')
                
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Speed (pixels/second)')
                ax2.set_title('Ball Speed Over Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"ball_trajectory_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_player_heatmap(self, player_data, frame_dimensions, title="Player Movement Heatmap"):
        fig, axes = plt.subplots(1, len(player_data), figsize=(6*len(player_data), 6))
        
        if len(player_data) == 1:
            axes = [axes]
        
        for idx, (player_id, positions) in enumerate(player_data.items()):
            if not positions:
                axes[idx].text(0.5, 0.5, f'No data for Player {player_id}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            # Create 2D histogram for heatmap
            positions_array = np.array(positions)
            x_positions = positions_array[:, 0]
            y_positions = positions_array[:, 1]
            
            # Create heatmap
            heatmap, xedges, yedges = np.histogram2d(
                x_positions, y_positions, 
                bins=[50, 50], 
                range=[[0, frame_dimensions[1]], [0, frame_dimensions[0]]]
            )
            
            # Plot heatmap
            im = axes[idx].imshow(heatmap.T, origin='lower', cmap='hot', 
                                 extent=[0, frame_dimensions[1], 0, frame_dimensions[0]],
                                 aspect='auto', alpha=0.8)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], label='Time Spent')
            
            axes[idx].set_xlabel('X Position (pixels)')
            axes[idx].set_ylabel('Y Position (pixels)')
            axes[idx].set_title(f'Player {player_id} Movement Heatmap')
            axes[idx].invert_yaxis()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"player_heatmap_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_court_analysis_plot(self, court_data, ball_data=None, player_data=None):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Draw tennis court outline
        self._draw_tennis_court(ax)
        
        if court_data and court_data.get('homography_available'):
            # If we have court detection, transform coordinates
            pass  # Implementation would depend on having the court detector instance
        
        # Plot ball trajectory on court if available
        if ball_data:
            # For now, plot raw positions (would need transformation in real implementation)
            positions = np.array([entry['position'] for entry in ball_data])
            # Normalize positions to court coordinates (simplified)
            normalized_x = (positions[:, 0] / 1000) * self.court_width  # Assuming 1000px width
            normalized_y = (positions[:, 1] / 1000) * self.court_length
            
            ax.scatter(normalized_x, normalized_y, c='yellow', s=30, 
                      alpha=0.7, label='Ball Positions', edgecolors='black', linewidth=0.5)
            ax.plot(normalized_x, normalized_y, color='yellow', alpha=0.3, linewidth=1)
        
        # Plot player positions if available
        if player_data:
            colors = ['red', 'blue', 'green', 'purple']
            for idx, (player_id, positions) in enumerate(player_data.items()):
                if positions:
                    positions_array = np.array(positions)
                    # Normalize positions (simplified)
                    norm_x = (positions_array[:, 0] / 1000) * self.court_width
                    norm_y = (positions_array[:, 1] / 1000) * self.court_length
                    
                    color = colors[idx % len(colors)]
                    ax.scatter(norm_x, norm_y, c=color, s=20, alpha=0.6, 
                             label=f'Player {player_id}')
        
        ax.set_xlabel('Court Width (meters)')
        ax.set_ylabel('Court Length (meters)')
        ax.set_title('Tennis Court Analysis View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"court_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _draw_tennis_court(self, ax):
        # Draw standard tennis court layout
        # Outer boundary
        court_rect = patches.Rectangle((0, 0), self.court_width, self.court_length, 
                                     linewidth=2, edgecolor='white', facecolor='darkgreen', alpha=0.8)
        ax.add_patch(court_rect)
        
        # Service boxes
        service_line_y1 = self.court_length * 0.186  # 6.4m from baseline
        service_line_y2 = self.court_length * 0.814  # 6.4m from other baseline
        
        # Service lines
        ax.plot([0, self.court_width], [service_line_y1, service_line_y1], 'white', linewidth=1.5)
        ax.plot([0, self.court_width], [service_line_y2, service_line_y2], 'white', linewidth=1.5)
        
        # Center line
        center_x = self.court_width / 2
        ax.plot([center_x, center_x], [service_line_y1, service_line_y2], 'white', linewidth=1.5)
        
        # Net line
        net_y = self.court_length / 2
        ax.plot([0, self.court_width], [net_y, net_y], 'white', linewidth=3)
        
        # Sidelines and baselines
        ax.plot([0, 0], [0, self.court_length], 'white', linewidth=2)  # Left sideline
        ax.plot([self.court_width, self.court_width], [0, self.court_length], 'white', linewidth=2)  # Right
        ax.plot([0, self.court_width], [0, 0], 'white', linewidth=2)  # Bottom baseline
        ax.plot([0, self.court_width], [self.court_length, self.court_length], 'white', linewidth=2)  # Top
    
    def create_statistics_dashboard(self, analysis_results):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract statistics from analysis results
        stats = analysis_results.get('statistics', {})
        ball_data = analysis_results.get('ball_tracking', [])
        player_data = analysis_results.get('player_positions', [])
        
        # 1. Ball detection rate over time
        ax1 = fig.add_subplot(gs[0, 0])
        if ball_data:
            timestamps = [entry['timestamp'] for entry in ball_data]
            detection_rate = self._calculate_detection_rate(timestamps, analysis_results['video_info']['duration'])
            ax1.plot(detection_rate['time_bins'], detection_rate['rate'])
            ax1.set_title('Ball Detection Rate')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Detection Rate (%)')
        
        # 2. Speed distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if ball_data and len(ball_data) > 1:
            speeds = self._calculate_speeds(ball_data)
            ax2.hist(speeds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Ball Speed Distribution')
            ax2.set_xlabel('Speed (px/s)')
            ax2.set_ylabel('Frequency')
        
        # 3. Player activity levels
        ax3 = fig.add_subplot(gs[0, 2])
        if player_data:
            player_activity = self._calculate_player_activity(player_data)
            players = list(player_activity.keys())
            activities = list(player_activity.values())
            ax3.bar(players, activities, color='lightcoral')
            ax3.set_title('Player Activity Levels')
            ax3.set_xlabel('Player ID')
            ax3.set_ylabel('Activity Score')
        
        # 4. Court coverage heatmap
        ax4 = fig.add_subplot(gs[1, :])
        if ball_data:
            self._plot_court_coverage(ax4, ball_data)
        
        # 5. Timeline of events
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_event_timeline(ax5, analysis_results)
        
        plt.suptitle('Tennis Video Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"analysis_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _calculate_detection_rate(self, timestamps, total_duration, window_size=10):
        time_bins = np.arange(0, total_duration, window_size)
        detection_counts = np.histogram(timestamps, bins=time_bins)[0]
        detection_rate = (detection_counts / window_size) * 100  # Percentage
        
        return {
            'time_bins': time_bins[:-1] + window_size/2,
            'rate': detection_rate
        }
    
    def _calculate_speeds(self, ball_data):
        speeds = []
        for i in range(1, len(ball_data)):
            dt = ball_data[i]['timestamp'] - ball_data[i-1]['timestamp']
            if dt > 0:
                pos1 = ball_data[i-1]['position']
                pos2 = ball_data[i]['position']
                distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                speed = distance / dt
                speeds.append(speed)
        return speeds
    
    def _calculate_player_activity(self, player_data):
        # Calculate activity levels for each player based on movement
        activity_levels = {}
        
        # Group by player ID
        players = {}
        for entry in player_data:
            for player in entry['players']:
                player_id = player.get('id', 'unknown')
                if player_id not in players:
                    players[player_id] = []
                players[player_id].append(player['center'])
        
        # Calculate movement for each player
        for player_id, positions in players.items():
            if len(positions) > 1:
                total_movement = 0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    movement = np.sqrt(dx**2 + dy**2)
                    total_movement += movement
                activity_levels[player_id] = total_movement
            else:
                activity_levels[player_id] = 0
        
        return activity_levels
    
    def _plot_court_coverage(self, ax, ball_data):
        if not ball_data:
            ax.text(0.5, 0.5, 'No ball data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        positions = np.array([entry['position'] for entry in ball_data])
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=30
        )
        
        # Plot heatmap
        im = ax.imshow(heatmap.T, origin='lower', cmap='hot', alpha=0.8, aspect='auto')
        ax.set_title('Ball Court Coverage')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Ball Presence')
    
    def _plot_event_timeline(self, ax, analysis_results):
        # Create a timeline showing different events
        events = []
        
        # Add ball tracking events
        ball_data = analysis_results.get('ball_tracking', [])
        if ball_data:
            events.extend([('Ball', entry['timestamp']) for entry in ball_data])
        
        # Add player detection events
        player_data = analysis_results.get('player_positions', [])
        if player_data:
            events.extend([('Player', entry['timestamp']) for entry in player_data])
        
        if not events:
            ax.text(0.5, 0.5, 'No timeline data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Group events by type
        event_types = {}
        for event_type, timestamp in events:
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(timestamp)
        
        # Plot timeline
        y_offset = 0
        colors = {'Ball': 'yellow', 'Player': 'blue'}
        
        for event_type, timestamps in event_types.items():
            ax.scatter(timestamps, [y_offset] * len(timestamps), 
                      c=colors.get(event_type, 'gray'), alpha=0.6, s=10)
            ax.text(-0.02, y_offset, event_type, transform=ax.get_yaxis_transform(), 
                   ha='right', va='center')
            y_offset += 1
        
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Event Timeline')
        ax.set_ylim(-0.5, len(event_types) - 0.5)
        ax.grid(True, alpha=0.3)
    
    def create_video_summary_report(self, analysis_results, output_path=None):
        # Create a comprehensive text report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"analysis_report_{timestamp}.txt"
        
        with open(output_path, 'w') as f:
            f.write("TENNIS VIDEO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Video information
            video_info = analysis_results.get('video_info', {})
            f.write("VIDEO INFORMATION:\n")
            f.write(f"  File: {video_info.get('path', 'Unknown')}\n")
            f.write(f"  Duration: {video_info.get('duration', 0):.2f} seconds\n")
            f.write(f"  FPS: {video_info.get('fps', 0):.2f}\n")
            f.write(f"  Frame Count: {video_info.get('frame_count', 0)}\n\n")
            
            # Ball tracking statistics
            ball_data = analysis_results.get('ball_tracking', [])
            f.write("BALL TRACKING:\n")
            f.write(f"  Total Detections: {len(ball_data)}\n")
            
            if ball_data:
                speeds = self._calculate_speeds(ball_data)
                if speeds:
                    f.write(f"  Average Speed: {np.mean(speeds):.2f} px/s\n")
                    f.write(f"  Maximum Speed: {np.max(speeds):.2f} px/s\n")
                    f.write(f"  Minimum Speed: {np.min(speeds):.2f} px/s\n")
            f.write("\n")
            
            # Player statistics
            player_data = analysis_results.get('player_positions', [])
            f.write("PLAYER TRACKING:\n")
            f.write(f"  Total Player Detections: {len(player_data)}\n")
            
            if player_data:
                activity_levels = self._calculate_player_activity(player_data)
                for player_id, activity in activity_levels.items():
                    f.write(f"  Player {player_id} Activity: {activity:.2f}\n")
            f.write("\n")
            
            # Court detection
            court_data = analysis_results.get('court_lines')
            f.write("COURT DETECTION:\n")
            if court_data:
                f.write(f"  Court Lines Detected: {len(court_data)}\n")
            else:
                f.write("  No court lines detected\n")
            f.write("\n")
            
            # Overall statistics
            stats = analysis_results.get('statistics', {})
            f.write("OVERALL STATISTICS:\n")
            for key, value in stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        
        return output_path
    
    def save_analysis_data(self, analysis_results, format='json'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = self.output_dir / f"analysis_data_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
        
        elif format == 'csv':
            # Save ball tracking data as CSV
            ball_data = analysis_results.get('ball_tracking', [])
            if ball_data:
                df = pd.DataFrame(ball_data)
                filename = self.output_dir / f"ball_tracking_{timestamp}.csv"
                df.to_csv(filename, index=False)
            
            # Save player data as CSV
            player_data = analysis_results.get('player_positions', [])
            if player_data:
                # Flatten player data
                flattened_data = []
                for entry in player_data:
                    for player in entry['players']:
                        row = {
                            'frame': entry['frame'],
                            'timestamp': entry['timestamp'],
                            'player_id': player.get('id', 'unknown'),
                            'center_x': player['center'][0],
                            'center_y': player['center'][1],
                            'bbox_x': player['bbox'][0],
                            'bbox_y': player['bbox'][1],
                            'bbox_w': player['bbox'][2],
                            'bbox_h': player['bbox'][3]
                        }
                        flattened_data.append(row)
                
                if flattened_data:
                    df = pd.DataFrame(flattened_data)
                    filename = self.output_dir / f"player_tracking_{timestamp}.csv"
                    df.to_csv(filename, index=False)
        
        return filename


def create_analysis_video(video_path, analysis_results, output_path, fps=30):
    # Create an annotated video with analysis overlays
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Get analysis data
    ball_data = {entry['frame']: entry for entry in analysis_results.get('ball_tracking', [])}
    player_data = {entry['frame']: entry for entry in analysis_results.get('player_positions', [])}
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw ball tracking
        if frame_num in ball_data:
            ball_pos = ball_data[frame_num]['position']
            cv2.circle(frame, tuple(map(int, ball_pos)), 10, (0, 255, 255), 2)
            cv2.putText(frame, 'Ball', (int(ball_pos[0]) + 15, int(ball_pos[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw player tracking
        if frame_num in player_data:
            for player in player_data[frame_num]['players']:
                bbox = player['bbox']
                player_id = player.get('id', 'Unknown')
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                
                # Draw player ID
                cv2.putText(frame, f'Player {player_id}', 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add frame number and timestamp
        timestamp = frame_num / original_fps
        cv2.putText(frame, f'Frame: {frame_num} | Time: {timestamp:.2f}s',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_num += 1
        
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames...")
    
    cap.release()
    out.release()
    print(f"Analysis video saved to: {output_path}")