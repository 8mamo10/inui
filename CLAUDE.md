# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a tennis video analysis system that uses computer vision and machine learning techniques to analyze tennis videos. The system can track ball movement, detect players, identify court boundaries, and generate comprehensive statistics and visualizations.

## Core Architecture

### Main Components

- **tennis_analyzer.py** - Main orchestration class that coordinates all analysis components
- **video_processor.py** - Handles video file operations, frame extraction, and basic video processing
- **ball_tracker.py** - Advanced ball tracking using multiple detection methods (color, motion, Hough circles) with Kalman filtering
- **player_detector.py** - Player detection and tracking with activity analysis
- **court_detector.py** - Tennis court detection and perspective transformation
- **analysis_visualizer.py** - Statistical analysis and visualization generation

### Key Features

- Multi-method ball tracking with confidence scoring
- Player detection and movement analysis
- Court line detection and homography transformation
- Real-time trajectory analysis
- Comprehensive visualization dashboard
- Export to multiple formats (JSON, CSV, reports)
- Annotated video generation

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Analysis
```bash
# Basic analysis
python tennis_analyzer.py path/to/video.mp4

# Full analysis with all features
python tennis_analyzer.py path/to/video.mp4 --create-video --extract-keyframes --verbose

# Custom output directory
python tennis_analyzer.py path/to/video.mp4 -o custom_output_dir
```

### Key Processing Functions

- **Frame extraction**: `VideoProcessor.extract_frames()` - extracts frames for analysis
- **Ball tracking**: `AdvancedBallTracker.track_ball()` - main ball detection method
- **Player detection**: `PlayerDetector.detect_players()` - finds and tracks players
- **Court detection**: `TennisCourtDetector.detect_court()` - identifies court boundaries
- **Visualization**: `TennisAnalysisVisualizer.create_*_plot()` - generates analysis plots

## Code Architecture Patterns

### Analysis Pipeline
1. Video loading and preprocessing (`VideoProcessor`)
2. Court detection in first frame (`TennisCourtDetector`)
3. Frame-by-frame analysis loop:
   - Ball tracking (`AdvancedBallTracker`)
   - Player detection (`PlayerDetector`)
   - Activity analysis (`PlayerActivityAnalyzer`)
4. Statistics compilation and trajectory analysis
5. Visualization and report generation (`TennisAnalysisVisualizer`)

### Tracking Architecture
- **Multi-method detection**: Each tracker uses multiple detection algorithms and combines results
- **Confidence scoring**: All detections include confidence scores for quality assessment
- **Kalman filtering**: Position prediction for robust tracking through occlusions
- **History management**: Trajectory analysis with configurable history lengths

### Output Structure
```
analysis_output/
├── analysis_YYYYMMDD_HHMMSS.json     # Complete analysis results
├── analysis_report_YYYYMMDD_HHMMSS.txt # Human-readable report
├── visualizations/                    # Generated plots and charts
│   ├── ball_trajectory_*.png
│   ├── player_heatmap_*.png
│   ├── court_analysis_*.png
│   └── analysis_dashboard_*.png
├── ball_tracking_*.csv               # Ball tracking data
├── player_tracking_*.csv            # Player tracking data
└── annotated_video_*.mp4            # Video with analysis overlays
```

## Important Implementation Details

### Ball Tracking Strategy
The system uses a three-pronged approach:
1. **Color-based detection**: HSV thresholding for tennis ball yellow/green
2. **Shape-based detection**: Hough circle transform for circular objects
3. **Motion-based detection**: Background subtraction for moving objects

### Player Detection Methods
- Primary: Background subtraction with morphological operations
- Fallback: Deep learning models (YOLO) when available
- Validation: Aspect ratio filtering and confidence scoring

### Court Detection Algorithm
- Edge detection with Canny
- Line detection with Hough Line Transform
- Line classification (horizontal/vertical)
- Corner point calculation from line intersections
- Homography matrix calculation for perspective correction

### Performance Considerations
- Frame processing is the main bottleneck
- Consider frame skipping for faster processing: `VideoProcessor.extract_frames(step=N)`
- Large videos should use chunked processing: `VideoProcessor.process_video_in_chunks()`
- Memory usage scales with trajectory history length

## Testing and Validation

The system works best with:
- Clear, well-lit tennis court footage
- Minimal camera shake
- Contrasting ball color (yellow/green tennis balls)
- Standard court layouts and angles

For best results, use videos with:
- Resolution: 720p or higher
- Frame rate: 30fps or higher
- Duration: Under 10 minutes for initial testing