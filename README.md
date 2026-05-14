# Vehicle Counting System

A professional vehicle counting system using YOLOv8 for detection and DeepSORT for tracking.

## Features

- Real-time vehicle detection (car, truck, bus, motorcycle)
- Multi-object tracking with DeepSORT
- Vehicle counting with customizable counting lines
- Entry/exit zone analysis
- Live statistics dashboard
- WebSocket-based real-time monitoring
- Video file processing with results download
- FastAPI REST API backend
- Clean architecture with type hints and docstrings

## Recent Fixes

### DeepSORT Confidence Value Validation
- Fixed an issue where DeepSORT tracks sometimes return None confidence values causing validation errors
- Added proper handling in the tracking service to assign default confidence values when None is returned
- Improved error handling and reporting for tracking operations
- Created comprehensive tracking system tests to verify the fix

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-counting.git
cd vehicle-counting

# Create virtual environment and install dependencies
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Start the system with a single command
./start.sh

# Open the web interface in your browser
# http://localhost:8000/static/index.html
```

## API Endpoints

### REST API

- `GET /api/v1/health`: Health check endpoint
- `GET /api/v1/model-info`: Get information about the loaded model
- `GET /api/v1/stats`: Get current counting statistics
- `POST /api/v1/reset-counts`: Reset all counting statistics
- `POST /api/v1/counting-lines`: Set counting lines for vehicle counting
- `POST /api/v1/process-video`: Upload and process a video file
- `GET /api/v1/job-status/{job_id}`: Get status of a video processing job
- `GET /api/v1/download-result/{job_id}`: Download processed video result
- `POST /api/v1/camera/start`: Start camera processing
- `POST /api/v1/camera/stop`: Stop camera processing
- `GET /api/v1/test-detection`: Test endpoint to verify detection service

### WebSocket Endpoints

- `/api/v1/webcam-stream`: WebSocket endpoint for real-time webcam processing
- `/api/v1/live-stats`: WebSocket endpoint for live statistics updates

## Components

- **Detection Service**: Uses YOLOv8 to detect vehicles in frames
- **Tracking Service**: Uses DeepSORT to track vehicles across frames
- **Counting Service**: Analyzes tracks to count vehicles crossing defined lines
- **Video Processing Service**: Integrates detection, tracking, and counting

## Project Structure

```
vehicle_counting/
├── app/                  # Main application code
│   ├── api/              # API routes and endpoints
│   ├── core/             # Core configuration
│   ├── models/           # Data models and schemas
│   ├── services/         # Business logic services
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── data/                 # Data storage
│   ├── outputs/          # Processed video outputs
│   ├── uploads/          # Temporary video uploads
│   └── videos/           # Sample videos for testing
├── logs/                 # Application logs
├── scripts/              # Utility scripts
├── static/               # Static web files
│   └── index.html        # Web interface
├── tests/                # Test files
├── requirements.txt      # Dependencies
└── start.sh              # Startup script
```

## License

MIT

## Acknowledgements

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [DeepSORT](https://github.com/levan92/deep_sort_realtime) for object tracking
- [FastAPI](https://fastapi.tiangolo.com/) for the web server framework