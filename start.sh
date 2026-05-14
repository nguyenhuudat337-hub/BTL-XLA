#!/bin/bash

# Vehicle Counting System Startup Script

echo "🚗 Starting Vehicle Counting System..."
echo "=====================================\n"

# Check if virtual environment exists
if [ ! -d ".env" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv .env"
    echo "   source .env/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .env/bin/activate

# Check if requirements are installed
echo "🔍 Checking dependencies..."
if ! python -c "import ultralytics, cv2, fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/uploads data/outputs logs static

# Download YOLOv8 model if not exists
if [ ! -f "yolov8n.pt" ]; then
    echo "⬇️  Downloading YOLOv8 model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# Start the application
echo "🚀 Starting FastAPI server..."
echo "\n📱 Web Interface: http://localhost:8000/static/index.html"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "💚 Health Check: http://localhost:8000/api/v1/health"
echo "\nPress Ctrl+C to stop the server\n"

# Run with hot reload for development
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
