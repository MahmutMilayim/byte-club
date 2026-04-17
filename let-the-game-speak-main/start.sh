#!/bin/bash

# Let The Game Speak - Full Stack Startup Script

echo "================================================"
echo "🎬 LET THE GAME SPEAK - Starting Full Stack"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
cd backend/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ../..

echo "✅ Backend started (PID: $BACKEND_PID)"
echo ""

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting React frontend on port 5173..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo ""

echo "================================================"
echo "✅ Full Stack Running!"
echo "================================================"
echo "🔙 Backend API: http://localhost:8000"
echo "🎨 Frontend:    http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"
echo "================================================"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
