# Let The Game Speak - Full Stack Startup Script (Windows)

Write-Host "================================================" -ForegroundColor Green
Write-Host "🎬 LET THE GAME SPEAK - Starting Full Stack" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "🔍 Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check Node.js
Write-Host "🔍 Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Node.js $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Node.js not found! Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Install backend dependencies
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install -r backend\requirements.txt

Write-Host ""

# Start backend
Write-Host "🚀 Starting FastAPI backend on port 8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend\app; uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

Write-Host "✅ Backend started in new window" -ForegroundColor Green
Write-Host ""

# Wait for backend to start
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Install frontend dependencies and start
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend
npm install

Write-Host ""
Write-Host "🎨 Starting React frontend on port 5173..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev"

Set-Location ..

Write-Host "✅ Frontend started in new window" -ForegroundColor Green
Write-Host ""

Write-Host "================================================" -ForegroundColor Green
Write-Host "✅ Full Stack Running!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "🔙 Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🎨 Frontend:    http://localhost:5173" -ForegroundColor Cyan
Write-Host "📚 API Docs:    http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Close the terminal windows to stop services" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Green
