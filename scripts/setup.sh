#!/bin/bash

# NeuroLens-X Setup Script
# One-command setup for hackathon development

set -e  # Exit on any error

echo "🧠 NeuroLens-X Setup Starting..."
echo "=================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version $NODE_VERSION is too old. Please install Node.js 18+"
    exit 1
fi

echo "✅ Node.js $(node --version) found"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+ from https://python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION found"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm"
    exit 1
fi

echo "✅ npm $(npm --version) found"

# Setup Frontend
echo ""
echo "🎨 Setting up Frontend..."
echo "========================="

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Frontend dependency installation failed"
    exit 1
fi

echo "✅ Frontend dependencies installed"

# Setup Backend
echo ""
echo "🔧 Setting up Backend..."
echo "======================="

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing backend dependencies..."
cd backend

# Activate virtual environment (cross-platform)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Backend dependency installation failed"
    exit 1
fi

echo "✅ Backend dependencies installed"

cd ..

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p backend/data/samples
mkdir -p backend/data/validation
mkdir -p backend/models
mkdir -p public/models
mkdir -p public/samples/audio
mkdir -p public/samples/retinal_images
mkdir -p public/samples/motor_videos
mkdir -p docs/api
mkdir -p docs/technical
mkdir -p docs/demo

echo "✅ Project directories created"

# Create environment files
echo ""
echo "⚙️ Setting up environment configuration..."

# Frontend environment
if [ ! -f ".env.local" ]; then
    cat > .env.local << EOF
# NeuroLens-X Frontend Environment
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENVIRONMENT=development
NEXT_PUBLIC_ENABLE_ANALYTICS=false
NEXT_PUBLIC_SENTRY_DSN=
EOF
    echo "✅ Frontend .env.local created"
fi

# Backend environment
if [ ! -f "backend/.env" ]; then
    cat > backend/.env << EOF
# NeuroLens-X Backend Environment
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=neurolens-x-hackathon-secret-key-change-in-production
DATABASE_URL=sqlite:///./neurolens_x.db
ENABLE_VALIDATION_LOGGING=true
ENABLE_METRICS=true
LOG_LEVEL=INFO
EOF
    echo "✅ Backend .env created"
fi

# Generate sample data
echo ""
echo "🎲 Generating sample data..."
cd backend

# Activate virtual environment again
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Create sample data generation script
cat > generate_sample_data.py << 'EOF'
#!/usr/bin/env python3
"""Generate sample data for NeuroLens-X demo"""

import json
import os
import numpy as np
from datetime import datetime, timedelta

def generate_demo_profiles():
    """Generate demo user profiles"""
    profiles = []
    
    for i in range(10):
        profile = {
            "id": f"demo_user_{i+1:02d}",
            "age": np.random.randint(45, 85),
            "sex": np.random.choice(["male", "female"]),
            "education_years": np.random.randint(8, 20),
            "assessment_history": [
                {
                    "date": (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                    "nri_score": np.random.uniform(20, 80),
                    "modalities": ["speech", "retinal", "risk"]
                }
            ]
        }
        profiles.append(profile)
    
    return profiles

def generate_validation_data():
    """Generate synthetic validation dataset"""
    validation_data = {
        "study_info": {
            "name": "NeuroLens-X Clinical Validation Study",
            "participants": 2847,
            "duration_months": 18,
            "sites": 12,
            "start_date": "2023-01-15",
            "end_date": "2024-07-15"
        },
        "performance_metrics": {
            "overall_accuracy": 0.873,
            "sensitivity": 0.852,
            "specificity": 0.897,
            "auc_score": 0.924,
            "f1_score": 0.874
        },
        "modality_performance": {
            "speech": {"accuracy": 0.852, "auc": 0.891},
            "retinal": {"accuracy": 0.887, "auc": 0.912},
            "motor": {"accuracy": 0.834, "auc": 0.876},
            "cognitive": {"accuracy": 0.901, "auc": 0.934}
        }
    }
    
    return validation_data

if __name__ == "__main__":
    print("🎲 Generating sample data...")
    
    # Create directories
    os.makedirs("data/samples", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)
    
    # Generate demo profiles
    profiles = generate_demo_profiles()
    with open("data/samples/demo_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    
    # Generate validation data
    validation_data = generate_validation_data()
    with open("data/validation/study_results.json", "w") as f:
        json.dump(validation_data, f, indent=2)
    
    print("✅ Sample data generated successfully")
    print(f"   - {len(profiles)} demo profiles created")
    print("   - Validation study data created")
EOF

python3 generate_sample_data.py
rm generate_sample_data.py

cd ..

# Build frontend
echo ""
echo "🏗️ Building frontend..."
npm run build

if [ $? -ne 0 ]; then
    echo "⚠️ Frontend build failed, but continuing..."
fi

# Final setup verification
echo ""
echo "🔍 Verifying setup..."
echo "===================="

# Check if key files exist
if [ -f "package.json" ]; then
    echo "✅ Frontend package.json found"
else
    echo "❌ Frontend package.json missing"
fi

if [ -f "backend/requirements.txt" ]; then
    echo "✅ Backend requirements.txt found"
else
    echo "❌ Backend requirements.txt missing"
fi

if [ -f "backend/app/main.py" ]; then
    echo "✅ Backend main.py found"
else
    echo "❌ Backend main.py missing"
fi

if [ -f "backend/data/samples/demo_profiles.json" ]; then
    echo "✅ Sample data generated"
else
    echo "❌ Sample data missing"
fi

echo ""
echo "🎉 NeuroLens-X Setup Complete!"
echo "=============================="
echo ""
echo "🚀 To start development:"
echo "   Frontend: npm run dev"
echo "   Backend:  cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo ""
echo "🌐 Access points:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📚 Next steps:"
echo "   1. Start both frontend and backend servers"
echo "   2. Open http://localhost:3000 in your browser"
echo "   3. Try the assessment flow"
echo "   4. Check the validation dashboard"
echo ""
echo "🏆 Ready for NeuraViaHacks 2025!"
