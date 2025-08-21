#!/bin/bash

# NeuroLens-X Deployment Script
# One-click deployment to production (Vercel + Heroku)

set -e  # Exit on any error

echo "🚀 NeuroLens-X Deployment Starting..."
echo "====================================="

# Configuration
FRONTEND_PLATFORM="vercel"  # vercel, netlify, or github-pages
BACKEND_PLATFORM="heroku"   # heroku, railway, or render

# Check prerequisites
echo "📋 Checking deployment prerequisites..."

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git"
    exit 1
fi

echo "✅ Git found"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please initialize git first:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    exit 1
fi

echo "✅ Git repository found"

# Frontend Deployment
echo ""
echo "🎨 Deploying Frontend to $FRONTEND_PLATFORM..."
echo "=============================================="

case $FRONTEND_PLATFORM in
    "vercel")
        # Check if Vercel CLI is installed
        if ! command -v vercel &> /dev/null; then
            echo "📦 Installing Vercel CLI..."
            npm install -g vercel
        fi
        
        echo "✅ Vercel CLI found"
        
        # Create vercel.json if it doesn't exist
        if [ ! -f "vercel.json" ]; then
            cat > vercel.json << EOF
{
  "name": "neurolens-x",
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "https://neurolens-x-backend.herokuapp.com/api/v1/\$1"
    }
  ],
  "env": {
    "NEXT_PUBLIC_API_URL": "https://neurolens-x-backend.herokuapp.com",
    "NEXT_PUBLIC_ENVIRONMENT": "production"
  }
}
EOF
            echo "✅ vercel.json created"
        fi
        
        # Deploy to Vercel
        echo "🚀 Deploying to Vercel..."
        vercel --prod
        
        if [ $? -eq 0 ]; then
            echo "✅ Frontend deployed to Vercel successfully"
        else
            echo "❌ Vercel deployment failed"
            exit 1
        fi
        ;;
        
    "netlify")
        echo "🌐 Netlify deployment not implemented yet"
        echo "   Please deploy manually to Netlify"
        ;;
        
    *)
        echo "❌ Unknown frontend platform: $FRONTEND_PLATFORM"
        exit 1
        ;;
esac

# Backend Deployment
echo ""
echo "🔧 Deploying Backend to $BACKEND_PLATFORM..."
echo "============================================"

case $BACKEND_PLATFORM in
    "heroku")
        # Check if Heroku CLI is installed
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI not found. Please install from https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        
        echo "✅ Heroku CLI found"
        
        # Check if logged in to Heroku
        if ! heroku auth:whoami &> /dev/null; then
            echo "🔐 Please log in to Heroku:"
            heroku login
        fi
        
        echo "✅ Heroku authentication verified"
        
        # Create Heroku app if it doesn't exist
        APP_NAME="neurolens-x-backend"
        if ! heroku apps:info $APP_NAME &> /dev/null; then
            echo "🆕 Creating Heroku app: $APP_NAME"
            heroku create $APP_NAME
            
            # Set environment variables
            heroku config:set ENVIRONMENT=production -a $APP_NAME
            heroku config:set DEBUG=false -a $APP_NAME
            heroku config:set SECRET_KEY=$(openssl rand -base64 32) -a $APP_NAME
            heroku config:set DATABASE_URL=sqlite:///./neurolens_x.db -a $APP_NAME
            
            echo "✅ Heroku app created and configured"
        else
            echo "✅ Heroku app $APP_NAME already exists"
        fi
        
        # Create Procfile if it doesn't exist
        if [ ! -f "backend/Procfile" ]; then
            cat > backend/Procfile << EOF
web: cd backend && gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:\$PORT
EOF
            echo "✅ Procfile created"
        fi
        
        # Create runtime.txt if it doesn't exist
        if [ ! -f "backend/runtime.txt" ]; then
            echo "python-3.9.18" > backend/runtime.txt
            echo "✅ runtime.txt created"
        fi
        
        # Add Heroku remote if it doesn't exist
        if ! git remote get-url heroku &> /dev/null; then
            heroku git:remote -a $APP_NAME
            echo "✅ Heroku remote added"
        fi
        
        # Deploy to Heroku
        echo "🚀 Deploying to Heroku..."
        
        # Create a temporary branch for deployment
        git checkout -b heroku-deploy
        
        # Move backend files to root for Heroku
        cp -r backend/* .
        git add .
        git commit -m "Prepare for Heroku deployment"
        
        # Push to Heroku
        git push heroku heroku-deploy:main --force
        
        if [ $? -eq 0 ]; then
            echo "✅ Backend deployed to Heroku successfully"
            
            # Clean up
            git checkout main
            git branch -D heroku-deploy
            
            # Get the app URL
            APP_URL=$(heroku apps:info $APP_NAME --json | python3 -c "import sys, json; print(json.load(sys.stdin)['app']['web_url'])")
            echo "🌐 Backend URL: $APP_URL"
            
        else
            echo "❌ Heroku deployment failed"
            git checkout main
            git branch -D heroku-deploy
            exit 1
        fi
        ;;
        
    "railway")
        echo "🚂 Railway deployment not implemented yet"
        echo "   Please deploy manually to Railway"
        ;;
        
    *)
        echo "❌ Unknown backend platform: $BACKEND_PLATFORM"
        exit 1
        ;;
esac

# Post-deployment verification
echo ""
echo "🔍 Verifying deployment..."
echo "=========================="

# Test frontend
echo "🎨 Testing frontend..."
if curl -s -o /dev/null -w "%{http_code}" https://neurolens-x.vercel.app | grep -q "200"; then
    echo "✅ Frontend is responding"
else
    echo "⚠️ Frontend may not be fully ready yet"
fi

# Test backend
echo "🔧 Testing backend..."
if curl -s -o /dev/null -w "%{http_code}" https://neurolens-x-backend.herokuapp.com/health | grep -q "200"; then
    echo "✅ Backend is responding"
else
    echo "⚠️ Backend may not be fully ready yet"
fi

# Final deployment summary
echo ""
echo "🎉 NeuroLens-X Deployment Complete!"
echo "==================================="
echo ""
echo "🌐 Production URLs:"
echo "   Frontend: https://neurolens-x.vercel.app"
echo "   Backend:  https://neurolens-x-backend.herokuapp.com"
echo "   API Docs: https://neurolens-x-backend.herokuapp.com/docs"
echo ""
echo "📊 Monitoring:"
echo "   Vercel Dashboard: https://vercel.com/dashboard"
echo "   Heroku Dashboard: https://dashboard.heroku.com/apps/neurolens-x-backend"
echo ""
echo "🔧 Management Commands:"
echo "   View logs: heroku logs --tail -a neurolens-x-backend"
echo "   Scale app: heroku ps:scale web=1 -a neurolens-x-backend"
echo "   Restart:   heroku restart -a neurolens-x-backend"
echo ""
echo "🏆 Ready for NeuraViaHacks 2025 judging!"
echo ""
echo "📝 Don't forget to:"
echo "   1. Test all assessment flows"
echo "   2. Verify validation dashboard"
echo "   3. Check mobile responsiveness"
echo "   4. Prepare demo script"
