# NeuraLens Backend

## 🎯 **Overview**

The NeuraLens backend is a high-performance FastAPI application that provides comprehensive neurological assessment capabilities through advanced machine learning models and real-time analysis.

### **Key Features**

- **Multi-Modal Assessment**: Speech, retinal, motor, and cognitive analysis
- **Real-Time Processing**: Sub-100ms response times for critical assessments
- **ML Model Integration**: Whisper-tiny, computer vision, and custom neurological models
- **Scalable Architecture**: Async FastAPI with Supabase PostgreSQL
- **Production Ready**: Comprehensive error handling, logging, and monitoring

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.9+ (3.11 recommended)
- PostgreSQL 14+ (via Supabase)
- Visual C++ Build Tools (Windows only, for webrtcvad)

### **Installation**

#### **Windows**
```bash
# Run the setup script
scripts\setup-env.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### **macOS/Linux**
```bash
# Run the setup script
chmod +x scripts/setup-env.sh
./scripts/setup-env.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### **Environment Configuration**

Create a `.env` file in the backend directory:

```env
# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key

# API Configuration
API_V1_STR=/api/v1
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development
DEBUG=true

# CORS Configuration
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:3001"]

# ML Model Configuration
MODEL_CACHE_DIR=./models
MAX_FILE_SIZE_MB=50
PROCESSING_TIMEOUT_SECONDS=30
```

### **Development Server**

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Start development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Server will be available at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

---

## 📁 **Project Structure**

```
backend/
├── app/
│   ├── api/                    # API routes and endpoints
│   │   └── v1/
│   │       ├── endpoints/      # Individual endpoint modules
│   │       └── api.py         # API router configuration
│   ├── core/                  # Core application configuration
│   │   ├── config.py         # Settings and configuration
│   │   ├── security.py       # Authentication and security
│   │   └── response.py       # Response formatting
│   ├── ml/                   # Machine learning modules
│   │   ├── models/           # ML model implementations
│   │   ├── realtime/         # Real-time processing
│   │   └── utils/            # ML utilities
│   ├── schemas/              # Pydantic models and schemas
│   ├── services/             # Business logic services
│   ├── utils/                # Utility functions
│   └── main.py              # FastAPI application entry point
├── scripts/                  # Setup and utility scripts
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

---

## 🔬 **API Endpoints**

### **Health Check**
- `GET /health` - Application health status
- `GET /api/v1/health` - Detailed health check with dependencies

### **Assessment Endpoints**
- `POST /api/v1/speech/analyze` - Speech analysis with Whisper-tiny
- `POST /api/v1/retinal/analyze` - Retinal image analysis
- `POST /api/v1/motor/analyze` - Motor function assessment
- `POST /api/v1/cognitive/analyze` - Cognitive assessment processing
- `POST /api/v1/nri/calculate` - NRI score calculation and fusion

### **File Upload**
- `POST /api/v1/upload` - Secure file upload with validation
- Maximum file size: 50MB
- Supported formats: WAV, MP3, JPG, PNG, MP4

---

## 🧠 **Machine Learning Models**

### **Speech Analysis**
- **Model**: OpenAI Whisper-tiny
- **Features**: MFCC extraction, voice activity detection, neurological biomarkers
- **Processing Time**: <100ms target
- **Biomarkers**: Fluency, tremor, pause patterns, articulation clarity

### **Retinal Analysis**
- **Model**: Custom CNN for vessel analysis
- **Features**: Vessel segmentation, tortuosity analysis, hemorrhage detection
- **Processing Time**: <200ms target
- **Biomarkers**: Vessel health, retinal pathology indicators

### **Motor Assessment**
- **Model**: Computer vision-based movement analysis
- **Features**: Tremor detection, coordination assessment, gait analysis
- **Processing Time**: <150ms target
- **Biomarkers**: Movement quality, tremor intensity, coordination scores

### **Cognitive Testing**
- **Model**: Response time and accuracy analysis
- **Features**: Reaction time, pattern recognition, memory assessment
- **Processing Time**: Real-time
- **Biomarkers**: Cognitive speed, accuracy, attention metrics

---

## 🧪 **Testing**

### **Run Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_speech_analysis.py

# Run with verbose output
pytest -v
```

### **Test Structure**
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for API endpoints
- `tests/ml/` - Machine learning model tests
- `tests/fixtures/` - Test data and fixtures

---

## 🔧 **Development**

### **Code Quality**
```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

### **Database Migrations**
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### **Adding New Dependencies**
1. Add to `requirements.txt` with pinned version
2. Update virtual environment: `pip install -r requirements.txt`
3. Test thoroughly
4. Update documentation

---

## 🚀 **Deployment**

### **Production Configuration**
- Set `ENVIRONMENT=production` in `.env`
- Use strong `SECRET_KEY`
- Configure proper CORS origins
- Set up SSL/TLS certificates
- Configure monitoring and logging

### **Docker Deployment**
```bash
# Build image
docker build -t neuralens-backend .

# Run container
docker run -p 8000:8000 --env-file .env neuralens-backend
```

### **Performance Optimization**
- Use Gunicorn with multiple workers in production
- Configure connection pooling for database
- Implement Redis caching for frequently accessed data
- Use CDN for static assets

---

## 📊 **Monitoring & Logging**

### **Health Monitoring**
- Application health checks at `/health`
- Database connection monitoring
- ML model availability checks
- Memory and CPU usage tracking

### **Logging**
- Structured logging with JSON format
- Request/response logging
- Error tracking and alerting
- Performance metrics collection

---

## 🔒 **Security**

### **Authentication**
- JWT token-based authentication
- Secure password hashing with bcrypt
- Token expiration and refresh

### **Data Protection**
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Rate limiting on API endpoints

### **File Upload Security**
- File type validation
- Size limits enforcement
- Virus scanning (recommended for production)
- Secure file storage

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Format code: `black app/ tests/`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

---

## 📞 **Support**

- **Issues**: GitHub Issues
- **Documentation**: `/docs` endpoint when server is running
- **Email**: steeltroops.ai@gmail.com

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.
