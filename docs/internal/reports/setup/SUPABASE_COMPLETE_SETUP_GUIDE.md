# NeuraLens Complete Supabase Setup Guide

## 🎯 Overview

This guide provides step-by-step instructions to migrate NeuraLens from SQLite to Supabase PostgreSQL with complete file storage integration. Your Supabase project `supabase-pink-park` (ID: juyebmhkqjjnttvelvpp) has been reactivated and is ready for setup.

## 📋 Prerequisites

- ✅ Supabase project reactivated: `supabase-pink-park`
- ✅ Python 3.8+ installed
- ✅ Bun package manager for frontend
- ✅ Internet connection for Supabase API calls

## 🚀 Step-by-Step Setup

### Step 1: Get Supabase Credentials

1. **Go to Supabase Dashboard**: https://app.supabase.com/project/juyebmhkqjjnttvelvpp

2. **Get API Keys** (Settings → API):
   - Project URL: `https://juyebmhkqjjnttvelvpp.supabase.co`
   - Anon Key: Copy the `anon` key
   - Service Role Key: Copy the `service_role` key (keep secret!)

3. **Get Database Credentials** (Settings → Database):
   - Host: `db.juyebmhkqjjnttvelvpp.supabase.co`
   - Database: `postgres`
   - User: `postgres`
   - Password: Your database password
   - Port: `5432`

### Step 2: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Configure Environment

1. **Create environment file**:
```bash
cp .env.supabase.example .env
```

2. **Edit `.env` with your credentials**:
```env
# Supabase Project Configuration
SUPABASE_URL=https://juyebmhkqjjnttvelvpp.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# Supabase Database Configuration
SUPABASE_DB_HOST=db.juyebmhkqjjnttvelvpp.supabase.co
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=your_database_password_here
SUPABASE_DB_NAME=postgres
SUPABASE_DB_PORT=5432
```

### Step 4: Run Complete Setup

```bash
cd backend
python setup_supabase_complete.py
```

This script will:
- ✅ Create complete database schema (12 tables)
- ✅ Set up Row Level Security policies
- ✅ Create storage buckets (audio, images, reports)
- ✅ Configure storage policies
- ✅ Test all integrations
- ✅ Generate configuration files

### Step 5: Verify Setup

```bash
# Test complete integration
python test_complete_integration.py

# Start backend server
python -m uvicorn app.main:app --reload

# Test API endpoints
curl http://localhost:8000/api/v1/validation/health
```

### Step 6: Test Frontend Integration

```bash
cd frontend
bun run dev
```

Visit http://localhost:3000 and test the assessment workflow.

## 📊 Database Schema Created

### Core Tables
- **users**: User accounts and demographics
- **user_profiles**: Extended user information
- **assessments**: Assessment sessions
- **assessment_results**: Results for each modality
- **nri_results**: NRI fusion results
- **assessment_history**: Historical tracking

### Validation Tables
- **validation_studies**: Clinical validation studies
- **validation_results**: Individual validation results
- **performance_metrics**: System performance tracking
- **system_health**: Health monitoring

### Features
- ✅ UUID primary keys for scalability
- ✅ JSONB columns for flexible data storage
- ✅ Proper foreign key relationships
- ✅ Indexes for optimal performance
- ✅ Automatic timestamp updates
- ✅ Data validation constraints

## 🗄️ Storage Buckets Created

### Audio Files (`neuralens-audio`)
- **Purpose**: Speech analysis audio files
- **Max Size**: 10MB per file
- **Formats**: WAV, MP3, M4A, WebM, OGG
- **Organization**: `user_id/speech/YYYY/MM/DD/session_uuid.ext`

### Images (`neuralens-images`)
- **Purpose**: Retinal analysis images
- **Max Size**: 5MB per file
- **Formats**: JPEG, PNG, JPG
- **Organization**: `user_id/retinal/YYYY/MM/DD/session_uuid.ext`

### Reports (`neuralens-reports`)
- **Purpose**: Generated assessment reports
- **Max Size**: 2MB per file
- **Formats**: PDF, JSON, TXT
- **Organization**: `user_id/reports/YYYY/MM/DD/report_uuid.ext`

## 🔒 Security Features

### Row Level Security (RLS)
- ✅ Users can only access their own data
- ✅ Clinicians can access assigned patient data
- ✅ Admins have full access
- ✅ Service role bypasses RLS for backend operations

### File Security
- ✅ Private buckets by default
- ✅ Signed URLs for temporary access
- ✅ User-based file organization
- ✅ Automatic cleanup after 30 days

### Authentication Ready
- ✅ JWT token support
- ✅ Role-based access control
- ✅ Secure API endpoints

## 🧪 Testing & Validation

### Automated Tests
```bash
# Complete integration test
python test_complete_integration.py

# Supabase-specific tests
python test_supabase_integration.py

# Database operations test
python verify_database_setup.py
```

### Manual Testing
1. **Database**: Check tables in Supabase dashboard
2. **Storage**: Upload files through API endpoints
3. **API**: Test all endpoints with Postman/curl
4. **Frontend**: Complete assessment workflow

## 📈 Performance Optimizations

### Database
- ✅ Connection pooling (10 connections, 20 overflow)
- ✅ SSL connections required
- ✅ Connection recycling every hour
- ✅ Optimized indexes for common queries

### Storage
- ✅ CDN caching for file access
- ✅ Organized file structure for performance
- ✅ Automatic file cleanup
- ✅ Signed URLs for secure access

### API
- ✅ Async/await for non-blocking operations
- ✅ Background tasks for file processing
- ✅ Proper error handling and timeouts
- ✅ Request/response optimization

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Error: Failed to connect to Supabase
   ```
   - ✅ Check credentials in `.env`
   - ✅ Ensure project is active
   - ✅ Verify network connectivity

2. **Permission Denied**
   ```
   Error: permission denied for table
   ```
   - ✅ Use service role key for backend
   - ✅ Check RLS policies in dashboard

3. **Storage Upload Failed**
   ```
   Error: Storage upload failed
   ```
   - ✅ Check bucket permissions
   - ✅ Verify file size limits
   - ✅ Ensure correct MIME types

### Debug Commands
```bash
# Check database connection
python -c "from app.core.database import engine; print(engine.execute('SELECT 1').scalar())"

# Test Supabase client
python -c "from app.core.supabase_config import supabase_client; print(supabase_client.health_check())"

# Check storage buckets
python -c "from app.services.supabase_storage import storage_service; print(storage_service.get_storage_stats())"
```

## 🎯 Next Steps

### Immediate Actions
1. ✅ Complete the setup following this guide
2. ✅ Test all functionality with provided scripts
3. ✅ Generate demo data for judge evaluation
4. ✅ Verify frontend-backend integration

### Production Deployment
1. 🔄 Set up environment variables in production
2. 🔄 Configure proper SSL certificates
3. 🔄 Set up monitoring and logging
4. 🔄 Implement backup strategies

### Advanced Features
1. 🔮 Implement real-time subscriptions
2. 🔮 Add advanced analytics dashboard
3. 🔮 Set up automated testing pipelines
4. 🔮 Implement advanced security features

## 📞 Support

If you encounter issues:

1. **Check Logs**: Look for detailed error messages
2. **Verify Setup**: Run the integration tests
3. **Manual Verification**: Check Supabase dashboard
4. **Fallback**: Use SQLite for development if needed

## 🏆 Success Metrics

After successful setup, you should have:

- ✅ **Database**: 12 tables with proper relationships
- ✅ **Storage**: 3 buckets with security policies
- ✅ **API**: All endpoints working with file upload
- ✅ **Frontend**: Complete assessment workflow
- ✅ **Security**: RLS policies protecting user data
- ✅ **Performance**: Sub-200ms API responses
- ✅ **Scalability**: Ready for thousands of users

**🎉 Congratulations! NeuraLens is now running on enterprise-grade Supabase infrastructure with complete file storage, security, and scalability features.**
