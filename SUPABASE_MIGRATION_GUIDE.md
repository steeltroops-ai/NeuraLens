# NeuraLens Supabase Migration Guide

## Overview

This guide will help you migrate NeuraLens from SQLite to Supabase PostgreSQL and implement proper file storage for audio and image files.

## Prerequisites

1. **Supabase Account**: Sign up at [supabase.com](https://supabase.com)
2. **Active Supabase Project**: Your project `supabase-pink-park` needs to be active
3. **Python Environment**: Ensure you have Python 3.8+ installed

## Step 1: Reactivate Your Supabase Project

Since your project is currently inactive, you'll need to reactivate it:

1. Go to [Supabase Dashboard](https://app.supabase.com/)
2. Find your project `supabase-pink-park` (ID: juyebmhkqjjnttvelvpp)
3. Click "Restore" or "Reactivate" if available
4. If the project cannot be reactivated, create a new project

## Step 2: Get Supabase Credentials

From your Supabase project dashboard:

### API Keys (Settings → API)
- **Project URL**: `https://juyebmhkqjjnttvelvpp.supabase.co`
- **Anon Key**: Copy the `anon` key
- **Service Role Key**: Copy the `service_role` key (keep this secret!)

### Database Credentials (Settings → Database)
- **Host**: `db.juyebmhkqjjnttvelvpp.supabase.co`
- **Database**: `postgres`
- **User**: `postgres`
- **Password**: Your database password (set during project creation)
- **Port**: `5432`

## Step 3: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## Step 4: Configure Environment

1. Copy the example environment file:
```bash
cp .env.supabase.example .env
```

2. Edit `.env` with your Supabase credentials:
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

## Step 5: Run Migration

Execute the migration script:

```bash
cd backend
python migrate_to_supabase.py
```

This script will:
- ✅ Verify Supabase connection
- ✅ Create database tables in PostgreSQL
- ✅ Set up storage buckets for files
- ✅ Verify the migration

## Step 6: Test the Setup

1. **Start the backend**:
```bash
cd backend
python -m uvicorn app.main:app --reload
```

2. **Test API endpoints**:
```bash
# Health check
curl http://localhost:8000/api/v1/status

# Test retinal analysis endpoint
curl -X POST "http://localhost:8000/api/v1/retinal/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image_file=@test_image.jpg"
```

3. **Check Supabase Dashboard**:
   - Go to Table Editor to see your tables
   - Go to Storage to see your buckets

## Step 7: Update Frontend (if needed)

The frontend should work without changes, but verify API calls are working:

```bash
cd frontend
bun run dev
```

## Storage Buckets Created

The migration creates three storage buckets:

1. **neuralens-audio**: For speech analysis audio files
   - Max size: 10MB
   - Allowed types: audio/wav, audio/mp3, audio/m4a, audio/webm

2. **neuralens-images**: For retinal analysis images
   - Max size: 5MB
   - Allowed types: image/jpeg, image/png, image/jpg

3. **neuralens-reports**: For generated reports
   - Max size: 2MB
   - Allowed types: application/pdf, text/plain, application/json

## Security Features

### Row Level Security (RLS)
- Automatically enabled on all tables
- Users can only access their own data
- Service role bypasses RLS for backend operations

### File Security
- Files are private by default
- Signed URLs for temporary access
- Automatic file cleanup after 30 days

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Error: Failed to connect to Supabase
   ```
   - Check your credentials in `.env`
   - Ensure your project is active
   - Verify network connectivity

2. **Permission Denied**
   ```
   Error: permission denied for table
   ```
   - Use the service role key for backend operations
   - Check RLS policies in Supabase dashboard

3. **Storage Upload Failed**
   ```
   Error: Storage upload failed
   ```
   - Check bucket permissions
   - Verify file size limits
   - Ensure correct MIME types

### Debug Mode

Enable debug logging:
```env
DATABASE_ECHO=true
LOG_LEVEL=DEBUG
```

## Performance Optimization

### Database
- Connection pooling is automatically configured
- SSL connections are required and enabled
- Connection recycling every hour

### Storage
- Files are organized by date: `speech/2024/01/15/filename.wav`
- Automatic cleanup of old files
- CDN caching for faster access

## Monitoring

### Health Checks
```bash
# Database health
curl http://localhost:8000/api/v1/validation/health

# Storage health
curl http://localhost:8000/api/v1/retinal/health
```

### Storage Statistics
The backend provides storage usage statistics through the API.

## Next Steps

1. **Generate Demo Data**:
```bash
cd backend
python generate_demo_data.py
```

2. **Run End-to-End Tests**:
```bash
cd backend
python test_end_to_end_flow.py
```

3. **Deploy to Production**:
   - Update environment variables for production
   - Configure proper SSL certificates
   - Set up monitoring and logging

## Support

If you encounter issues:

1. Check the logs: `tail -f backend/logs/app.log`
2. Verify Supabase project status in dashboard
3. Test individual components with the provided test scripts
4. Check the comprehensive error messages in the migration script

## File Structure After Migration

```
backend/
├── app/
│   ├── core/
│   │   ├── supabase_config.py     # Supabase configuration
│   │   ├── database.py            # Updated for PostgreSQL
│   │   └── config.py              # Updated with Supabase settings
│   └── services/
│       └── supabase_storage.py    # File storage service
├── migrate_to_supabase.py         # Migration script
├── .env.supabase.example          # Environment template
└── requirements.txt               # Updated dependencies
```

This migration provides:
- ✅ Scalable PostgreSQL database
- ✅ Secure file storage with automatic cleanup
- ✅ Production-ready configuration
- ✅ Comprehensive error handling
- ✅ Performance optimization
- ✅ Security best practices
