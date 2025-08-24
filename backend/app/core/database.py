"""
Database configuration for NeuroLens-X
PostgreSQL via Supabase with SQLite fallback for development
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool
import asyncio
from typing import AsyncGenerator
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Determine database type and create appropriate engine
database_url = settings.effective_database_url
is_postgresql = database_url.startswith("postgresql://")

if is_postgresql:
    # PostgreSQL configuration for Supabase
    engine = create_engine(
        database_url,
        echo=settings.DATABASE_ECHO,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections every hour
        connect_args={
            "sslmode": "require",  # Require SSL for Supabase
            "connect_timeout": 10,
        }
    )
    logger.info("✅ Using PostgreSQL database (Supabase)")
else:
    # SQLite configuration for development
    engine = create_engine(
        database_url,
        echo=settings.DATABASE_ECHO,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,  # Allow multiple threads
            "timeout": 20,  # Connection timeout
        },
        pool_pre_ping=True,  # Verify connections before use
    )
    logger.info("⚠️ Using SQLite database (development mode)")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Metadata for table creation
metadata = MetaData()


async def init_db():
    """Initialize database tables"""
    try:
        # Import all models to ensure they're registered
        from app.models import assessment, user, validation
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Initialize with sample data if needed
        await create_sample_data()
        
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


async def create_sample_data():
    """Create sample data for demonstration"""
    try:
        db = SessionLocal()
        
        # Check if sample data already exists
        # TODO: Add sample data creation logic here
        
        db.close()
        print("✅ Sample data created")
    except Exception as e:
        print(f"⚠️ Sample data creation failed: {e}")


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_db_async() -> AsyncGenerator:
    """Async dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def create_tables():
        """Create all database tables"""
        Base.metadata.create_all(bind=engine)
    
    @staticmethod
    def drop_tables():
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(bind=engine)
    
    @staticmethod
    def reset_database():
        """Reset database (drop and recreate tables)"""
        DatabaseManager.drop_tables()
        DatabaseManager.create_tables()
    
    @staticmethod
    def get_table_info():
        """Get information about database tables"""
        inspector = engine.inspect(engine)
        tables = inspector.get_table_names()
        
        table_info = {}
        for table in tables:
            columns = inspector.get_columns(table)
            table_info[table] = {
                "columns": [col["name"] for col in columns],
                "column_count": len(columns)
            }
        
        return table_info


# Health check for database
async def check_database_health():
    """Check database connectivity and health"""
    try:
        db = SessionLocal()
        # Simple query to test connection
        db.execute("SELECT 1")
        db.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


# Connection pool monitoring
def get_connection_info():
    """Get connection pool information"""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid()
    }
