"""
NeuraLens Database Configuration - Neon PostgreSQL
Cloud-native serverless PostgreSQL database integration
"""

import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool

from app.core.config import settings


# Database URL configuration
def get_database_url(async_mode: bool = True) -> str:
    """
    Get the database URL based on environment.
    Uses Neon PostgreSQL in production/development, SQLite for testing.
    """
    db_url = os.getenv("DATABASE_URL", settings.DATABASE_URL)
    
    if "postgresql" in db_url or "postgres" in db_url:
        # For Neon PostgreSQL
        if async_mode:
            # Convert to asyncpg driver format
            if db_url.startswith("postgresql://"):
                return db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif db_url.startswith("postgres://"):
                return db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        else:
            # Sync format with psycopg2
            if db_url.startswith("postgres://"):
                return db_url.replace("postgres://", "postgresql://", 1)
        return db_url
    else:
        # SQLite fallback for testing
        if async_mode:
            return db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        return db_url


# SQLAlchemy Base for ORM models
Base = declarative_base()

# Naming convention for constraints (important for migrations)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
Base.metadata = metadata


# Async engine and session factory
_async_engine = None
_async_session_factory = None

# Sync engine and session factory (for non-async contexts)
_sync_engine = None
_sync_session_factory = None


def get_async_engine():
    """Get or create the async database engine."""
    global _async_engine
    if _async_engine is None:
        db_url = get_database_url(async_mode=True)
        
        # Use NullPool for Neon serverless to avoid connection issues
        pool_class = NullPool if "neon" in db_url.lower() or "postgres" in db_url.lower() else None
        
        engine_kwargs = {
            "echo": settings.DATABASE_ECHO,
            "future": True,
        }
        
        if pool_class:
            engine_kwargs["poolclass"] = pool_class
        
        _async_engine = create_async_engine(db_url, **engine_kwargs)
    
    return _async_engine


def get_sync_engine():
    """Get or create the sync database engine."""
    global _sync_engine
    if _sync_engine is None:
        db_url = get_database_url(async_mode=False)
        
        _sync_engine = create_engine(
            db_url,
            echo=settings.DATABASE_ECHO,
            future=True,
        )
    
    return _sync_engine


def get_async_session_factory():
    """Get or create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_factory


def get_sync_session_factory():
    """Get or create the sync session factory."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(),
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _sync_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.
    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db_sync() -> Session:
    """
    Get a synchronous database session.
    Remember to close the session after use.
    """
    session_factory = get_sync_session_factory()
    return session_factory()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for async database sessions.
    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize the database by creating all tables.
    Called during application startup.
    """
    try:
        # Import all models to register them with Base.metadata
        from app.models import (
            User, Assessment, AssessmentResult,
            ValidationStudy, ValidationResult
        )
        
        # Try to import retinal models if they exist
        try:
            from app.pipelines.retinal.models import (
                RetinalAssessment, RetinalAuditLog
            )
        except ImportError:
            print("⚠️  Retinal models not found, skipping...")
        
        engine = get_async_engine()
        
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ Database tables created successfully")
        
    except Exception as e:
        print(f"⚠️  Database initialization failed: {e}")
        print("🔄 Continuing without database initialization...")


async def close_db():
    """
    Close database connections.
    Called during application shutdown.
    """
    global _async_engine, _sync_engine
    
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
    
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
    
    print("✅ Database connections closed")


async def health_check() -> dict:
    """
    Check database connectivity for health endpoint.
    """
    try:
        async with get_db_context() as db:
            result = await db.execute(text("SELECT 1"))
            result.scalar()
        return {
            "status": "healthy",
            "database": "neon_postgresql",
            "connected": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "neon_postgresql",
            "connected": False,
            "error": str(e)
        }


class DatabaseManager:
    """
    Utility class for database management operations.
    Used for health checks and administrative tasks.
    """
    
    @staticmethod
    def health_check() -> dict:
        """Synchronous health check for the database."""
        try:
            session = get_db_sync()
            result = session.execute(text("SELECT 1"))
            result.scalar()
            session.close()
            return {
                "status": "healthy",
                "database": "neon_postgresql",
                "connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "neon_postgresql", 
                "connected": False,
                "error": str(e)
            }
    
    @staticmethod
    async def get_table_stats() -> dict:
        """Get statistics about database tables."""
        try:
            async with get_db_context() as db:
                # Get table row counts
                tables = [
                    "users", "assessments", "assessment_results",
                    "validation_studies", "validation_results",
                    "retinal_assessments", "retinal_audit_logs"
                ]
                
                stats = {}
                for table in tables:
                    try:
                        result = await db.execute(
                            text(f"SELECT COUNT(*) FROM {table}")
                        )
                        stats[table] = result.scalar()
                    except Exception:
                        stats[table] = 0
                
                return {"status": "success", "tables": stats}
        except Exception as e:
            return {"status": "error", "error": str(e)}
