"""
Neon Postgres Connection Manager
Serverless-optimized connection pooling for MediLens
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.orm import declarative_base
from contextlib import asynccontextmanager

# Base for all models
Base = declarative_base()


class NeonDatabase:
    """Neon Postgres database manager with serverless optimizations"""
    
    def __init__(self):
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker | None = None
        self._initialized = False
    
    def initialize(
        self,
        database_url: str | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        echo: bool = False
    ):
        """Initialize database connection with Neon-optimized settings"""
        
        # Get database URL from environment or parameter
        db_url = database_url or os.getenv("NEON_DATABASE_URL")
        
        if not db_url:
            raise ValueError("NEON_DATABASE_URL environment variable not set")
        
        # Ensure async driver
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif not db_url.startswith("postgresql+asyncpg://"):
            raise ValueError("Database URL must use asyncpg driver for async support")
            
        # Clean unsupported query parameters for asyncpg
        if "?" in db_url:
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            # Remove sslmode and other unsupported params
            u = urlparse(db_url)
            qs = parse_qs(u.query)
            unsupported = ["sslmode", "channel_binding"]
            new_qs = {k: v for k, v in qs.items() if k not in unsupported}
            
            # Rebuild URL if changed
            if len(new_qs) != len(qs):
                query = urlencode(new_qs, doseq=True)
                db_url = urlunparse((u.scheme, u.netloc, u.path, u.params, query, u.fragment))
        
        # Determine if we're in serverless mode (Vercel, AWS Lambda, etc.)
        is_serverless = os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        
        # Connection arguments optimized for Neon
        connect_args = {
            "server_settings": {
                "application_name": "medilens-backend",
                "jit": "off",  # Disable JIT for consistent performance
            },
            "command_timeout": 30,  # 30 second query timeout
            "timeout": 10,  # 10 second connection timeout
            "ssl": "require",
        }
        
        # Engine configuration
        engine_kwargs = {
            "echo": echo,
            "pool_pre_ping": True,  # Verify connections before use
            "pool_recycle": 3600,  # Recycle connections after 1 hour
            "connect_args": connect_args,
        }
        
        # Use NullPool for serverless, regular pool for long-running servers
        if is_serverless:
            engine_kwargs["poolclass"] = NullPool
        else:
            engine_kwargs["pool_size"] = pool_size
            engine_kwargs["max_overflow"] = max_overflow
            engine_kwargs["pool_timeout"] = pool_timeout
        
        # Create async engine
        self.engine = create_async_engine(db_url, **engine_kwargs)
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        
        self._initialized = True
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def session_scope(self):
        """Context manager for database session with automatic commit/rollback"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_all_tables(self):
        """Create all tables (for development/testing only)"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_all_tables(self):
        """Drop all tables (for testing only)"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False


# Global database instance
db = NeonDatabase()


# FastAPI dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI endpoints"""
    async for session in db.get_session():
        yield session


# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan_manager(app):
    """Manage database lifecycle with FastAPI app"""
    # Startup
    db.initialize(
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        echo=os.getenv("DB_ECHO", "False").lower() == "true"
    )
    
    yield
    
    # Shutdown
    await db.close()
