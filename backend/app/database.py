"""
MediLens Database Setup
Async SQLAlchemy with connection pooling for performance
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from .config import settings

# Fix URL for async
url = settings.DATABASE_URL
if "sqlite://" in url and "aiosqlite" not in url:
    url = url.replace("sqlite://", "sqlite+aiosqlite://")

# Use NullPool for SQLite (no connection pooling needed)
# Use AsyncAdaptedQueuePool for PostgreSQL/production
is_sqlite = "sqlite" in url
pool_class = NullPool if is_sqlite else AsyncAdaptedQueuePool

# Engine with connection pooling configuration
engine_kwargs = {
    "echo": settings.db_echo,
}

if not is_sqlite:
    # Add pool configuration for PostgreSQL
    engine_kwargs.update({
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_recycle": settings.DB_POOL_RECYCLE,
        "pool_pre_ping": settings.DB_POOL_PRE_PING,
    })
else:
    # SQLite doesn't support connection pooling
    engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(url, **engine_kwargs)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


async def get_db():
    """Get database session with automatic cleanup."""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

