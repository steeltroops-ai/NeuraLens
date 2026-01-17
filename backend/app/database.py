"""
MediLens Database Setup
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from .config import settings

# Fix URL for async
url = settings.DATABASE_URL
if "sqlite://" in url and "aiosqlite" not in url:
    url = url.replace("sqlite://", "sqlite+aiosqlite://")

engine = create_async_engine(url, echo=settings.DEBUG)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


async def get_db():
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
