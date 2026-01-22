"""
Alembic Migration Environment
Configure async SQLAlchemy migrations for Neon Postgres
"""

import asyncio
import os
import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load .env file manually
env_file = backend_dir / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove surrounding quotes if present
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value

# Import models to ensure they're registered
from app.database import Base
from app.database.models import *  # noqa

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata

# Get database URL from environment
DATABASE_URL = os.getenv("NEON_DATABASE_URL")
if DATABASE_URL:
    # Ensure async driver
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # Remove sslmode and channel_binding query params which cause issues with asyncpg
    if "?" in DATABASE_URL:
        base_url, query = DATABASE_URL.split("?", 1)
        params = query.split("&")
        # specific list of unsupported params for asyncpg
        unsupported = ["sslmode", "channel_binding"]
        params = [p for p in params if not any(p.startswith(k + "=") for k in unsupported)]
        if params:
            DATABASE_URL = f"{base_url}?{'&'.join(params)}"
        else:
            DATABASE_URL = base_url

    config.set_main_option("sqlalchemy.url", DATABASE_URL)
else:
    raise ValueError("NEON_DATABASE_URL not found in environment or .env file")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine"""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"ssl": "require"},
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
