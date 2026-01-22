"""
Database package for MediLens
Neon Postgres with SQLAlchemy async ORM
"""

from .neon_connection import (
    db,
    get_db,
    lifespan_manager,
    Base,
    NeonDatabase
)

# Import all models to register with Base
from .models import *  # noqa

__all__ = [
    "db",
    "get_db",
    "lifespan_manager",
    "Base",
    "NeonDatabase",
]
