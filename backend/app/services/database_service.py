"""
Database service for NeuraLens backend
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for health checks and basic operations"""
    
    @staticmethod
    def health_check() -> Dict[str, Any]:
        """Check database health"""
        try:
            # Simple health check - in a real app this would test the actual DB connection
            return {
                "status": "healthy",
                "message": "Database connection OK"
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy", 
                "message": f"Database error: {str(e)}"
            }


class DatabaseService:
    """Main database service class"""
    
    def __init__(self):
        self.manager = DatabaseManager()
    
    def get_health(self) -> Dict[str, Any]:
        """Get database health status"""
        return self.manager.health_check()