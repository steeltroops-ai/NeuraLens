"""
User service providing CRUD operations for users and profiles
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import uuid
import logging

from app.models.user import User, UserProfile, AssessmentHistory
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing users and profiles"""
    
    def __init__(self):
        self.user_db = DatabaseService(User)
        self.profile_db = DatabaseService(UserProfile)
        self.history_db = DatabaseService(AssessmentHistory)
    
    def create_user(
        self, 
        db: Session,
        email: Optional[str] = None,
        username: Optional[str] = None,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        **kwargs
    ) -> User:
        """Create a new user"""
        try:
            user_data = {
                "id": str(uuid.uuid4()),
                "email": email,
                "username": username,
                "age": age,
                "sex": sex,
                "created_at": datetime.utcnow(),
                "consent_given": kwargs.get("consent_given", False),
                **kwargs
            }
            
            user = self.user_db.create(db, obj_in=user_data)
            logger.info(f"Created user: {user.id}")
            return user
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    def get_user(self, db: Session, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            return self.user_db.get(db, user_id)
        except SQLAlchemyError as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            raise
    
    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            return db.query(User).filter(User.email == email).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            raise
    
    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            return db.query(User).filter(User.username == username).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by username {username}: {str(e)}")
            raise
    
    def update_user(
        self, 
        db: Session, 
        user_id: str, 
        update_data: Dict[str, Any]
    ) -> Optional[User]:
        """Update user information"""
        try:
            user = self.get_user(db, user_id)
            if user:
                return self.user_db.update(db, db_obj=user, obj_in=update_data)
            return None
        except SQLAlchemyError as e:
            logger.error(f"Error updating user {user_id}: {str(e)}")
            raise
    
    def update_last_assessment(
        self, 
        db: Session, 
        user_id: str
    ) -> Optional[User]:
        """Update user's last assessment timestamp"""
        try:
            return self.update_user(
                db, 
                user_id, 
                {"last_assessment": datetime.utcnow()}
            )
        except SQLAlchemyError as e:
            logger.error(f"Error updating last assessment for user {user_id}: {str(e)}")
            raise
    
    def create_user_profile(
        self, 
        db: Session,
        user_id: str,
        profile_data: Dict[str, Any]
    ) -> UserProfile:
        """Create extended user profile"""
        try:
            profile_data.update({
                "user_id": user_id,
                "created_at": datetime.utcnow()
            })
            
            profile = self.profile_db.create(db, obj_in=profile_data)
            logger.info(f"Created profile for user: {user_id}")
            return profile
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating user profile: {str(e)}")
            raise
    
    def get_user_profile(self, db: Session, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        try:
            return db.query(UserProfile).filter(
                UserProfile.user_id == user_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user profile for {user_id}: {str(e)}")
            raise
    
    def update_user_profile(
        self, 
        db: Session, 
        user_id: str, 
        profile_data: Dict[str, Any]
    ) -> Optional[UserProfile]:
        """Update user profile"""
        try:
            profile = self.get_user_profile(db, user_id)
            if profile:
                return self.profile_db.update(db, db_obj=profile, obj_in=profile_data)
            return None
        except SQLAlchemyError as e:
            logger.error(f"Error updating user profile for {user_id}: {str(e)}")
            raise
    
    def update_baseline_scores(
        self, 
        db: Session, 
        user_id: str,
        cognitive_score: Optional[float] = None,
        motor_score: Optional[float] = None,
        speech_score: Optional[float] = None,
        retinal_score: Optional[float] = None
    ) -> Optional[UserProfile]:
        """Update baseline assessment scores"""
        try:
            update_data = {}
            if cognitive_score is not None:
                update_data["baseline_cognitive_score"] = cognitive_score
            if motor_score is not None:
                update_data["baseline_motor_score"] = motor_score
            if speech_score is not None:
                update_data["baseline_speech_score"] = speech_score
            if retinal_score is not None:
                update_data["baseline_retinal_score"] = retinal_score
            
            if update_data:
                return self.update_user_profile(db, user_id, update_data)
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Error updating baseline scores for user {user_id}: {str(e)}")
            raise
    
    def add_assessment_history(
        self, 
        db: Session,
        user_id: str,
        assessment_id: int,
        nri_score: float,
        risk_category: str,
        score_change: Optional[float] = None,
        trend_direction: Optional[str] = None
    ) -> AssessmentHistory:
        """Add assessment to user history"""
        try:
            history_data = {
                "user_id": user_id,
                "assessment_id": assessment_id,
                "nri_score": nri_score,
                "overall_risk_category": risk_category,
                "score_change": score_change,
                "trend_direction": trend_direction,
                "assessment_date": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            history = self.history_db.create(db, obj_in=history_data)
            logger.info(f"Added assessment history for user: {user_id}")
            return history
            
        except SQLAlchemyError as e:
            logger.error(f"Error adding assessment history: {str(e)}")
            raise
    
    def get_user_assessment_history(
        self, 
        db: Session, 
        user_id: str,
        skip: int = 0,
        limit: int = 10
    ) -> List[AssessmentHistory]:
        """Get user's assessment history"""
        try:
            return self.history_db.get_multi(
                db,
                skip=skip,
                limit=limit,
                filters={"user_id": user_id},
                order_by="assessment_date",
                order_desc=True
            )
        except SQLAlchemyError as e:
            logger.error(f"Error getting assessment history for user {user_id}: {str(e)}")
            raise
    
    def get_user_trend_analysis(
        self, 
        db: Session, 
        user_id: str
    ) -> Dict[str, Any]:
        """Get user's assessment trend analysis"""
        try:
            history = self.get_user_assessment_history(db, user_id, limit=10)
            
            if not history:
                return {"status": "no_data", "message": "No assessment history found"}
            
            # Calculate trends
            scores = [h.nri_score for h in history]
            latest_score = scores[0] if scores else 0
            
            trend_analysis = {
                "latest_score": latest_score,
                "assessment_count": len(history),
                "score_trend": "stable",  # Default
                "average_score": sum(scores) / len(scores) if scores else 0,
                "score_range": {
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0
                }
            }
            
            # Determine trend direction
            if len(scores) >= 2:
                recent_change = scores[0] - scores[1]
                if recent_change > 5:
                    trend_analysis["score_trend"] = "improving"
                elif recent_change < -5:
                    trend_analysis["score_trend"] = "declining"
                else:
                    trend_analysis["score_trend"] = "stable"
            
            return trend_analysis
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting trend analysis for user {user_id}: {str(e)}")
            raise
    
    def get_users_summary(self, db: Session) -> Dict[str, Any]:
        """Get summary statistics for all users"""
        try:
            total_users = self.user_db.count(db)
            users_with_assessments = self.user_db.count(
                db, 
                filters={"last_assessment": "NOT NULL"}
            )
            
            return {
                "total_users": total_users,
                "users_with_assessments": users_with_assessments,
                "completion_rate": (users_with_assessments / total_users * 100) if total_users > 0 else 0
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting users summary: {str(e)}")
            raise


# Global service instance
user_service = UserService()
