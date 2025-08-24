"""
Database initialization script with sample data
Creates sample users, assessments, and validation data for testing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import uuid

from app.core.database import SessionLocal, init_db
from app.services.user_service import user_service
from app.services.assessment_service import assessment_service
from app.services.validation_service import validation_service

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Database initialization with sample data"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def create_sample_users(self) -> List[str]:
        """Create sample users for testing"""
        sample_users = [
            {
                "email": "john.doe@example.com",
                "username": "john_doe",
                "age": 65,
                "sex": "male",
                "education_years": 16,
                "family_history": {
                    "parkinsons": False,
                    "alzheimers": True,
                    "stroke": False
                },
                "medical_history": {
                    "hypertension": True,
                    "diabetes": False,
                    "heart_disease": False
                },
                "lifestyle_factors": {
                    "exercise_frequency": "moderate",
                    "smoking": False,
                    "alcohol": "occasional"
                },
                "consent_given": True
            },
            {
                "email": "jane.smith@example.com", 
                "username": "jane_smith",
                "age": 58,
                "sex": "female",
                "education_years": 14,
                "family_history": {
                    "parkinsons": True,
                    "alzheimers": False,
                    "stroke": True
                },
                "medical_history": {
                    "hypertension": False,
                    "diabetes": True,
                    "heart_disease": False
                },
                "lifestyle_factors": {
                    "exercise_frequency": "high",
                    "smoking": False,
                    "alcohol": "none"
                },
                "consent_given": True
            },
            {
                "email": "robert.wilson@example.com",
                "username": "robert_wilson", 
                "age": 72,
                "sex": "male",
                "education_years": 12,
                "family_history": {
                    "parkinsons": False,
                    "alzheimers": False,
                    "stroke": False
                },
                "medical_history": {
                    "hypertension": True,
                    "diabetes": True,
                    "heart_disease": True
                },
                "lifestyle_factors": {
                    "exercise_frequency": "low",
                    "smoking": True,
                    "alcohol": "frequent"
                },
                "consent_given": True
            }
        ]
        
        created_users = []
        for user_data in sample_users:
            try:
                user = user_service.create_user(self.db, **user_data)
                created_users.append(user.id)
                
                # Create user profile
                profile_data = {
                    "ethnicity": random.choice(["Caucasian", "African American", "Hispanic", "Asian"]),
                    "occupation": random.choice(["Retired", "Teacher", "Engineer", "Healthcare Worker"]),
                    "handedness": random.choice(["right", "left"]),
                    "language_preference": "en",
                    "baseline_cognitive_score": random.uniform(0.7, 0.95),
                    "baseline_motor_score": random.uniform(0.6, 0.9),
                    "baseline_speech_score": random.uniform(0.75, 0.95),
                    "baseline_retinal_score": random.uniform(0.65, 0.9)
                }
                
                user_service.create_user_profile(self.db, user.id, profile_data)
                logger.info(f"Created sample user: {user.email}")
                
            except Exception as e:
                logger.error(f"Error creating sample user: {str(e)}")
        
        return created_users
    
    def create_sample_assessments(self, user_ids: List[str]) -> List[int]:
        """Create sample assessments for users"""
        created_assessments = []
        
        for user_id in user_ids:
            try:
                # Create 2-3 assessments per user
                num_assessments = random.randint(2, 3)
                
                for i in range(num_assessments):
                    # Create assessment with random date in the past
                    days_ago = random.randint(1, 90)
                    assessment_date = datetime.utcnow() - timedelta(days=days_ago)
                    
                    assessment = assessment_service.create_assessment(
                        self.db,
                        user_id=user_id,
                        modalities=["speech", "retinal", "motor", "cognitive"],
                        assessment_type="full"
                    )
                    
                    # Update assessment date
                    assessment.created_at = assessment_date
                    self.db.commit()
                    
                    created_assessments.append(assessment.id)
                    
                    # Update user's last assessment
                    user_service.update_last_assessment(self.db, user_id)
                    
                    logger.info(f"Created sample assessment: {assessment.session_id}")
                    
            except Exception as e:
                logger.error(f"Error creating sample assessments: {str(e)}")
        
        return created_assessments
    
    def create_sample_validation_study(self) -> int:
        """Create a sample validation study"""
        try:
            study = validation_service.create_validation_study(
                self.db,
                name="NeuraLens Multi-Modal Validation Study",
                description="Comprehensive validation of multi-modal neurological risk assessment platform",
                principal_investigator="Dr. Sarah Johnson",
                institution="University Medical Center",
                study_type="prospective",
                target_participants=500,
                actual_participants=342,
                start_date=datetime.utcnow() - timedelta(days=180),
                status="active",
                overall_accuracy=0.87,
                sensitivity=0.82,
                specificity=0.91,
                auc_score=0.89,
                f1_score=0.85
            )
            
            # Add sample validation results
            modalities = ["speech", "retinal", "motor", "cognitive"]
            risk_categories = ["low", "moderate", "high"]
            
            for i in range(50):  # Create 50 sample validation results
                participant_id = f"PARTICIPANT_{i+1:03d}"
                modality = random.choice(modalities)
                
                # Generate realistic predictions
                actual_category = random.choice(risk_categories)
                predicted_category = actual_category if random.random() > 0.15 else random.choice(risk_categories)
                
                risk_score_map = {"low": 0.2, "moderate": 0.5, "high": 0.8}
                predicted_risk = risk_score_map[predicted_category] + random.uniform(-0.15, 0.15)
                predicted_risk = max(0.0, min(1.0, predicted_risk))
                
                validation_service.add_validation_result(
                    self.db,
                    study_id=study.id,
                    participant_id=participant_id,
                    modality=modality,
                    predicted_risk=predicted_risk,
                    predicted_category=predicted_category,
                    confidence=random.uniform(0.7, 0.95),
                    actual_diagnosis=f"Clinical diagnosis for {participant_id}",
                    actual_risk_category=actual_category,
                    age=random.randint(45, 85),
                    sex=random.choice(["male", "female"]),
                    education_years=random.randint(8, 20),
                    processing_time=random.uniform(0.5, 3.0)
                )
            
            logger.info(f"Created validation study: {study.name}")
            return study.id
            
        except Exception as e:
            logger.error(f"Error creating validation study: {str(e)}")
            return None
    
    def create_sample_performance_metrics(self):
        """Create sample performance metrics"""
        try:
            components = ["frontend", "backend", "ml_model"]
            modalities = ["speech", "retinal", "motor", "cognitive"]
            metrics = [
                ("response_time", "ms", 50, 2000),
                ("accuracy", "percentage", 80, 95),
                ("throughput", "requests/sec", 10, 100),
                ("error_rate", "percentage", 0, 5)
            ]
            
            # Create metrics for the last 7 days
            for day in range(7):
                date = datetime.utcnow() - timedelta(days=day)
                
                for component in components:
                    for modality in modalities:
                        for metric_name, unit, min_val, max_val in metrics:
                            value = random.uniform(min_val, max_val)
                            
                            validation_service.record_performance_metric(
                                self.db,
                                metric_name=metric_name,
                                value=value,
                                component=component,
                                modality=modality,
                                unit=unit,
                                timestamp=date,
                                environment="development"
                            )
            
            logger.info("Created sample performance metrics")
            
        except Exception as e:
            logger.error(f"Error creating performance metrics: {str(e)}")
    
    def create_sample_system_health(self):
        """Create sample system health records"""
        try:
            components = [
                "database", "api_gateway", "speech_analyzer", 
                "retinal_analyzer", "motor_analyzer", "cognitive_analyzer",
                "nri_fusion", "validation_engine"
            ]
            
            for component in components:
                status = random.choice(["healthy", "healthy", "healthy", "warning"])  # Bias toward healthy
                response_time = random.uniform(10, 500)
                
                validation_service.record_system_health(
                    self.db,
                    component=component,
                    status=status,
                    response_time=response_time,
                    cpu_usage=random.uniform(10, 80),
                    memory_usage=random.uniform(20, 70),
                    disk_usage=random.uniform(15, 60),
                    error_count=random.randint(0, 3) if status == "warning" else 0
                )
            
            logger.info("Created sample system health records")
            
        except Exception as e:
            logger.error(f"Error creating system health records: {str(e)}")
    
    async def initialize_database(self):
        """Initialize database with all sample data"""
        try:
            logger.info("Starting database initialization...")
            
            # Initialize database tables
            await init_db()
            
            # Create sample data
            logger.info("Creating sample users...")
            user_ids = self.create_sample_users()
            
            logger.info("Creating sample assessments...")
            assessment_ids = self.create_sample_assessments(user_ids)
            
            logger.info("Creating sample validation study...")
            study_id = self.create_sample_validation_study()
            
            logger.info("Creating sample performance metrics...")
            self.create_sample_performance_metrics()
            
            logger.info("Creating sample system health records...")
            self.create_sample_system_health()
            
            logger.info("✅ Database initialization completed successfully!")
            
            return {
                "status": "success",
                "users_created": len(user_ids),
                "assessments_created": len(assessment_ids),
                "validation_study_created": study_id is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}")
            raise


async def main():
    """Main initialization function"""
    with DatabaseInitializer() as initializer:
        result = await initializer.initialize_database()
        print(f"Database initialization result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
