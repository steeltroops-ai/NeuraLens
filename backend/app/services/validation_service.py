"""
Validation service providing CRUD operations for validation studies and metrics
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import uuid
import logging

from app.models.validation import (
    ValidationStudy, ValidationResult, PerformanceMetric, SystemHealth
)
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class ValidationService:
    """Service for managing validation studies and performance metrics"""
    
    def __init__(self):
        self.study_db = DatabaseService(ValidationStudy)
        self.result_db = DatabaseService(ValidationResult)
        self.metric_db = DatabaseService(PerformanceMetric)
        self.health_db = DatabaseService(SystemHealth)
    
    def create_validation_study(
        self, 
        db: Session,
        name: str,
        description: str,
        principal_investigator: str,
        institution: str,
        **kwargs
    ) -> ValidationStudy:
        """Create a new validation study"""
        try:
            study_data = {
                "study_id": str(uuid.uuid4()),
                "name": name,
                "description": description,
                "principal_investigator": principal_investigator,
                "institution": institution,
                "created_at": datetime.utcnow(),
                **kwargs
            }
            
            study = self.study_db.create(db, obj_in=study_data)
            logger.info(f"Created validation study: {study.study_id}")
            return study
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating validation study: {str(e)}")
            raise
    
    def get_validation_study(self, db: Session, study_id: str) -> Optional[ValidationStudy]:
        """Get validation study by ID"""
        try:
            return db.query(ValidationStudy).filter(
                ValidationStudy.study_id == study_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting validation study {study_id}: {str(e)}")
            raise
    
    def add_validation_result(
        self, 
        db: Session,
        study_id: int,
        participant_id: str,
        modality: str,
        predicted_risk: float,
        predicted_category: str,
        confidence: float,
        actual_diagnosis: str,
        actual_risk_category: str,
        **kwargs
    ) -> ValidationResult:
        """Add a validation result to a study"""
        try:
            result_data = {
                "study_id": study_id,
                "participant_id": participant_id,
                "modality": modality,
                "predicted_risk": predicted_risk,
                "predicted_category": predicted_category,
                "confidence": confidence,
                "actual_diagnosis": actual_diagnosis,
                "actual_risk_category": actual_risk_category,
                "correct_prediction": predicted_category == actual_risk_category,
                "prediction_error": abs(predicted_risk - kwargs.get("actual_risk_score", 0.5)),
                "assessment_date": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                **kwargs
            }
            
            result = self.result_db.create(db, obj_in=result_data)
            logger.info(f"Added validation result for participant: {participant_id}")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Error adding validation result: {str(e)}")
            raise
    
    def get_study_results(
        self, 
        db: Session, 
        study_id: int,
        modality: Optional[str] = None
    ) -> List[ValidationResult]:
        """Get all results for a validation study"""
        try:
            filters = {"study_id": study_id}
            if modality:
                filters["modality"] = modality
                
            return self.result_db.get_multi(
                db,
                filters=filters,
                order_by="assessment_date"
            )
        except SQLAlchemyError as e:
            logger.error(f"Error getting study results: {str(e)}")
            raise
    
    def calculate_study_metrics(
        self, 
        db: Session, 
        study_id: int,
        modality: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate performance metrics for a validation study"""
        try:
            results = self.get_study_results(db, study_id, modality)
            
            if not results:
                return {"status": "no_data", "message": "No validation results found"}
            
            # Calculate basic metrics
            total_predictions = len(results)
            correct_predictions = sum(1 for r in results if r.correct_prediction)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Calculate by category
            categories = {}
            for result in results:
                cat = result.actual_risk_category
                if cat not in categories:
                    categories[cat] = {"total": 0, "correct": 0}
                categories[cat]["total"] += 1
                if result.correct_prediction:
                    categories[cat]["correct"] += 1
            
            # Calculate category-specific metrics
            category_metrics = {}
            for cat, data in categories.items():
                category_metrics[cat] = {
                    "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                    "count": data["total"]
                }
            
            # Calculate confidence statistics
            confidences = [r.confidence for r in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Calculate prediction errors
            errors = [r.prediction_error for r in results if r.prediction_error is not None]
            avg_error = sum(errors) / len(errors) if errors else 0
            
            return {
                "study_id": study_id,
                "modality": modality or "all",
                "total_predictions": total_predictions,
                "accuracy": accuracy,
                "average_confidence": avg_confidence,
                "average_prediction_error": avg_error,
                "category_metrics": category_metrics,
                "calculated_at": datetime.utcnow().isoformat()
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error calculating study metrics: {str(e)}")
            raise
    
    def record_performance_metric(
        self, 
        db: Session,
        metric_name: str,
        value: float,
        component: str,
        modality: Optional[str] = None,
        unit: Optional[str] = None,
        **kwargs
    ) -> PerformanceMetric:
        """Record a performance metric"""
        try:
            metric_data = {
                "metric_name": metric_name,
                "value": value,
                "component": component,
                "modality": modality,
                "unit": unit,
                "timestamp": datetime.utcnow(),
                **kwargs
            }
            
            metric = self.metric_db.create(db, obj_in=metric_data)
            logger.info(f"Recorded performance metric: {metric_name} = {value}")
            return metric
            
        except SQLAlchemyError as e:
            logger.error(f"Error recording performance metric: {str(e)}")
            raise
    
    def get_performance_metrics(
        self, 
        db: Session,
        component: Optional[str] = None,
        modality: Optional[str] = None,
        metric_name: Optional[str] = None,
        hours_back: int = 24
    ) -> List[PerformanceMetric]:
        """Get performance metrics with optional filtering"""
        try:
            filters = {}
            if component:
                filters["component"] = component
            if modality:
                filters["modality"] = modality
            if metric_name:
                filters["metric_name"] = metric_name
            
            # Get recent metrics
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            return self.metric_db.get_multi(
                db,
                filters=filters,
                order_by="timestamp",
                order_desc=True,
                limit=1000
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            raise
    
    def record_system_health(
        self, 
        db: Session,
        component: str,
        status: str,
        response_time: Optional[float] = None,
        **kwargs
    ) -> SystemHealth:
        """Record system health status"""
        try:
            health_data = {
                "component": component,
                "status": status,
                "response_time": response_time,
                "timestamp": datetime.utcnow(),
                **kwargs
            }
            
            health = self.health_db.create(db, obj_in=health_data)
            logger.info(f"Recorded system health: {component} = {status}")
            return health
            
        except SQLAlchemyError as e:
            logger.error(f"Error recording system health: {str(e)}")
            raise
    
    def get_system_health_status(self, db: Session) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            # Get latest health records for each component
            from sqlalchemy import func
            
            latest_health = db.query(
                SystemHealth.component,
                func.max(SystemHealth.timestamp).label('latest_timestamp')
            ).group_by(SystemHealth.component).subquery()
            
            current_health = db.query(SystemHealth).join(
                latest_health,
                (SystemHealth.component == latest_health.c.component) &
                (SystemHealth.timestamp == latest_health.c.latest_timestamp)
            ).all()
            
            health_status = {}
            overall_status = "healthy"
            
            for health in current_health:
                health_status[health.component] = {
                    "status": health.status,
                    "response_time": health.response_time,
                    "timestamp": health.timestamp.isoformat(),
                    "error_count": health.error_count,
                    "last_error": health.last_error
                }
                
                # Determine overall status
                if health.status in ["critical", "down"]:
                    overall_status = "critical"
                elif health.status == "warning" and overall_status == "healthy":
                    overall_status = "warning"
            
            return {
                "overall_status": overall_status,
                "components": health_status,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting system health status: {str(e)}")
            raise


# Global service instance
validation_service = ValidationService()
