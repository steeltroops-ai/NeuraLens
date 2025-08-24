"""
Advanced Result Storage and Retrieval Service
Comprehensive assessment result management with caching and analytics
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from app.core.database import get_db
from app.models.assessment import Assessment, AssessmentResult, AssessmentSession
from app.schemas.assessment import (
    AssessmentResultCreate,
    AssessmentResultResponse,
    AssessmentHistoryResponse,
    AssessmentAnalyticsResponse
)
from app.core.security import encrypt_data, decrypt_data
from app.core.cache import cache_manager


class ResultStorageService:
    """Advanced result storage and retrieval service"""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.encryption_enabled = True
    
    async def store_assessment_result(
        self,
        session_id: str,
        result_data: Dict[str, Any],
        user_id: Optional[str] = None,
        db: Session = None
    ) -> AssessmentResultResponse:
        """Store complete assessment result with metadata"""
        
        if db is None:
            db = next(get_db())
        
        try:
            # Create assessment session if not exists
            session = db.query(AssessmentSession).filter(
                AssessmentSession.session_id == session_id
            ).first()
            
            if not session:
                session = AssessmentSession(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.utcnow(),
                    status='completed'
                )
                db.add(session)
                db.flush()
            
            # Encrypt sensitive data if enabled
            encrypted_data = result_data
            if self.encryption_enabled:
                encrypted_data = await self._encrypt_result_data(result_data)
            
            # Create assessment result
            result = AssessmentResult(
                session_id=session_id,
                result_data=json.dumps(encrypted_data),
                risk_score=result_data.get('nri_result', {}).get('nri_score', 0.0),
                risk_category=result_data.get('overall_risk_category', 'moderate'),
                confidence_score=result_data.get('nri_result', {}).get('confidence', 0.0),
                processing_time=result_data.get('total_processing_time', 0),
                modalities_used=self._extract_modalities(result_data),
                quality_metrics=self._calculate_quality_metrics(result_data),
                created_at=datetime.utcnow()
            )
            
            db.add(result)
            db.commit()
            
            # Cache result for quick access
            await self._cache_result(session_id, result_data)
            
            # Update analytics
            await self._update_analytics(result_data, db)
            
            return AssessmentResultResponse(
                id=result.id,
                session_id=session_id,
                result_data=result_data,
                risk_score=result.risk_score,
                risk_category=result.risk_category,
                confidence_score=result.confidence_score,
                processing_time=result.processing_time,
                created_at=result.created_at,
                modalities_used=result.modalities_used,
                quality_metrics=result.quality_metrics
            )
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to store assessment result: {str(e)}")
    
    async def get_assessment_result(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        db: Session = None
    ) -> Optional[AssessmentResultResponse]:
        """Retrieve assessment result by session ID"""
        
        # Try cache first
        cached_result = await self._get_cached_result(session_id)
        if cached_result:
            return cached_result
        
        if db is None:
            db = next(get_db())
        
        try:
            # Query database
            query = db.query(AssessmentResult).filter(
                AssessmentResult.session_id == session_id
            )
            
            if user_id:
                query = query.join(AssessmentSession).filter(
                    AssessmentSession.user_id == user_id
                )
            
            result = query.first()
            
            if not result:
                return None
            
            # Decrypt data if needed
            result_data = json.loads(result.result_data)
            if self.encryption_enabled:
                result_data = await self._decrypt_result_data(result_data)
            
            response = AssessmentResultResponse(
                id=result.id,
                session_id=result.session_id,
                result_data=result_data,
                risk_score=result.risk_score,
                risk_category=result.risk_category,
                confidence_score=result.confidence_score,
                processing_time=result.processing_time,
                created_at=result.created_at,
                modalities_used=result.modalities_used,
                quality_metrics=result.quality_metrics
            )
            
            # Cache for future requests
            await self._cache_result(session_id, result_data)
            
            return response
            
        except Exception as e:
            raise Exception(f"Failed to retrieve assessment result: {str(e)}")
    
    async def get_assessment_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        risk_category: Optional[str] = None,
        db: Session = None
    ) -> AssessmentHistoryResponse:
        """Get assessment history for a user"""
        
        if db is None:
            db = next(get_db())
        
        try:
            # Build query
            query = db.query(AssessmentResult).join(AssessmentSession).filter(
                AssessmentSession.user_id == user_id
            )
            
            # Apply filters
            if date_from:
                query = query.filter(AssessmentResult.created_at >= date_from)
            if date_to:
                query = query.filter(AssessmentResult.created_at <= date_to)
            if risk_category:
                query = query.filter(AssessmentResult.risk_category == risk_category)
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and ordering
            results = query.order_by(desc(AssessmentResult.created_at)).offset(offset).limit(limit).all()
            
            # Convert to response format
            assessment_results = []
            for result in results:
                result_data = json.loads(result.result_data)
                if self.encryption_enabled:
                    result_data = await self._decrypt_result_data(result_data)
                
                assessment_results.append(AssessmentResultResponse(
                    id=result.id,
                    session_id=result.session_id,
                    result_data=result_data,
                    risk_score=result.risk_score,
                    risk_category=result.risk_category,
                    confidence_score=result.confidence_score,
                    processing_time=result.processing_time,
                    created_at=result.created_at,
                    modalities_used=result.modalities_used,
                    quality_metrics=result.quality_metrics
                ))
            
            return AssessmentHistoryResponse(
                results=assessment_results,
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=offset + len(results) < total_count
            )
            
        except Exception as e:
            raise Exception(f"Failed to retrieve assessment history: {str(e)}")
    
    async def get_assessment_analytics(
        self,
        user_id: str,
        days: int = 30,
        db: Session = None
    ) -> AssessmentAnalyticsResponse:
        """Get assessment analytics and trends"""
        
        if db is None:
            db = next(get_db())
        
        try:
            date_from = datetime.utcnow() - timedelta(days=days)
            
            # Get assessments in date range
            results = db.query(AssessmentResult).join(AssessmentSession).filter(
                and_(
                    AssessmentSession.user_id == user_id,
                    AssessmentResult.created_at >= date_from
                )
            ).order_by(AssessmentResult.created_at).all()
            
            if not results:
                return AssessmentAnalyticsResponse(
                    total_assessments=0,
                    risk_distribution={},
                    trend_data=[],
                    average_scores={},
                    modality_usage={},
                    quality_trends={}
                )
            
            # Calculate analytics
            total_assessments = len(results)
            
            # Risk distribution
            risk_distribution = {}
            for result in results:
                category = result.risk_category
                risk_distribution[category] = risk_distribution.get(category, 0) + 1
            
            # Trend data (weekly averages)
            trend_data = self._calculate_trend_data(results, days)
            
            # Average scores by modality
            average_scores = self._calculate_average_scores(results)
            
            # Modality usage statistics
            modality_usage = self._calculate_modality_usage(results)
            
            # Quality trends
            quality_trends = self._calculate_quality_trends(results)
            
            return AssessmentAnalyticsResponse(
                total_assessments=total_assessments,
                risk_distribution=risk_distribution,
                trend_data=trend_data,
                average_scores=average_scores,
                modality_usage=modality_usage,
                quality_trends=quality_trends
            )
            
        except Exception as e:
            raise Exception(f"Failed to calculate assessment analytics: {str(e)}")
    
    async def export_assessment_data(
        self,
        user_id: str,
        format: str = 'json',
        session_ids: Optional[List[str]] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """Export assessment data in specified format"""
        
        if db is None:
            db = next(get_db())
        
        try:
            # Build query
            query = db.query(AssessmentResult).join(AssessmentSession).filter(
                AssessmentSession.user_id == user_id
            )
            
            if session_ids:
                query = query.filter(AssessmentResult.session_id.in_(session_ids))
            
            results = query.order_by(desc(AssessmentResult.created_at)).all()
            
            # Prepare export data
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'total_assessments': len(results),
                'assessments': []
            }
            
            for result in results:
                result_data = json.loads(result.result_data)
                if self.encryption_enabled:
                    result_data = await self._decrypt_result_data(result_data)
                
                export_data['assessments'].append({
                    'session_id': result.session_id,
                    'created_at': result.created_at.isoformat(),
                    'risk_score': result.risk_score,
                    'risk_category': result.risk_category,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time,
                    'modalities_used': result.modalities_used,
                    'quality_metrics': result.quality_metrics,
                    'detailed_results': result_data
                })
            
            return export_data
            
        except Exception as e:
            raise Exception(f"Failed to export assessment data: {str(e)}")
    
    async def delete_assessment_result(
        self,
        session_id: str,
        user_id: str,
        db: Session = None
    ) -> bool:
        """Delete assessment result (HIPAA compliance)"""
        
        if db is None:
            db = next(get_db())
        
        try:
            # Find and delete result
            result = db.query(AssessmentResult).join(AssessmentSession).filter(
                and_(
                    AssessmentResult.session_id == session_id,
                    AssessmentSession.user_id == user_id
                )
            ).first()
            
            if result:
                db.delete(result)
                db.commit()
                
                # Remove from cache
                await self._remove_cached_result(session_id)
                
                return True
            
            return False
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to delete assessment result: {str(e)}")
    
    # Private helper methods
    async def _encrypt_result_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive result data"""
        # Implementation would use proper encryption
        return data  # Placeholder
    
    async def _decrypt_result_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt result data"""
        # Implementation would use proper decryption
        return data  # Placeholder
    
    def _extract_modalities(self, result_data: Dict[str, Any]) -> List[str]:
        """Extract used modalities from result data"""
        modalities = []
        if result_data.get('speech_result'):
            modalities.append('speech')
        if result_data.get('retinal_result'):
            modalities.append('retinal')
        if result_data.get('motor_result'):
            modalities.append('motor')
        if result_data.get('cognitive_result'):
            modalities.append('cognitive')
        return modalities
    
    def _calculate_quality_metrics(self, result_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics from result data"""
        metrics = {}
        
        if result_data.get('speech_result'):
            metrics['speech_quality'] = result_data['speech_result'].get('quality_score', 0.0)
        if result_data.get('retinal_result'):
            metrics['retinal_quality'] = result_data['retinal_result'].get('quality_score', 0.0)
        
        return metrics
    
    async def _cache_result(self, session_id: str, result_data: Dict[str, Any]):
        """Cache result for quick access"""
        cache_key = f"assessment_result:{session_id}"
        await cache_manager.set(cache_key, result_data, ttl=self.cache_ttl)
    
    async def _get_cached_result(self, session_id: str) -> Optional[AssessmentResultResponse]:
        """Get cached result"""
        cache_key = f"assessment_result:{session_id}"
        cached_data = await cache_manager.get(cache_key)
        return cached_data
    
    async def _remove_cached_result(self, session_id: str):
        """Remove result from cache"""
        cache_key = f"assessment_result:{session_id}"
        await cache_manager.delete(cache_key)
    
    async def _update_analytics(self, result_data: Dict[str, Any], db: Session):
        """Update analytics data"""
        # Implementation would update analytics tables
        pass
    
    def _calculate_trend_data(self, results: List[AssessmentResult], days: int) -> List[Dict[str, Any]]:
        """Calculate trend data for analytics"""
        # Implementation would calculate weekly/daily trends
        return []
    
    def _calculate_average_scores(self, results: List[AssessmentResult]) -> Dict[str, float]:
        """Calculate average scores by modality"""
        # Implementation would calculate averages
        return {}
    
    def _calculate_modality_usage(self, results: List[AssessmentResult]) -> Dict[str, int]:
        """Calculate modality usage statistics"""
        # Implementation would calculate usage stats
        return {}
    
    def _calculate_quality_trends(self, results: List[AssessmentResult]) -> Dict[str, List[float]]:
        """Calculate quality trends over time"""
        # Implementation would calculate quality trends
        return {}


# Global instance
result_storage_service = ResultStorageService()
