"""
Standardized API Response Wrapper
Ensures consistent response format across all endpoints
"""

from typing import Any, Dict, Optional, Union
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import uuid
import time


class ApiMetadata(BaseModel):
    """API response metadata"""
    timestamp: str
    request_id: str
    processing_time: Optional[float] = None
    api_version: str = "1.0.0"


class ApiError(BaseModel):
    """Standardized error format"""
    code: str
    message: str
    details: Optional[Any] = None


class StandardResponse(BaseModel):
    """Standardized API response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[ApiError] = None
    metadata: ApiMetadata


class ResponseBuilder:
    """Builder class for creating standardized responses"""
    
    def __init__(self):
        self.request_id = str(uuid.uuid4())
        self.start_time = time.perf_counter()
    
    def success(
        self, 
        data: Any, 
        processing_time: Optional[float] = None
    ) -> StandardResponse:
        """Create successful response"""
        
        if processing_time is None:
            processing_time = time.perf_counter() - self.start_time
        
        return StandardResponse(
            success=True,
            data=data,
            metadata=ApiMetadata(
                timestamp=datetime.utcnow().isoformat(),
                request_id=self.request_id,
                processing_time=processing_time
            )
        )
    
    def error(
        self, 
        code: str, 
        message: str, 
        details: Optional[Any] = None,
        status_code: int = 400
    ) -> JSONResponse:
        """Create error response"""
        
        processing_time = time.perf_counter() - self.start_time
        
        response = StandardResponse(
            success=False,
            error=ApiError(
                code=code,
                message=message,
                details=details
            ),
            metadata=ApiMetadata(
                timestamp=datetime.utcnow().isoformat(),
                request_id=self.request_id,
                processing_time=processing_time
            )
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response.dict()
        )
    
    def validation_error(
        self, 
        message: str, 
        details: Optional[Any] = None
    ) -> JSONResponse:
        """Create validation error response (422)"""
        return self.error(
            code="VALIDATION_ERROR",
            message=message,
            details=details,
            status_code=422
        )
    
    def not_found(
        self, 
        resource: str = "Resource"
    ) -> JSONResponse:
        """Create not found error response (404)"""
        return self.error(
            code="NOT_FOUND",
            message=f"{resource} not found",
            status_code=404
        )
    
    def internal_error(
        self, 
        message: str = "Internal server error", 
        details: Optional[Any] = None
    ) -> JSONResponse:
        """Create internal server error response (500)"""
        return self.error(
            code="INTERNAL_ERROR",
            message=message,
            details=details,
            status_code=500
        )
    
    def unauthorized(
        self, 
        message: str = "Unauthorized access"
    ) -> JSONResponse:
        """Create unauthorized error response (401)"""
        return self.error(
            code="UNAUTHORIZED",
            message=message,
            status_code=401
        )
    
    def forbidden(
        self, 
        message: str = "Access forbidden"
    ) -> JSONResponse:
        """Create forbidden error response (403)"""
        return self.error(
            code="FORBIDDEN",
            message=message,
            status_code=403
        )
    
    def rate_limited(
        self, 
        message: str = "Rate limit exceeded"
    ) -> JSONResponse:
        """Create rate limit error response (429)"""
        return self.error(
            code="RATE_LIMITED",
            message=message,
            status_code=429
        )


# Utility functions for common response patterns
def success_response(data: Any, processing_time: Optional[float] = None) -> StandardResponse:
    """Quick success response"""
    builder = ResponseBuilder()
    return builder.success(data, processing_time)


def error_response(
    code: str, 
    message: str, 
    details: Optional[Any] = None, 
    status_code: int = 400
) -> JSONResponse:
    """Quick error response"""
    builder = ResponseBuilder()
    return builder.error(code, message, details, status_code)


# Exception handler for FastAPI
async def api_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Global exception handler for consistent error responses"""
    
    builder = ResponseBuilder()
    
    # Map HTTP status codes to error codes
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED", 
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE"
    }
    
    error_code = error_code_map.get(exc.status_code, "UNKNOWN_ERROR")
    
    return builder.error(
        code=error_code,
        message=exc.detail,
        status_code=exc.status_code
    )


# Health check response
class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    environment: str
    timestamp: str
    components: Optional[Dict[str, str]] = None


def health_check_response(
    status: str = "healthy",
    service: str = "NeuraLens API",
    version: str = "1.0.0",
    environment: str = "development",
    components: Optional[Dict[str, str]] = None
) -> HealthCheckResponse:
    """Create health check response"""
    
    return HealthCheckResponse(
        status=status,
        service=service,
        version=version,
        environment=environment,
        timestamp=datetime.utcnow().isoformat(),
        components=components or {}
    )


# API Status response
class ApiStatusResponse(BaseModel):
    """API status response model"""
    status: str
    version: str
    endpoints: Dict[str, str]
    features: Dict[str, bool]


def api_status_response(
    status: str = "operational",
    version: str = "1.0.0",
    endpoints: Optional[Dict[str, str]] = None,
    features: Optional[Dict[str, bool]] = None
) -> ApiStatusResponse:
    """Create API status response"""
    
    default_endpoints = {
        "speech": "operational",
        "retinal": "operational", 
        "motor": "operational",
        "cognitive": "operational",
        "nri": "operational",
        "validation": "operational"
    }
    
    default_features = {
        "real_time_analysis": True,
        "batch_processing": False,
        "multi_modal_fusion": True,
        "validation_metrics": True
    }
    
    return ApiStatusResponse(
        status=status,
        version=version,
        endpoints=endpoints or default_endpoints,
        features=features or default_features
    )


# Service info response
class ServiceInfoResponse(BaseModel):
    """Service information response model"""
    service: str
    version: str
    description: str
    capabilities: Optional[Dict[str, Any]] = None
    requirements: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


def service_info_response(
    service: str,
    description: str,
    capabilities: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    performance: Optional[Dict[str, Any]] = None
) -> ServiceInfoResponse:
    """Create service info response"""
    
    return ServiceInfoResponse(
        service=service,
        version="1.0.0",
        description=description,
        capabilities=capabilities,
        requirements=requirements,
        performance=performance
    )
