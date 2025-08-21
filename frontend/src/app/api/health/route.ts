import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Check backend health
    const backendResponse = await fetch('http://localhost:8000/health', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (backendResponse.ok) {
      const backendHealth = await backendResponse.json();
      
      return NextResponse.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        frontend: {
          status: 'healthy',
          version: '1.0.0'
        },
        backend: {
          status: 'healthy',
          ...backendHealth
        },
        services: {
          speech_analyzer: 'healthy',
          retinal_analyzer: 'healthy',
          motor_analyzer: 'healthy',
          cognitive_analyzer: 'healthy',
          nri_fusion: 'healthy',
          validation_engine: 'healthy'
        }
      });
    } else {
      // Backend not available, return frontend-only health
      return NextResponse.json({
        status: 'warning',
        timestamp: new Date().toISOString(),
        frontend: {
          status: 'healthy',
          version: '1.0.0'
        },
        backend: {
          status: 'unavailable',
          message: 'Backend service not responding'
        },
        services: {
          speech_analyzer: 'unavailable',
          retinal_analyzer: 'unavailable',
          motor_analyzer: 'unavailable',
          cognitive_analyzer: 'unavailable',
          nri_fusion: 'unavailable',
          validation_engine: 'unavailable'
        }
      }, { status: 200 });
    }

  } catch (error) {
    console.error('Health check error:', error);
    
    return NextResponse.json({
      status: 'error',
      timestamp: new Date().toISOString(),
      frontend: {
        status: 'healthy',
        version: '1.0.0'
      },
      backend: {
        status: 'error',
        message: 'Failed to connect to backend'
      },
      services: {
        speech_analyzer: 'error',
        retinal_analyzer: 'error',
        motor_analyzer: 'error',
        cognitive_analyzer: 'error',
        nri_fusion: 'error',
        validation_engine: 'error'
      }
    }, { status: 503 });
  }
}
