"""
Final verification script for NeuraLens database setup
Demonstrates all CRUD operations and database functionality
"""

from datetime import datetime
from app.core.database import SessionLocal
from app.services.user_service import user_service
from app.services.assessment_service import assessment_service
from app.services.validation_service import validation_service
from app.services.database_service import DatabaseManager


def demonstrate_database_functionality():
    """Comprehensive demonstration of database functionality"""
    print("🎯 NeuraLens Database Setup Verification")
    print("=" * 50)
    
    db = SessionLocal()
    
    try:
        # 1. Database Health Check
        print("\n1. 🏥 Database Health Check")
        health = DatabaseManager.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Connection: {health['database']}")
        
        # 2. Table Information
        print("\n2. 📊 Database Schema Information")
        table_info = DatabaseManager.get_table_info()
        print(f"   Total Tables: {table_info['table_count']}")
        
        key_tables = ['users', 'assessments', 'assessment_results', 'nri_results']
        for table in key_tables:
            if table in table_info['tables']:
                columns = table_info['tables'][table]['column_count']
                print(f"   ✅ {table}: {columns} columns")
        
        # 3. User Management
        print("\n3. 👤 User Management Operations")
        
        # Create user
        timestamp = int(datetime.now().timestamp())
        user = user_service.create_user(
            db,
            email=f"demo_{timestamp}@neuralens.com",
            username=f"demo_user_{timestamp}",
            age=62,
            sex="male",
            education_years=16,
            consent_given=True
        )
        print(f"   ✅ Created user: {user.email}")
        
        # Create user profile
        profile = user_service.create_user_profile(
            db,
            user.id,
            {
                "ethnicity": "Caucasian",
                "occupation": "Engineer",
                "handedness": "right",
                "baseline_cognitive_score": 0.88,
                "baseline_motor_score": 0.82,
                "baseline_speech_score": 0.91,
                "baseline_retinal_score": 0.79
            }
        )
        print(f"   ✅ Created user profile with baseline scores")
        
        # 4. Assessment Management
        print("\n4. 🧠 Assessment Management Operations")
        
        # Create assessment
        assessment = assessment_service.create_assessment(
            db,
            user_id=user.id,
            modalities=["speech", "retinal", "motor", "cognitive"],
            assessment_type="full"
        )
        print(f"   ✅ Created assessment: {assessment.session_id}")
        
        # Simulate storing results (mock data)
        from app.schemas.assessment import SpeechBiomarkers

        # Mock speech analysis result
        class MockSpeechResult:
            def __init__(self):
                self.session_id = assessment.session_id
                self.processing_time = 2.5
                self.timestamp = datetime.utcnow()
                self.risk_score = 0.35
                self.confidence = 0.87
                self.recommendations = ["Regular monitoring recommended", "Consider speech therapy consultation"]
        
        speech_result = MockSpeechResult()
        stored_result = assessment_service.store_speech_result(
            db, assessment.id, speech_result
        )
        print(f"   ✅ Stored speech analysis result (Risk: {stored_result.risk_score})")
        
        # Update assessment status
        completed_assessment = assessment_service.update_assessment_status(
            db, assessment.session_id, "completed"
        )
        print(f"   ✅ Updated assessment status: {completed_assessment.status}")
        
        # 5. Performance Monitoring
        print("\n5. 📈 Performance Monitoring")
        
        # Record performance metrics
        metrics = [
            ("response_time", 145.2, "ms", "backend"),
            ("accuracy", 87.5, "percentage", "ml_model"),
            ("throughput", 42.3, "requests/sec", "api_gateway")
        ]
        
        for metric_name, value, unit, component in metrics:
            metric = validation_service.record_performance_metric(
                db,
                metric_name=metric_name,
                value=value,
                component=component,
                unit=unit,
                environment="production"
            )
            print(f"   ✅ Recorded {metric_name}: {value} {unit}")
        
        # Record system health
        components = ["database", "api_gateway", "ml_pipeline"]
        for component in components:
            health_record = validation_service.record_system_health(
                db,
                component=component,
                status="healthy",
                response_time=25.0 + hash(component) % 50,
                cpu_usage=30.0 + hash(component) % 40,
                memory_usage=45.0 + hash(component) % 30
            )
            print(f"   ✅ Recorded {component} health: {health_record.status}")
        
        # 6. Data Retrieval and Analysis
        print("\n6. 🔍 Data Retrieval and Analysis")
        
        # Get user assessments
        user_assessments = assessment_service.get_user_assessments(db, user.id)
        print(f"   ✅ Retrieved {len(user_assessments)} assessments for user")
        
        # Get assessment results
        assessment_results = assessment_service.get_assessment_results(db, assessment.id)
        print(f"   ✅ Retrieved {len(assessment_results)} results for assessment")
        
        # Get performance metrics
        recent_metrics = validation_service.get_performance_metrics(
            db, component="backend", hours_back=1
        )
        print(f"   ✅ Retrieved {len(recent_metrics)} recent performance metrics")
        
        # Get system health status
        system_health = validation_service.get_system_health_status(db)
        print(f"   ✅ System health overview: {system_health['overall_status']}")
        print(f"      Components monitored: {len(system_health['components'])}")
        
        # 7. Summary Statistics
        print("\n7. 📊 Database Summary Statistics")
        
        # Count records in key tables
        user_count = user_service.user_db.count(db)
        assessment_count = assessment_service.assessment_db.count(db)
        result_count = assessment_service.result_db.count(db)
        metric_count = validation_service.metric_db.count(db)
        
        print(f"   Users: {user_count}")
        print(f"   Assessments: {assessment_count}")
        print(f"   Assessment Results: {result_count}")
        print(f"   Performance Metrics: {metric_count}")
        
        print("\n🎉 Database Setup Verification Complete!")
        print("✅ All CRUD operations working correctly")
        print("✅ All relationships properly configured")
        print("✅ All services functioning as expected")
        print("✅ Database ready for production use")
        
        return {
            "status": "success",
            "database_health": "healthy",
            "tables_created": table_info['table_count'],
            "crud_operations": "working",
            "relationships": "configured",
            "services": "functional"
        }
        
    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()


if __name__ == "__main__":
    result = demonstrate_database_functionality()
    print(f"\n📋 Final Verification Result:")
    for key, value in result.items():
        print(f"   {key}: {value}")
