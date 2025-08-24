"""
Test script to verify database operations work correctly
"""

import asyncio
from datetime import datetime
from app.core.database import SessionLocal
from app.services.user_service import user_service
from app.services.assessment_service import assessment_service
from app.services.validation_service import validation_service


def test_database_operations():
    """Test all database CRUD operations"""
    db = SessionLocal()
    
    try:
        print("🧪 Testing Database Operations...")
        
        # Test 1: Create a test user
        print("\n1. Testing User Creation...")
        test_user = user_service.create_user(
            db,
            email=f"test_{datetime.now().timestamp()}@example.com",
            username=f"test_user_{int(datetime.now().timestamp())}",
            age=45,
            sex="female",
            consent_given=True
        )
        print(f"✅ Created user: {test_user.id}")
        
        # Test 2: Create user profile
        print("\n2. Testing User Profile Creation...")
        profile = user_service.create_user_profile(
            db,
            test_user.id,
            {
                "ethnicity": "Test",
                "occupation": "Tester",
                "handedness": "right",
                "baseline_cognitive_score": 0.85
            }
        )
        print(f"✅ Created user profile for: {test_user.id}")
        
        # Test 3: Create assessment
        print("\n3. Testing Assessment Creation...")
        assessment = assessment_service.create_assessment(
            db,
            user_id=test_user.id,
            modalities=["speech", "retinal"],
            assessment_type="test"
        )
        print(f"✅ Created assessment: {assessment.session_id}")
        
        # Test 4: Test database queries
        print("\n4. Testing Database Queries...")
        
        # Get user by ID
        retrieved_user = user_service.get_user(db, test_user.id)
        print(f"✅ Retrieved user: {retrieved_user.email}")
        
        # Get user assessments
        user_assessments = user_service.get_user_assessment_history(db, test_user.id)
        print(f"✅ Found {len(user_assessments)} assessments for user")
        
        # Get assessment by session ID
        retrieved_assessment = assessment_service.get_assessment(db, assessment.session_id)
        print(f"✅ Retrieved assessment: {retrieved_assessment.status}")
        
        # Test 5: Update operations
        print("\n5. Testing Update Operations...")
        
        # Update user
        updated_user = user_service.update_user(
            db, 
            test_user.id, 
            {"age": 46}
        )
        print(f"✅ Updated user age to: {updated_user.age}")
        
        # Update assessment status
        updated_assessment = assessment_service.update_assessment_status(
            db,
            assessment.session_id,
            "completed"
        )
        print(f"✅ Updated assessment status to: {updated_assessment.status}")
        
        # Test 6: Performance metrics
        print("\n6. Testing Performance Metrics...")
        
        metric = validation_service.record_performance_metric(
            db,
            metric_name="test_metric",
            value=95.5,
            component="test_component",
            modality="test",
            unit="percentage"
        )
        print(f"✅ Recorded performance metric: {metric.metric_name} = {metric.value}")
        
        # Test 7: System health
        print("\n7. Testing System Health...")
        
        health = validation_service.record_system_health(
            db,
            component="test_system",
            status="healthy",
            response_time=25.5
        )
        print(f"✅ Recorded system health: {health.component} = {health.status}")
        
        print("\n🎉 All database operations completed successfully!")
        
        return {
            "status": "success",
            "user_id": test_user.id,
            "assessment_id": assessment.id,
            "tests_passed": 7
        }
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()


def test_database_health():
    """Test database health and connection"""
    print("\n🏥 Testing Database Health...")
    
    try:
        from app.services.database_service import DatabaseManager
        
        # Test database health
        health = DatabaseManager.health_check()
        print(f"Database Status: {health['status']}")
        
        # Test table info
        table_info = DatabaseManager.get_table_info()
        print(f"Tables Found: {table_info['table_count']}")
        
        for table_name, info in table_info['tables'].items():
            print(f"  - {table_name}: {info['column_count']} columns")
        
        return health
        
    except Exception as e:
        print(f"❌ Database health check failed: {str(e)}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    print("🚀 Starting NeuraLens Database Tests...")
    
    # Test database health
    health_result = test_database_health()
    
    if health_result["status"] == "healthy":
        # Test database operations
        test_result = test_database_operations()
        print(f"\n📊 Final Test Result: {test_result}")
    else:
        print(f"❌ Database health check failed: {health_result}")
