
import json
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

import asyncio
from sqlalchemy import select, text
from app.database import db
from app.database.models import Patient, Assessment, User
from app.database.repositories import AssessmentRepository

async def verify_data_flow():
    print("🚀 Starting Data Verification...")
    
    # Initialize the database connection for script usage
    db.initialize()
    
    try:
        async with db.session_scope() as db_session:
            repo = AssessmentRepository(db_session)
        
            # 1. Create Test Organization & User if needed
            user_id = uuid.uuid4()
            
            # 2. Create Real Patient
            patient_id = uuid.uuid4()
            print(f"👤 Creating test patient {patient_id}...")
            patient = Patient(
                id=patient_id,
                full_name="Test Verifier Patient",
                phone_number="555-0199",
                email="verify@test.com",
                medical_notes="Verification test subject"
            )
            db_session.add(patient)
            
            # Create a dummy user for the clinician
            clinician = User(
                id=user_id,
                clerk_user_id=f"user_{uuid.uuid4()}",
                email=f"doctor_{uuid.uuid4()}@test.com",
                is_active=True
            )
            db_session.add(clinician)
            
            await db_session.flush() # flush to get IDs if needed or prepare for query
            
            # 3. Create Assessment Linked to Patient
            session_id = f"sess_{uuid.uuid4()}"
            print(f"📝 Creating assessment {session_id} linked to patient...")
            
            assessment = await repo.create_assessment(
                user_id=user_id,
                patient_id=patient_id,
                pipeline_type="verification_test",
                session_id=session_id,
                risk_score=0.15
            )
            
            # 4. Verify Linkage
            print("🔍 Verifying database records...")
            # We are in session_scope, so transaction is active.
            # However, create_assessment might have committed?
            # AssessmentRepository usually uses the session passed.
            
            stmt = select(Assessment).where(Assessment.id == assessment.id)
            result = await db_session.execute(stmt)
            fetched_assessment = result.scalar_one()
            
            assert fetched_assessment.patient_id == patient_id, "❌ Patient ID mismatch!"
            assert fetched_assessment.user_id == user_id, "❌ User ID mismatch!"
            
            print("✅ Assessment correctly linked to Patient!")
            print(f"   Assessment ID: {fetched_assessment.id}")
            print(f"   Patient ID:    {fetched_assessment.patient_id}")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("🧹 Cleaning up test data...")
        try:
            # Re-initialize session for cleanup as the previous scope might have closed/committed
            async with db.session_scope() as cleanup_session:
                # Use raw SQL for cleanup to avoid object detachment issues
                if 'assessment' in locals() and assessment:
                     await cleanup_session.execute(
                         text(f"DELETE FROM assessments WHERE id = '{assessment.id}'")
                     )
                
                if 'patient_id' in locals():
                     await cleanup_session.execute(
                         text(f"DELETE FROM patients WHERE id = '{patient_id}'")
                     )
                     
                if 'user_id' in locals():
                     await cleanup_session.execute(
                         text(f"DELETE FROM users WHERE id = '{user_id}'")
                     )
                print("✨ Cleanup complete.")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup failed: {cleanup_error}")

        await db.close()

    print("VERIFICATION SUCCESS")

if __name__ == "__main__":
    asyncio.run(verify_data_flow())
