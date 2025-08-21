"""
End-to-End Assessment Flow Test
Tests P0-3: Complete assessment workflow from data input to NRI calculation
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from PIL import Image
import io

# Import all components
from app.ml.models.speech_analyzer import SpeechAnalyzer
from app.ml.models.retinal_analyzer import retinal_analyzer
from app.ml.models.motor_analyzer import motor_analyzer
from app.ml.models.cognitive_analyzer import cognitive_analyzer
from app.ml.models.nri_fusion import nri_fusion_engine
from app.ml.models.validation_engine import validation_engine

from app.schemas.assessment import (
    MotorAssessmentRequest, CognitiveAssessmentRequest, NRIFusionRequest
)

async def test_complete_assessment_flow():
    """Test complete multi-modal assessment flow"""
    
    print("🔄 Testing Complete End-to-End Assessment Flow")
    print("=" * 60)
    
    session_id = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📋 Session ID: {session_id}")
    
    assessment_results = {}
    
    # Step 1: Speech Assessment
    print("\n🎤 Step 1: Speech Assessment")
    try:
        speech_analyzer = SpeechAnalyzer()
        
        # Create realistic speech audio (5 seconds)
        sample_rate = 22050
        duration = 5.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Simulate speech with formants and natural variation
        speech_signal = (
            0.4 * np.sin(2 * np.pi * 150 * t) +  # Fundamental frequency
            0.3 * np.sin(2 * np.pi * 300 * t) +  # First harmonic
            0.2 * np.sin(2 * np.pi * 450 * t) +  # Second harmonic
            0.1 * np.random.normal(0, 1, num_samples)  # Natural noise
        )
        
        # Add amplitude modulation for natural speech patterns
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
        speech_signal = speech_signal * envelope
        
        # Normalize and convert to bytes
        speech_signal = speech_signal / np.max(np.abs(speech_signal))
        audio_bytes = (speech_signal * 32767).astype(np.int16).tobytes()
        
        speech_result = await speech_analyzer.analyze(audio_bytes, f"{session_id}_speech")
        assessment_results['speech'] = {
            'risk_score': speech_result.risk_score,
            'confidence': speech_result.confidence,
            'biomarkers': {
                'fluency_score': speech_result.biomarkers.fluency_score,
                'voice_tremor': speech_result.biomarkers.voice_tremor,
                'speaking_rate': speech_result.biomarkers.speaking_rate
            }
        }
        
        print(f"   ✅ Speech analysis completed")
        print(f"      Risk Score: {speech_result.risk_score:.3f}")
        print(f"      Confidence: {speech_result.confidence:.3f}")
        print(f"      Fluency: {speech_result.biomarkers.fluency_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Speech assessment failed: {e}")
        assessment_results['speech'] = {'error': str(e)}
    
    # Step 2: Retinal Assessment
    print("\n👁️ Step 2: Retinal Assessment")
    try:
        # Create realistic retinal fundus image
        image = Image.new('RGB', (512, 512), color='black')
        pixels = image.load()
        
        for i in range(512):
            for j in range(512):
                # Create retinal features
                center_x, center_y = 256, 256
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                if distance < 50:  # Optic disc
                    pixels[i, j] = (220, 180, 120)
                elif distance < 150:  # Vessel area
                    vessel_intensity = int(180 - distance * 0.5)
                    pixels[i, j] = (vessel_intensity, vessel_intensity//2, vessel_intensity//3)
                else:  # Background retina
                    bg_intensity = int(120 - (distance - 150) * 0.1)
                    bg_intensity = max(50, bg_intensity)
                    pixels[i, j] = (bg_intensity, bg_intensity//2, bg_intensity//4)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        retinal_result = await retinal_analyzer.analyze(image_bytes, f"{session_id}_retinal")
        assessment_results['retinal'] = {
            'risk_score': retinal_result.risk_score,
            'confidence': retinal_result.confidence,
            'biomarkers': {
                'vessel_tortuosity': retinal_result.biomarkers.vessel_tortuosity,
                'av_ratio': retinal_result.biomarkers.av_ratio,
                'cup_disc_ratio': retinal_result.biomarkers.cup_disc_ratio
            }
        }
        
        print(f"   ✅ Retinal analysis completed")
        print(f"      Risk Score: {retinal_result.risk_score:.3f}")
        print(f"      Confidence: {retinal_result.confidence:.3f}")
        print(f"      Vessel Tortuosity: {retinal_result.biomarkers.vessel_tortuosity:.3f}")
        
    except Exception as e:
        print(f"   ❌ Retinal assessment failed: {e}")
        assessment_results['retinal'] = {'error': str(e)}
    
    # Step 3: Motor Assessment
    print("\n🤲 Step 3: Motor Assessment")
    try:
        # Create realistic finger tapping sensor data
        duration = 20.0  # 20 seconds
        sample_rate = 50  # 50 Hz
        num_samples = int(duration * sample_rate)
        
        # Simulate finger tapping motion
        tapping_frequency = 4.5  # 4.5 Hz tapping
        time_points = np.linspace(0, duration, num_samples)
        
        # Base tapping motion with some irregularity
        base_motion = np.sin(2 * np.pi * tapping_frequency * time_points)
        irregularity = 0.1 * np.random.normal(0, 1, num_samples)
        
        sensor_data = {
            "accelerometer": [
                {
                    "x": 0.1 + base_motion[i] * 2.0 + irregularity[i],
                    "y": 0.2 + np.random.normal(0, 0.1),
                    "z": 9.8 + np.random.normal(0, 0.2)
                }
                for i in range(num_samples)
            ],
            "gyroscope": [
                {
                    "x": base_motion[i] * 0.5 + np.random.normal(0, 0.1),
                    "y": np.random.normal(0, 0.05),
                    "z": np.random.normal(0, 0.05)
                }
                for i in range(num_samples)
            ],
            "timestamp": [i / sample_rate for i in range(num_samples)]
        }
        
        motor_request = MotorAssessmentRequest(
            assessment_type="finger_tapping",
            sensor_data=sensor_data,
            duration=duration
        )
        
        motor_result = await motor_analyzer.analyze(motor_request, f"{session_id}_motor")
        assessment_results['motor'] = {
            'risk_score': motor_result.risk_score,
            'confidence': motor_result.confidence,
            'biomarkers': {
                'movement_frequency': motor_result.biomarkers.movement_frequency,
                'coordination_index': motor_result.biomarkers.coordination_index,
                'tremor_severity': motor_result.biomarkers.tremor_severity
            }
        }
        
        print(f"   ✅ Motor analysis completed")
        print(f"      Risk Score: {motor_result.risk_score:.3f}")
        print(f"      Confidence: {motor_result.confidence:.3f}")
        print(f"      Movement Frequency: {motor_result.biomarkers.movement_frequency:.1f} Hz")
        
    except Exception as e:
        print(f"   ❌ Motor assessment failed: {e}")
        assessment_results['motor'] = {'error': str(e)}
    
    # Step 4: Cognitive Assessment
    print("\n🧠 Step 4: Cognitive Assessment")
    try:
        # Realistic cognitive test results
        test_results = {
            "memory": {
                "immediate_recall": 0.82,
                "delayed_recall": 0.76,
                "recognition": 0.91
            },
            "attention": {
                "sustained_attention": 0.78,
                "selective_attention": 0.84,
                "divided_attention": 0.69
            },
            "executive": {
                "planning": 0.73,
                "inhibition": 0.81,
                "flexibility": 0.77
            },
            "language": {
                "fluency": 0.85,
                "naming": 0.89,
                "comprehension": 0.87
            }
        }
        
        cognitive_request = CognitiveAssessmentRequest(
            test_battery=["memory", "attention", "executive", "language"],
            test_results=test_results,
            difficulty_level="standard"
        )
        
        cognitive_result = await cognitive_analyzer.analyze(cognitive_request, f"{session_id}_cognitive")
        assessment_results['cognitive'] = {
            'risk_score': cognitive_result.risk_score,
            'confidence': cognitive_result.confidence,
            'biomarkers': {
                'memory_score': cognitive_result.biomarkers.memory_score,
                'attention_score': cognitive_result.biomarkers.attention_score,
                'executive_score': cognitive_result.biomarkers.executive_score
            }
        }
        
        print(f"   ✅ Cognitive analysis completed")
        print(f"      Risk Score: {cognitive_result.risk_score:.3f}")
        print(f"      Confidence: {cognitive_result.confidence:.3f}")
        print(f"      Memory Score: {cognitive_result.biomarkers.memory_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Cognitive assessment failed: {e}")
        assessment_results['cognitive'] = {'error': str(e)}
    
    # Step 5: NRI Fusion
    print("\n🔬 Step 5: NRI Fusion")
    try:
        # Collect risk scores from successful assessments
        modality_scores = {}
        modality_confidences = {}
        
        for modality, result in assessment_results.items():
            if 'risk_score' in result:
                modality_scores[modality] = result['risk_score']
                modality_confidences[modality] = result['confidence']
        
        if len(modality_scores) >= 2:  # Need at least 2 modalities for fusion
            nri_request = NRIFusionRequest(
                session_id=f"{session_id}_nri",
                modalities=list(modality_scores.keys()),
                modality_scores=modality_scores,
                modality_confidences=modality_confidences,
                fusion_method="bayesian"
            )
            
            nri_result = await nri_fusion_engine.calculate_nri(nri_request)
            
            print(f"   ✅ NRI fusion completed")
            print(f"      NRI Score: {nri_result.nri_score:.1f}/100")
            print(f"      Risk Category: {nri_result.risk_category.upper()}")
            print(f"      Confidence: {nri_result.confidence:.3f}")
            print(f"      Consistency: {nri_result.consistency_score:.3f}")
            
            # Display modality contributions
            print(f"      Modality Contributions:")
            for contrib in nri_result.modality_contributions:
                print(f"        {contrib.modality.capitalize()}: {contrib.contribution:.1%} (score: {contrib.risk_score:.3f})")
            
            assessment_results['nri'] = {
                'nri_score': nri_result.nri_score,
                'risk_category': nri_result.risk_category,
                'confidence': nri_result.confidence,
                'consistency_score': nri_result.consistency_score
            }
            
        else:
            print(f"   ⚠️ Insufficient modalities for NRI fusion ({len(modality_scores)}/2 minimum)")
            
    except Exception as e:
        print(f"   ❌ NRI fusion failed: {e}")
        assessment_results['nri'] = {'error': str(e)}
    
    # Step 6: Validation Metrics
    print("\n📊 Step 6: Validation Metrics")
    try:
        validation_metrics = await validation_engine.get_validation_metrics("all", "performance")
        print(f"   ✅ Validation metrics retrieved")
        print(f"      Available metrics for {len(validation_metrics.get('performance', {}))} modalities")
        
    except Exception as e:
        print(f"   ❌ Validation metrics failed: {e}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎯 END-TO-END ASSESSMENT SUMMARY")
    print("=" * 60)
    
    successful_assessments = sum(1 for result in assessment_results.values() if 'risk_score' in result)
    total_assessments = len(assessment_results)
    
    print(f"✅ Successful Assessments: {successful_assessments}/{total_assessments}")
    
    if 'nri' in assessment_results and 'nri_score' in assessment_results['nri']:
        nri_data = assessment_results['nri']
        print(f"\n🧠 FINAL NEUROLOGICAL RISK INDEX")
        print(f"   Score: {nri_data['nri_score']:.1f}/100")
        print(f"   Category: {nri_data['risk_category'].upper()}")
        print(f"   Confidence: {nri_data['confidence']:.1%}")
        print(f"   Consistency: {nri_data['consistency_score']:.1%}")
        
        if nri_data['nri_score'] < 25:
            print(f"   🟢 LOW RISK - Continue routine monitoring")
        elif nri_data['nri_score'] < 50:
            print(f"   🟡 MODERATE RISK - Consider follow-up assessment")
        elif nri_data['nri_score'] < 75:
            print(f"   🟠 HIGH RISK - Recommend clinical evaluation")
        else:
            print(f"   🔴 VERY HIGH RISK - Urgent clinical consultation recommended")
    
    print(f"\n🎉 End-to-End Assessment Flow Test Completed!")
    print(f"📋 Session: {session_id}")
    
    return assessment_results

if __name__ == "__main__":
    results = asyncio.run(test_complete_assessment_flow())
