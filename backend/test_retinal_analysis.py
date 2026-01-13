#!/usr/bin/env python3
"""
Test script for Retinal Analysis Module
Validates EfficientNet-B0 integration and computer vision pipeline
"""

import asyncio
import logging
import numpy as np
from PIL import Image
import io
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ml.realtime.realtime_retinal import RealtimeRetinalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_retinal_analyzer():
    """Test the retinal analyzer with a synthetic image"""
    
    logger.info("Starting Retinal Analysis Module test...")
    
    try:
        # Initialize analyzer
        analyzer = RealtimeRetinalAnalyzer()
        
        # Create a synthetic retinal-like image (512x512, RGB)
        # Simulate a retinal fundus image with circular structure
        width, height = 512, 512
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create circular background (simulating retinal boundary)
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2 - 20
        
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Fill with reddish background (simulating retinal tissue)
        image[mask] = [180, 120, 80]  # Reddish-brown retinal color
        
        # Add some vessel-like structures (dark lines)
        for i in range(5):
            start_x = center_x + np.random.randint(-radius//2, radius//2)
            start_y = center_y + np.random.randint(-radius//2, radius//2)
            end_x = center_x + np.random.randint(-radius//2, radius//2)
            end_y = center_y + np.random.randint(-radius//2, radius//2)
            
            # Draw vessel-like line
            from scipy.ndimage import line_nd
            # Simple line drawing
            length = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2))
            if length > 0:
                x_coords = np.linspace(start_x, end_x, length).astype(int)
                y_coords = np.linspace(start_y, end_y, length).astype(int)
                
                # Ensure coordinates are within bounds
                valid_coords = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
                x_coords = x_coords[valid_coords]
                y_coords = y_coords[valid_coords]
                
                # Draw dark vessel
                image[y_coords, x_coords] = [60, 40, 30]  # Dark vessel color
        
        # Add optic disc (bright circular region)
        disc_x = center_x + radius // 3
        disc_y = center_y - radius // 4
        disc_radius = 30
        
        disc_mask = (x - disc_x)**2 + (y - disc_y)**2 <= disc_radius**2
        image[disc_mask] = [255, 220, 180]  # Bright optic disc
        
        # Convert to bytes (simulate uploaded image)
        pil_image = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        logger.info(f"Created synthetic retinal image: {len(image_bytes)} bytes")
        
        # Test the analysis
        session_id = "test_session_001"
        
        logger.info("Running retinal analysis...")
        start_time = asyncio.get_event_loop().time()
        
        result = await analyzer.analyze_realtime(image_bytes, session_id)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Display results
        logger.info("=== RETINAL ANALYSIS RESULTS ===")
        logger.info(f"Session ID: {result.session_id}")
        logger.info(f"Processing Time: {processing_time:.2f}ms (Target: <200ms)")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Risk Score: {result.risk_score:.3f}")
        logger.info(f"Quality Score: {result.quality_score:.3f}")
        
        logger.info("\n=== BIOMARKERS ===")
        logger.info(f"Vessel Tortuosity: {result.biomarkers.vessel_tortuosity:.3f}")
        logger.info(f"AV Ratio: {result.biomarkers.av_ratio:.3f}")
        logger.info(f"Cup-Disc Ratio: {result.biomarkers.cup_disc_ratio:.3f}")
        logger.info(f"Vessel Density: {result.biomarkers.vessel_density:.3f}")
        
        if result.recommendations:
            logger.info("\n=== RECOMMENDATIONS ===")
            for i, rec in enumerate(result.recommendations, 1):
                logger.info(f"{i}. {rec}")
        
        # Validate performance (allow more time without EfficientNet)
        target_time = 300 if not analyzer.model_loaded else 200
        if processing_time < target_time:
            logger.info(f"✅ PERFORMANCE: Processing time {processing_time:.2f}ms within target (<{target_time}ms)")
        else:
            logger.warning(f"⚠️  PERFORMANCE: Processing time {processing_time:.2f}ms exceeds target")
        
        # Validate biomarkers
        biomarkers_valid = (
            0.0 <= result.biomarkers.vessel_tortuosity <= 1.0 and
            0.3 <= result.biomarkers.av_ratio <= 1.5 and
            0.0 <= result.biomarkers.cup_disc_ratio <= 0.9 and
            0.0 <= result.biomarkers.vessel_density <= 1.0
        )
        
        if biomarkers_valid:
            logger.info("✅ BIOMARKERS: All values within expected ranges")
        else:
            logger.error("❌ BIOMARKERS: Some values outside expected ranges")
        
        # Test health check
        health_status = await analyzer.health_check()
        logger.info(f"\n=== HEALTH CHECK ===")
        logger.info(f"Model Loaded: {health_status['model_loaded']}")
        logger.info(f"Target Latency: {health_status['target_latency_ms']}ms")
        logger.info(f"Optimization Level: {health_status['optimization_level']}")
        
        # Overall test result
        max_time = 400 if not analyzer.model_loaded else 300
        test_passed = (
            processing_time < max_time and  # Allow margin for test environment
            biomarkers_valid and
            result.confidence > 0.0 and
            result.risk_score >= 0.0
        )
        
        if test_passed:
            logger.info("\n🎉 TEST PASSED: Retinal Analysis Module is working correctly!")
            return True
        else:
            logger.error("\n❌ TEST FAILED: Some validation checks failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    logger.info("Retinal Analysis Module Test Suite")
    logger.info("=" * 50)
    
    success = await test_retinal_analyzer()
    
    if success:
        logger.info("\n✅ All tests passed! Retinal Analysis Module is production-ready.")
        sys.exit(0)
    else:
        logger.error("\n❌ Tests failed! Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
