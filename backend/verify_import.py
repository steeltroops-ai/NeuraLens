import sys
import os
sys.path.append(os.getcwd())
try:
    print("Attempting imports...")
    from app.pipelines.speech.router import router
    print("Router imported.")
    from app.pipelines.speech.analyzer import SpeechPipelineService
    print("Analyzer imported.")
    from app.pipelines.speech.processor import AcousticProcessor
    print("Processor imported.")
    
    # Test Instantiation
    p = AcousticProcessor()
    s = SpeechPipelineService()
    print("Instantiation Successful")
    
except Exception as e:
    print(f"Import Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
