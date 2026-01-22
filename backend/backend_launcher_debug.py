
import uvicorn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    try:
        print("Starting Uvicorn...")
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")
    except Exception as e:
        print(f"Failed to start: {e}")
        import traceback
        traceback.print_exc()
