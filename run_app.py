#!/usr/bin/env python3
"""
Run the Shakespeare Text Generator FastAPI app
"""
import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🎭 Starting Shakespeare Text Generator...")
    print("📝 Loading model (this may take a moment)...")
    
    # Pre-load the model
    try:
        from app.inference import load_model
        load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load model: {e}")
    
    print("🚀 Starting FastAPI server...")
    print("📱 Open http://localhost:8000 in your browser")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False  # Set to True for development
    )
