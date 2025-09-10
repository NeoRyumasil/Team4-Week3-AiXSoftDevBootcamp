#!/usr/bin/env python3
"""
Startup script for Personal Knowledge Assistant
This script sets proper environment variables and launches the application.
"""

import os
import sys
import warnings
from pathlib import Path

# Set environment variables before importing anything else
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*")

def main():
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import and run the main application
        import subprocess
        
        print("🧠 Starting Personal Knowledge Assistant...")
        print("📱 Launching web interface...")
        
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/streamlit_app.py",
            "--server.headless=false",
            "--server.runOnSave=false",
            "--browser.gatherUsageStats=false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Try running: python main.py --ui")

if __name__ == "__main__":
    main()