#!/usr/bin/env python3
"""
Apple Detection Web App Launcher

This script launches the Flask web application for apple detection.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Launch the Flask application."""
    try:
        from app import app

        print("🍎 Apple Detection Web App")
        print("=" * 40)
        print("Starting Flask server...")
        print("App will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print()

        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True
        )

    except ImportError as e:
        print(f"❌ Error: Could not import the application. {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()