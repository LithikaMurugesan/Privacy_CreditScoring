"""
run_api.py
==========
Entry point for the FastAPI credit scoring REST API.

Usage:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8000

The API will be available at:
    http://localhost:8000
    http://localhost:8000/docs  (interactive Swagger UI)
    http://localhost:8000/predict (POST endpoint)

Train a model first to get real predictions:
    python run_experiment.py --mode fedavg_dp

Or just run the API for demo purposes (untrained model).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Start FL Credit Scoring API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Enable hot-reload for development")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    try:
        from src.api.server import app
        if app is None:
            print("FastAPI not installed. Run: pip install fastapi uvicorn")
            sys.exit(1)
    except ImportError as e:
        print(f"Failed to import API: {e}")
        sys.exit(1)

    print(f"\n  FL Credit Scoring API")
    print(f"  Starting on http://{args.host}:{args.port}")
    print(f"  Swagger UI: http://localhost:{args.port}/docs")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
