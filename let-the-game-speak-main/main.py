"""
Main Entry Point: AI Football Commentator
Simplified main script that uses the modular pipeline.
"""
import sys
from pathlib import Path

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.pipeline import run_pipeline


def main():
    """
    Main entry point for the AI Football Commentator.
    
    Usage:
        python main.py <video_path> [options]
        
    Examples:
        python main.py test.mp4
        python main.py test.mp4 --output ./my_output
        python main.py test.mp4 --no-audio
    """
    if len(sys.argv) < 2:
        print("❌ Error: No video file specified")
        print("\nUsage:")
        print("  python main.py <video_path> [options]")
        print("\nOptions:")
        print("  --output <dir>     Output directory (default: ./output)")
        print("  --model <path>     YOLO model path (default: ./weights/last.pt)")
        print("  --no-audio         Disable audio generation (skip GPT + TTS)")
        print("\nExample:")
        print("  python main.py test.mp4")
        sys.exit(1)
    
    # Parse arguments
    video_path = sys.argv[1]
    output_dir = "./output"
    model_path = "./weights/last.pt"
    enable_audio = True  # Enable narrative and speech generation
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-audio":
            enable_audio = False
            i += 1
        else:
            print(f"⚠️  Unknown option: {sys.argv[i]}")
            i += 1
    
    # Validate inputs
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print("💡 Hint: Make sure you have downloaded the YOLO weights to ./weights/last.pt")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            model_path=model_path,
            enable_narrative=enable_audio,
            enable_speech=enable_audio
        )
        
        print("\n✅ Success! Check the output directory for results.")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
