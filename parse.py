import argparse
import sys

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="BLIP trainer/annotation tool")

    # Project root (supports both --project-root and --project_root)
    parser.add_argument(
        "--project-root", "--project_root",
        dest="project_root",
        default="/Volumes/abstract-2T/project/",
        help="Root directory for the project"
    )

    # Core selection
    parser.add_argument("--index", type=int, default=-1, help="Item index from metadata CSV")
    parser.add_argument("--action", choices=["annotate", "erase"], help="Action to perform")
    parser.add_argument("--type", choices=["shot", "scene"], default="shot", help="Annotation type (default: shot)")
    parser.add_argument("--media", choices=["movie", "gameplay"], default="movie", help="Media library (default: movie)")

    # Annotation controls
    parser.add_argument("--shot_index", type=int, default=1, help="Starting shot index (1-based)")
    parser.add_argument("--annotation_count", type=int, default=None, help="Number of shots to annotate (default: all)")
    parser.add_argument("--filelist", type=str, default=None, help="Text file with one video filename per line")

    # Model/runtime
    parser.add_argument("--model", type=str, default="gemma3:27b", help="Ollama model name")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window")
    parser.add_argument("--temperature", type=float, default=0.3, help="Model temperature (0.0-1.0, default 0.3)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Show help if no args
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args()