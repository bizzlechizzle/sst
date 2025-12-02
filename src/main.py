"""Social Screenshot Tool - Main entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .utils.logging import setup_logging
from .utils.ffmpeg import check_ffmpeg


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Social Screenshot Tool - Extract Instagram-worthy frames from wedding videos"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in command-line mode (no GUI)'
    )
    parser.add_argument(
        '--folder',
        type=Path,
        help='Video folder to analyze (CLI mode)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output folder for exports (CLI mode)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    logger.info("Social Screenshot Tool starting...")
    
    # Check FFmpeg
    if not check_ffmpeg():
        logger.error("FFmpeg not found! Please install FFmpeg and add it to PATH.")
        logger.error("Download from: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    if args.cli:
        # Command-line mode
        if not args.folder:
            logger.error("--folder is required in CLI mode")
            sys.exit(1)
        
        run_cli(args.folder, args.output)
    else:
        # GUI mode
        run_gui()


def run_gui():
    """Launch the GUI application."""
    from .gui.main_window import main as gui_main
    gui_main()


def run_cli(folder: Path, output: Path = None):
    """Run analysis in command-line mode.
    
    Args:
        folder: Video folder to analyze
        output: Output folder for exports
    """
    import logging
    from .analysis.pipeline import AnalysisPipeline
    from .export.screenshot import export_frame_all_crops
    from .core.config import DEFAULT_QUOTAS
    
    logger = logging.getLogger('sst')
    
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        return
    
    if output is None:
        output = folder / 'sst_output'
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline
    pipeline = AnalysisPipeline()
    pipeline.set_progress_callback(
        lambda pct, msg: logger.info(f"[{pct:3d}%] {msg}")
    )
    
    # Analyze
    logger.info(f"Analyzing videos in: {folder}")
    project = pipeline.analyze_project(folder)
    
    # Export top frames
    logger.info(f"Exporting to: {output}")
    
    # People
    people_quota = DEFAULT_QUOTAS['people_screenshots']
    for frame in project.people_frames[:people_quota]:
        try:
            export_frame_all_crops(frame, output)
        except Exception as e:
            logger.warning(f"Failed to export frame: {e}")
    
    # Details
    details_quota = DEFAULT_QUOTAS['details_screenshots']
    for frame in project.details_frames[:details_quota]:
        try:
            export_frame_all_crops(frame, output)
        except Exception as e:
            logger.warning(f"Failed to export frame: {e}")
    
    # Venue
    venue_quota = DEFAULT_QUOTAS['venue_screenshots']
    for frame in project.venue_frames[:venue_quota]:
        try:
            export_frame_all_crops(frame, output)
        except Exception as e:
            logger.warning(f"Failed to export frame: {e}")
    
    logger.info("Export complete!")
    logger.info(f"  People: {min(len(project.people_frames), people_quota)} frames")
    logger.info(f"  Details: {min(len(project.details_frames), details_quota)} frames")
    logger.info(f"  Venue: {min(len(project.venue_frames), venue_quota)} frames")


if __name__ == '__main__':
    main()
