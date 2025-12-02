"""FFmpeg utilities."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> Optional[Path]:
    """Find FFmpeg executable.
    
    Returns:
        Path to ffmpeg or None if not found
    """
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return Path(ffmpeg_path)
    
    # Common installation locations
    common_paths = [
        Path('/usr/bin/ffmpeg'),
        Path('/usr/local/bin/ffmpeg'),
        Path('C:/ffmpeg/bin/ffmpeg.exe'),
        Path('C:/Program Files/ffmpeg/bin/ffmpeg.exe'),
    ]
    
    for path in common_paths:
        if path.exists():
            return path
    
    return None


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and working.
    
    Returns:
        True if FFmpeg is available
    """
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        logger.error("FFmpeg not found in PATH")
        return False
    
    try:
        result = subprocess.run(
            [str(ffmpeg_path), '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse version from output
        version_line = result.stdout.split('\n')[0]
        logger.info(f"FFmpeg found: {version_line}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg check failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {e}")
        return False


def get_ffprobe_path() -> Optional[Path]:
    """Find FFprobe executable.
    
    Returns:
        Path to ffprobe or None if not found
    """
    ffprobe_path = shutil.which('ffprobe')
    if ffprobe_path:
        return Path(ffprobe_path)
    
    # If ffmpeg exists, ffprobe should be alongside it
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path:
        ffprobe_path = ffmpeg_path.parent / 'ffprobe'
        if ffprobe_path.exists():
            return ffprobe_path
        
        # Windows
        ffprobe_path = ffmpeg_path.parent / 'ffprobe.exe'
        if ffprobe_path.exists():
            return ffprobe_path
    
    return None


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        raise RuntimeError("FFprobe not found")
    
    cmd = [
        str(ffprobe_path),
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """Get video resolution.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height)
    """
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        raise RuntimeError("FFprobe not found")
    
    cmd = [
        str(ffprobe_path),
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = result.stdout.strip().split('x')
    return int(width), int(height)
