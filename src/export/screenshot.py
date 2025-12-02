"""Screenshot export functions."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from ..core.config import JPEG_QUALITY
from ..core.models import Frame, Video
from ..crops import crop_to_preset, crop_all_presets

logger = logging.getLogger(__name__)


def export_screenshot(
    image: Image.Image,
    output_path: Path,
    quality: int = JPEG_QUALITY,
) -> Path:
    """Save an image as JPEG.
    
    Args:
        image: PIL Image to save
        output_path: Where to save (should end in .jpg)
        quality: JPEG quality (1-100)
    
    Returns:
        Path to saved file
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to RGB if necessary (JPEG doesn't support alpha)
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    
    # Save with quality setting
    image.save(
        output_path,
        'JPEG',
        quality=quality,
        optimize=True,
    )
    
    logger.debug(f"Saved screenshot: {output_path}")
    return output_path


def extract_frame(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path = None,
) -> Image.Image:
    """Extract a single frame from a video.
    
    Args:
        video_path: Path to video file
        timestamp_sec: Time position in seconds
        output_path: Optional path to save the frame
    
    Returns:
        PIL Image of the frame
    """
    # Create temp file for frame
    cleanup = False
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
        cleanup = True
    
    # Use FFmpeg to extract frame
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(timestamp_sec),  # Seek to position
        '-i', str(video_path),
        '-frames:v', '1',           # Extract 1 frame
        '-q:v', '2',                # High quality
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        image = Image.open(output_path).copy()  # Copy to release file handle
        return image
    finally:
        if cleanup and output_path.exists():
            output_path.unlink()


def export_frame(
    frame: Frame,
    output_folder: Path,
    preset: str,
    quality: int = JPEG_QUALITY,
) -> Path:
    """Export a frame with a specific crop preset.
    
    Args:
        frame: Frame object to export
        output_folder: Base output folder
        preset: Crop preset name
        quality: JPEG quality
        
    Returns:
        Path to saved file
    """
    # Extract the frame
    image = extract_frame(frame.video.path, frame.timestamp_sec)
    
    # Apply crop
    cropped = crop_to_preset(image, preset, frame.faces)
    
    # Generate filename
    # Format: videoname_timestamp_category_preset.jpg
    filename = (
        f"{frame.video.stem}_"
        f"{frame.timestamp_sec:.1f}s_"
        f"{frame.category.name.lower()}_"
        f"{preset}.jpg"
    )
    
    # Organize by category
    category_folder = output_folder / frame.category.name.lower()
    output_path = category_folder / filename
    
    return export_screenshot(cropped, output_path, quality)


def export_frame_all_crops(
    frame: Frame,
    output_folder: Path,
    presets: list[str] = None,
    quality: int = JPEG_QUALITY,
) -> list[Path]:
    """Export a frame with all crop presets.
    
    Args:
        frame: Frame object to export
        output_folder: Base output folder
        presets: List of preset names (None = all)
        quality: JPEG quality
        
    Returns:
        List of paths to saved files
    """
    # Extract the frame once
    image = extract_frame(frame.video.path, frame.timestamp_sec)
    
    # Get all crops
    cropped_images = crop_all_presets(image, frame.faces, presets)
    
    paths = []
    for preset, cropped in cropped_images.items():
        filename = (
            f"{frame.video.stem}_"
            f"{frame.timestamp_sec:.1f}s_"
            f"{frame.category.name.lower()}_"
            f"{preset}.jpg"
        )
        
        category_folder = output_folder / frame.category.name.lower()
        output_path = category_folder / filename
        
        export_screenshot(cropped, output_path, quality)
        paths.append(output_path)
    
    return paths


def export_frames_batch(
    frames: list[Frame],
    output_folder: Path,
    presets: list[str] = None,
    quality: int = JPEG_QUALITY,
    progress_callback=None,
) -> list[Path]:
    """Export multiple frames with progress tracking.
    
    Args:
        frames: List of Frame objects
        output_folder: Base output folder
        presets: List of preset names
        quality: JPEG quality
        progress_callback: Optional callback(current, total, path)
        
    Returns:
        List of all saved paths
    """
    all_paths = []
    total = len(frames)
    
    for i, frame in enumerate(frames):
        try:
            paths = export_frame_all_crops(frame, output_folder, presets, quality)
            all_paths.extend(paths)
            
            if progress_callback:
                progress_callback(i + 1, total, paths[0] if paths else None)
                
        except Exception as e:
            logger.error(f"Failed to export frame {frame.timecode}: {e}")
            continue
    
    logger.info(f"Exported {len(all_paths)} screenshots from {total} frames")
    return all_paths
