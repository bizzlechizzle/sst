"""Video clip export functions."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from ..core.config import VIDEO_CRF, DEFAULT_LOG_LUT, LUT_PRESETS
from ..core.models import ClipSuggestion, Video, BoundingBox, LogProfile
from ..crops.presets import CROP_PRESETS
from ..crops.smart_crop import calculate_crop_box

logger = logging.getLogger(__name__)


def get_lut_for_video(video: Video) -> Optional[Path]:
    """Get the appropriate LUT for a video based on its LOG profile.

    Args:
        video: Video object with log_profile attribute

    Returns:
        Path to LUT file if LOG footage, None otherwise
    """
    if not hasattr(video, 'log_profile') or video.log_profile == LogProfile.NONE:
        return None

    # Check for profile-specific LUT first
    profile_map = {
        LogProfile.SLOG3: 'slog3',
        LogProfile.CLOG3: 'clog3',
        LogProfile.VLOG: 'vlog',
        LogProfile.NLOG: 'nlog',
        LogProfile.FLAT: None,  # Use default
    }

    profile_key = profile_map.get(video.log_profile)
    if profile_key and LUT_PRESETS.get(profile_key):
        lut_path = Path(LUT_PRESETS[profile_key])
        if lut_path.exists():
            logger.debug(f"Using profile-specific LUT: {lut_path}")
            return lut_path

    # Fall back to default LOG LUT
    if DEFAULT_LOG_LUT.exists():
        logger.debug(f"Using default LOG LUT: {DEFAULT_LOG_LUT}")
        return DEFAULT_LOG_LUT

    logger.warning(f"No LUT found for {video.log_profile}, exporting without color correction")
    return None


def export_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    crop_box: BoundingBox = None,
    target_width: int = None,
    target_height: int = None,
    source_width: int = None,
    source_height: int = None,
    crf: int = VIDEO_CRF,
    lut_path: Path = None,
) -> Path:
    """Export a video clip.

    Args:
        video_path: Source video path
        output_path: Where to save clip
        start_sec: Start time in seconds
        end_sec: End time in seconds
        crop_box: Optional crop box coordinates
        target_width: Target output width (for scaling)
        target_height: Target output height (for scaling)
        source_width: Source video width (for anamorphic detection)
        source_height: Source video height (for anamorphic detection)
        crf: Quality setting (lower = better, larger file)
        lut_path: Optional LUT file to apply

    Returns:
        Path to exported clip
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_sec - start_sec

    # Build filter chain
    filters = []

    # HDV anamorphic correction: 1440x1080 should display as 1920x1080 (4:3 SAR → 16:9 DAR)
    # This must come FIRST before any cropping or scaling
    if source_width == 1440 and source_height == 1080:
        logger.debug("Applying HDV anamorphic stretch: 1440x1080 → 1920x1080")
        filters.append("scale=1920:1080:flags=lanczos,setsar=1:1")

    # Crop if specified
    if crop_box:
        filters.append(f"crop={crop_box.width}:{crop_box.height}:{crop_box.x}:{crop_box.y}")
    
    # Apply LUT if specified
    if lut_path and lut_path.exists():
        filters.append(f"lut3d='{lut_path}'")
    
    # Scale if target dimensions specified
    if target_width and target_height:
        filters.append(f"scale={target_width}:{target_height}:flags=lanczos")
    else:
        # Default: scale to 1080p max width
        filters.append("scale='min(1920,iw)':-2:flags=lanczos")
    
    # Sharpen slightly
    filters.append("unsharp=5:5:0.5:5:5:0.5")
    
    # Build FFmpeg command
    # Force yuv420p and High profile for QuickTime/browser compatibility
    # (source 10-bit video would otherwise encode as High 10 profile)
    # Level 4.0 ensures broad device compatibility (including TOD/HDV sources)
    # faststart moves moov atom to beginning for streaming/web playback
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', str(video_path),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-level', '4.0',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', str(crf),
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
    ]
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    cmd.append(str(output_path))
    
    logger.debug(f"Exporting clip: {start_sec:.1f}s - {end_sec:.1f}s")
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg clip export failed: {e.stderr.decode()}")
        raise
    
    logger.info(f"Exported clip: {output_path}")
    return output_path


def export_clip_with_preset(
    video: Video,
    start_sec: float,
    end_sec: float,
    output_folder: Path,
    preset: str,
    faces: list = None,
    crf: int = VIDEO_CRF,
    lut_path: Path = None,
) -> Path:
    """Export a clip with a specific crop preset.

    Args:
        video: Video object
        start_sec: Start time
        end_sec: End time
        output_folder: Output folder
        preset: Crop preset name
        faces: Detected faces for smart cropping
        crf: Quality setting
        lut_path: Optional LUT file (auto-detected from video LOG profile if None)

    Returns:
        Path to exported clip
    """
    if preset not in CROP_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    # Auto-detect LUT for LOG footage if not explicitly provided
    if lut_path is None:
        lut_path = get_lut_for_video(video)

    target_width, target_height = CROP_PRESETS[preset]

    # Get source dimensions, accounting for HDV anamorphic stretch
    # HDV 1440x1080 will be stretched to 1920x1080 before cropping
    source_width = video.metadata.width
    source_height = video.metadata.height
    effective_width = source_width
    effective_height = source_height

    if source_width == 1440 and source_height == 1080:
        effective_width = 1920  # Post-stretch dimensions for crop calculation

    # Calculate crop box using effective (post-stretch) dimensions
    crop_box = calculate_crop_box(
        effective_width,
        effective_height,
        target_width,
        target_height,
        faces=faces,
        bias='left' if preset == 'story_l' else ('right' if preset == 'story_r' else 'center'),
    )
    
    # Generate filename
    filename = f"{video.stem}_{start_sec:.1f}-{end_sec:.1f}s_{preset}.mp4"
    output_path = output_folder / filename
    
    return export_clip(
        video.path,
        output_path,
        start_sec,
        end_sec,
        crop_box=crop_box,
        target_width=target_width,
        target_height=target_height,
        source_width=video.metadata.width,
        source_height=video.metadata.height,
        crf=crf,
        lut_path=lut_path,
    )


def export_clip_suggestion(
    suggestion: ClipSuggestion,
    output_folder: Path,
    preset: str = 'landscape',
    faces: list = None,
    crf: int = VIDEO_CRF,
    lut_path: Path = None,
) -> Path:
    """Export a ClipSuggestion to a file.
    
    Args:
        suggestion: The clip to export
        output_folder: Where to save
        preset: Crop preset to use
        faces: Detected faces for smart cropping
        crf: Quality setting
        lut_path: Optional LUT file
    
    Returns:
        Path to exported clip
    """
    # Organize by category
    category_folder = output_folder / suggestion.category.name.lower()
    
    return export_clip_with_preset(
        suggestion.video,
        suggestion.start_sec,
        suggestion.end_sec,
        category_folder,
        preset,
        faces=faces,
        crf=crf,
        lut_path=lut_path,
    )


def export_clips_batch(
    suggestions: list[ClipSuggestion],
    output_folder: Path,
    presets: list[str] = None,
    crf: int = VIDEO_CRF,
    lut_path: Path = None,
    progress_callback=None,
) -> list[Path]:
    """Export multiple clips with progress tracking.
    
    Args:
        suggestions: List of ClipSuggestion objects
        output_folder: Base output folder
        presets: List of preset names (default: ['landscape'])
        crf: Quality setting
        lut_path: Optional LUT file
        progress_callback: Optional callback(current, total, path)
        
    Returns:
        List of all saved paths
    """
    if presets is None:
        presets = ['landscape']
    
    all_paths = []
    total = len(suggestions) * len(presets)
    current = 0
    
    for suggestion in suggestions:
        for preset in presets:
            try:
                path = export_clip_suggestion(
                    suggestion,
                    output_folder,
                    preset,
                    crf=crf,
                    lut_path=lut_path,
                )
                all_paths.append(path)
                current += 1
                
                if progress_callback:
                    progress_callback(current, total, path)
                    
            except Exception as e:
                logger.error(f"Failed to export clip: {e}")
                current += 1
                continue
    
    logger.info(f"Exported {len(all_paths)} clips from {len(suggestions)} suggestions")
    return all_paths
