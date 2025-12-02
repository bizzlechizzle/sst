"""LUT (Look-Up Table) handling."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..core.models import LogProfile
from ..core.config import LUT_PRESETS

logger = logging.getLogger(__name__)


def get_lut_for_profile(profile: LogProfile) -> Optional[Path]:
    """Get the LUT file path for a given LOG profile.
    
    Args:
        profile: The LogProfile enum value
        
    Returns:
        Path to LUT file or None if not configured/needed
    """
    if profile == LogProfile.NONE:
        return None
    
    # Map profile to preset key
    profile_map = {
        LogProfile.SLOG3: 'slog3',
        LogProfile.CLOG3: 'clog3',
        LogProfile.VLOG: 'vlog',
        LogProfile.NLOG: 'nlog',
        LogProfile.FLAT: None,  # No standard LUT for generic flat
    }
    
    key = profile_map.get(profile)
    if key is None:
        return None
    
    lut_path = LUT_PRESETS.get(key)
    if lut_path is None:
        logger.warning(f"No LUT configured for {profile.name}")
        return None
    
    lut_path = Path(lut_path)
    if not lut_path.exists():
        logger.warning(f"LUT file not found: {lut_path}")
        return None
    
    return lut_path


def apply_lut(
    input_path: Path,
    output_path: Path,
    lut_path: Path,
) -> Path:
    """Apply a LUT to an image file.
    
    Args:
        input_path: Source image path
        output_path: Destination path
        lut_path: Path to .cube LUT file
        
    Returns:
        Path to output file
        
    Raises:
        FileNotFoundError: If input or LUT not found
        RuntimeError: If FFmpeg fails
    """
    import subprocess
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', f"lut3d='{lut_path}'",
        '-q:v', '2',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg LUT application failed: {e.stderr.decode()}")
    
    logger.debug(f"Applied LUT {lut_path.name} to {input_path.name}")
    return output_path


def validate_lut_file(lut_path: Path) -> bool:
    """Check if a LUT file is valid.
    
    Args:
        lut_path: Path to .cube file
        
    Returns:
        True if file appears to be a valid .cube LUT
    """
    if not lut_path.exists():
        return False
    
    if lut_path.suffix.lower() != '.cube':
        logger.warning(f"LUT file should be .cube format: {lut_path}")
        return False
    
    try:
        with open(lut_path, 'r') as f:
            # Check first few lines for .cube format markers
            content = f.read(1000)
            
            # .cube files should have LUT_3D_SIZE or TITLE
            if 'LUT_3D_SIZE' in content or 'TITLE' in content:
                return True
            
            logger.warning(f"LUT file missing expected headers: {lut_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error reading LUT file: {e}")
        return False


def list_available_luts(lut_folder: Path) -> list[Path]:
    """Find all .cube files in a folder.
    
    Args:
        lut_folder: Folder to search
        
    Returns:
        List of paths to .cube files
    """
    if not lut_folder.exists():
        return []
    
    luts = list(lut_folder.glob('*.cube'))
    luts.extend(lut_folder.glob('*.CUBE'))
    
    # Validate each one
    valid_luts = [lut for lut in luts if validate_lut_file(lut)]
    
    logger.debug(f"Found {len(valid_luts)} valid LUTs in {lut_folder}")
    return sorted(valid_luts, key=lambda p: p.name.lower())
