"""Crop preset definitions."""
from __future__ import annotations

# Format: (width, height)
CROP_PRESETS = {
    'square':    (1080, 1080),   # Instagram feed 1:1
    'portrait':  (1080, 1350),   # Instagram feed 4:5
    'landscape': (1920, 1080),   # YouTube 16:9
    'story':     (1080, 1920),   # Stories 9:16
    'story_l':   (1080, 1920),   # Story - subject left
    'story_r':   (1080, 1920),   # Story - subject right
}


def get_preset_info(preset: str) -> dict:
    """Get detailed info about a preset.
    
    Args:
        preset: Preset name
        
    Returns:
        Dictionary with preset details
        
    Raises:
        ValueError: If preset is unknown
    """
    if preset not in CROP_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CROP_PRESETS.keys())}")
    
    width, height = CROP_PRESETS[preset]
    
    # Determine orientation
    if width == height:
        orientation = 'square'
    elif height > width:
        orientation = 'portrait'
    else:
        orientation = 'landscape'
    
    # Calculate aspect ratio
    from math import gcd
    g = gcd(width, height)
    aspect_w = width // g
    aspect_h = height // g
    
    return {
        'name': preset,
        'width': width,
        'height': height,
        'aspect_ratio': f"{aspect_w}:{aspect_h}",
        'orientation': orientation,
        'pixels': width * height,
    }


def get_all_presets() -> list[str]:
    """Get list of all preset names."""
    return list(CROP_PRESETS.keys())


def get_presets_by_orientation(orientation: str) -> list[str]:
    """Get presets filtered by orientation.
    
    Args:
        orientation: 'square', 'portrait', or 'landscape'
        
    Returns:
        List of preset names matching orientation
    """
    result = []
    for name in CROP_PRESETS:
        info = get_preset_info(name)
        if info['orientation'] == orientation:
            result.append(name)
    return result
