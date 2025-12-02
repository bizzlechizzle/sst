"""Configuration constants for SST."""
from __future__ import annotations
from pathlib import Path

# === VIDEO FILE EXTENSIONS ===
# IMPORTANT: Include .tod and .mod for JVC cameras!
VIDEO_EXTENSIONS = {
    # Common formats
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
    # MPEG formats
    '.mpg', '.mpeg', '.mts', '.m2ts',
    # JVC cameras - CRITICAL, often forgotten!
    '.tod', '.mod',
    # Mobile formats
    '.3gp', '.3g2',
    # Professional formats
    '.mxf',
    # Legacy formats
    '.dv', '.vob',
}

# === CROP DIMENSIONS ===
# Format: (width, height)
CROP_PRESETS = {
    'square':    (1080, 1080),   # Instagram feed
    'portrait':  (1080, 1350),   # Instagram feed (4:5)
    'landscape': (1920, 1080),   # YouTube/general (16:9)
    'story':     (1080, 1920),   # Stories (9:16)
    'story_l':   (1080, 1920),   # Story - subject on LEFT
    'story_r':   (1080, 1920),   # Story - subject on RIGHT
}

# === SCORING WEIGHTS ===
# How much each factor matters for each category
# All weights should add up to 1.0
CATEGORY_WEIGHTS = {
    'PEOPLE': {
        'face_presence': 0.35,
        'aesthetic': 0.25,
        'composition': 0.20,
        'sharpness': 0.20,
    },
    'B_ROLL': {
        'aesthetic': 0.35,
        'composition': 0.30,
        'sharpness': 0.20,
        'contrast': 0.15,
        # Note: Face penalty applied in pipeline.py
    },
    'ARTSY': {
        'aesthetic': 0.50,
        'composition': 0.25,
        'blur_bonus': 0.15,  # Low sharpness is intentional
        'contrast': 0.10,
    },
}

# === DEFAULT QUOTAS ===
DEFAULT_QUOTAS = {
    'people_screenshots': 50,
    'people_clips': 10,
    'broll_screenshots': 30,
    'broll_clips': 5,
    'artsy_screenshots': 20,
    'artsy_clips': 5,
}

# === QUALITY SETTINGS ===
JPEG_QUALITY = 90  # 1-100, higher = better quality, larger file
VIDEO_CRF = 22     # 0-51, lower = better quality, larger file

# === CLIP DURATIONS ===
# Per original spec: Single (2-18s) vs Multi (8-58s)
CLIP_TYPES = {
    'single': {
        'min': 2.0,
        'max': 18.0,
        'default': 8.0,
        'description': 'Quick moment (2-18 seconds)'
    },
    'multi': {
        'min': 8.0,
        'max': 58.0,
        'default': 30.0,
        'description': 'Extended sequence (8-58 seconds)'
    }
}
CLIP_DURATION_SHORT = 8.0   # seconds - for quick moments
CLIP_DURATION_LONG = 30.0   # seconds - for speeches, dances

# === RESOLUTION TARGETS ===
# Per original spec: 1080px max, 2048 for Facebook
RESOLUTION_PRESETS = {
    'instagram': {
        'max_dimension': 1080,
        'description': 'Instagram optimal'
    },
    'facebook': {
        'max_dimension': 2048,
        'description': 'Facebook high-res'
    },
    'twitter': {
        'max_dimension': 1600,
        'description': 'Twitter/X optimal'
    },
    'original': {
        'max_dimension': None,
        'description': 'Keep original size'
    },
}

# === ANALYSIS SETTINGS ===
FRAME_SAMPLE_INTERVAL = 1.0  # seconds between sampled frames
MIN_FACE_CONFIDENCE = 0.5    # minimum confidence for face detection
MIN_CATEGORY_CONFIDENCE = 0.3  # minimum confidence for categorization

# === PATHS ===
DEFAULT_CONFIG_DIR = Path.home() / '.sst'
DEFAULT_LOG_DIR = DEFAULT_CONFIG_DIR / 'logs'
DEFAULT_CACHE_DIR = DEFAULT_CONFIG_DIR / 'cache'

# === LUT PRESETS ===
# Default LUT for all LOG footage (Fuji FLog2 WDR - works well for most LOG profiles)
# Located in resources/luts/default_log.3dlut
DEFAULT_LOG_LUT = Path(__file__).parent.parent.parent / 'resources' / 'luts' / 'default_log.3dlut'

# Profile-specific LUTs (optional overrides)
LUT_PRESETS = {
    'slog3': None,  # Path to S-Log3 to Rec.709 LUT
    'clog3': None,  # Path to C-Log3 to Rec.709 LUT
    'vlog': None,   # Path to V-Log to Rec.709 LUT
    'nlog': None,   # Path to N-Log to Rec.709 LUT
    'flog2': None,  # Path to F-Log2 to Rec.709 LUT (uses default if None)
}
