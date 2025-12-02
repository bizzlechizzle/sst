"""Crop presets and smart cropping."""

from .presets import CROP_PRESETS, get_preset_info
from .smart_crop import smart_crop, crop_to_preset, crop_all_presets

__all__ = [
    'CROP_PRESETS',
    'get_preset_info',
    'smart_crop',
    'crop_to_preset',
    'crop_all_presets',
]
