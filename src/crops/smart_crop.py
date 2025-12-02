"""Face-aware smart cropping."""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from ..core.models import Face, BoundingBox, Medium
from .presets import CROP_PRESETS
from .super8_gate import detect_super8_gate

logger = logging.getLogger(__name__)


def smart_crop(
    image: Image.Image,
    target_width: int,
    target_height: int,
    faces: list[Face] = None,
    bias: str = 'center',
) -> Image.Image:
    """Crop an image while keeping faces in frame.
    
    Args:
        image: Source image
        target_width: Desired output width
        target_height: Desired output height
        faces: List of detected faces (optional)
        bias: Where to position crop if no faces ('center', 'left', 'right')
    
    Returns:
        Cropped and resized image
    """
    if faces is None:
        faces = []
    
    img_width, img_height = image.size
    
    # Calculate the crop box dimensions at source scale
    target_ratio = target_width / target_height
    source_ratio = img_width / img_height
    
    if source_ratio > target_ratio:
        # Source is wider - crop sides
        crop_height = img_height
        crop_width = int(img_height * target_ratio)
    else:
        # Source is taller - crop top/bottom
        crop_width = img_width
        crop_height = int(img_width / target_ratio)
    
    # Find optimal crop position
    if faces:
        crop_x, crop_y = _position_for_faces(
            img_width, img_height,
            crop_width, crop_height,
            faces,
        )
    else:
        crop_x, crop_y = _position_with_bias(
            img_width, img_height,
            crop_width, crop_height,
            bias,
        )
    
    # Perform the crop
    cropped = image.crop((
        crop_x,
        crop_y,
        crop_x + crop_width,
        crop_y + crop_height,
    ))
    
    # Resize to target dimensions
    return cropped.resize((target_width, target_height), Image.LANCZOS)


def _position_for_faces(
    img_width: int,
    img_height: int,
    crop_width: int,
    crop_height: int,
    faces: list[Face],
) -> tuple[int, int]:
    """Find crop position that includes the most/largest faces.
    
    Uses weighted centroid of faces (larger faces have more weight).
    """
    # Calculate the centroid of all faces, weighted by size
    total_weight = sum(f.bbox.area for f in faces)
    if total_weight == 0:
        # Fallback to center
        return (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
        )
    
    weighted_x = sum(f.bbox.center_x * f.bbox.area for f in faces) / total_weight
    weighted_y = sum(f.bbox.center_y * f.bbox.area for f in faces) / total_weight
    
    # Position crop to center on face centroid
    crop_x = int(weighted_x - crop_width / 2)
    crop_y = int(weighted_y - crop_height / 2)
    
    # Clamp to image bounds
    crop_x = max(0, min(crop_x, img_width - crop_width))
    crop_y = max(0, min(crop_y, img_height - crop_height))
    
    logger.debug(f"Face-aware crop at ({crop_x}, {crop_y}) for {len(faces)} faces")
    
    return crop_x, crop_y


def _position_with_bias(
    img_width: int,
    img_height: int,
    crop_width: int,
    crop_height: int,
    bias: str,
) -> tuple[int, int]:
    """Position crop based on bias preference.
    
    Args:
        img_width: Source image width
        img_height: Source image height
        crop_width: Crop box width
        crop_height: Crop box height
        bias: 'center', 'left', or 'right'
        
    Returns:
        Tuple of (crop_x, crop_y)
    """
    # Vertical: always center
    crop_y = (img_height - crop_height) // 2
    
    # Horizontal: based on bias
    if bias == 'left':
        crop_x = 0
    elif bias == 'right':
        crop_x = img_width - crop_width
    else:  # center
        crop_x = (img_width - crop_width) // 2
    
    return crop_x, crop_y


def crop_to_preset(
    image: Image.Image,
    preset: str,
    faces: list[Face] = None,
) -> Image.Image:
    """Crop image to a named preset.
    
    Args:
        image: Source image
        preset: Preset name ('square', 'portrait', 'landscape', 'story', etc.)
        faces: Detected faces (optional)
    
    Returns:
        Cropped image
        
    Raises:
        ValueError: If preset is unknown
    """
    if preset not in CROP_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    
    target_width, target_height = CROP_PRESETS[preset]
    
    # Handle story_l and story_r biases
    if preset == 'story_l':
        bias = 'left'
    elif preset == 'story_r':
        bias = 'right'
    else:
        bias = 'center'
    
    return smart_crop(image, target_width, target_height, faces, bias)


def crop_all_presets(
    image: Image.Image,
    faces: list[Face] = None,
    presets: list[str] = None,
) -> dict[str, Image.Image]:
    """Crop image to multiple presets at once.
    
    Args:
        image: Source image
        faces: Detected faces (optional)
        presets: List of preset names (default: all)
    
    Returns:
        Dict mapping preset name to cropped image
    """
    if presets is None:
        presets = list(CROP_PRESETS.keys())
    
    return {
        preset: crop_to_preset(image, preset, faces)
        for preset in presets
    }


def calculate_crop_box(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    faces: list[Face] = None,
    bias: str = 'center',
) -> BoundingBox:
    """Calculate crop box without actually cropping.
    
    Useful for video cropping where you need the box coordinates.
    
    Args:
        source_width: Source image/video width
        source_height: Source image/video height
        target_width: Target width
        target_height: Target height
        faces: Detected faces (optional)
        bias: Position bias
        
    Returns:
        BoundingBox for the crop
    """
    if faces is None:
        faces = []
    
    # Calculate crop dimensions
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height
    
    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = int(source_height * target_ratio)
    else:
        crop_width = source_width
        crop_height = int(source_width / target_ratio)
    
    # Find position
    if faces:
        crop_x, crop_y = _position_for_faces(
            source_width, source_height,
            crop_width, crop_height,
            faces,
        )
    else:
        crop_x, crop_y = _position_with_bias(
            source_width, source_height,
            crop_width, crop_height,
            bias,
        )
    
    return BoundingBox(
        x=crop_x,
        y=crop_y,
        width=crop_width,
        height=crop_height,
    )


def smart_crop_super8_vertical(
    image: Image.Image,
    target_width: int,
    target_height: int,
    faces: list[Face] = None,
    bias: str = 'center',
) -> Image.Image:
    """Crop a Super 8 full gate scan for vertical (9:16) output.

    For Super 8 full gate scans, vertical crops need special handling:
    1. Detect the single frame region (excluding adjacent frames and sprocket)
    2. Crop within that region for clean vertical output

    For horizontal/square crops, use regular smart_crop to preserve the
    overscan aesthetic.

    Args:
        image: Source Super 8 full gate scan
        target_width: Desired output width
        target_height: Desired output height
        faces: List of detected faces (optional)
        bias: Where to position crop if no faces ('center', 'left', 'right')

    Returns:
        Cropped and resized image from the detected frame region
    """
    if faces is None:
        faces = []

    # Detect the single frame content region
    content_box = detect_super8_gate(image)

    if content_box is None:
        # Fallback to regular smart_crop if detection fails
        logger.warning("Super 8 gate detection failed, using standard crop")
        return smart_crop(image, target_width, target_height, faces, bias)

    # Crop to content region first
    content_image = image.crop((
        content_box.x,
        content_box.y,
        content_box.x + content_box.width,
        content_box.y + content_box.height,
    ))

    logger.debug(f"Super 8 content region: {content_box.width}x{content_box.height}")

    # Adjust face coordinates relative to content region
    adjusted_faces = []
    for face in faces:
        # Check if face is within content region
        if (face.bbox.x >= content_box.x and
            face.bbox.y >= content_box.y and
            face.bbox.x + face.bbox.width <= content_box.x + content_box.width and
            face.bbox.y + face.bbox.height <= content_box.y + content_box.height):
            # Adjust coordinates relative to content region
            adjusted_bbox = BoundingBox(
                x=face.bbox.x - content_box.x,
                y=face.bbox.y - content_box.y,
                width=face.bbox.width,
                height=face.bbox.height,
            )
            adjusted_faces.append(Face(
                bbox=adjusted_bbox,
                confidence=face.confidence,
                landmarks=face.landmarks,
            ))

    # Now do standard smart crop on the content region
    return smart_crop(content_image, target_width, target_height, adjusted_faces, bias)


def is_vertical_crop(target_width: int, target_height: int) -> bool:
    """Check if target dimensions are for a vertical (portrait) crop."""
    return target_height > target_width


def smart_crop_for_medium(
    image: Image.Image,
    target_width: int,
    target_height: int,
    faces: list[Face] = None,
    bias: str = 'center',
    medium: Optional[Medium] = None,
) -> Image.Image:
    """Smart crop with medium-aware handling.

    For Super 8 content with vertical target dimensions, uses gate detection
    to crop from the single frame region. Otherwise uses standard smart crop.

    Args:
        image: Source image
        target_width: Desired output width
        target_height: Desired output height
        faces: List of detected faces (optional)
        bias: Where to position crop if no faces
        medium: Source medium type (optional)

    Returns:
        Cropped and resized image
    """
    # Use Super 8 gate detection for vertical crops from Super 8 content
    if medium == Medium.SUPER_8 and is_vertical_crop(target_width, target_height):
        return smart_crop_super8_vertical(
            image, target_width, target_height, faces, bias
        )

    # Standard smart crop for everything else
    return smart_crop(image, target_width, target_height, faces, bias)
