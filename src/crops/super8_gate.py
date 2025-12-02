"""Super 8 full gate scan content region detection.

Super 8 full gate scans typically include:
- Black borders on left/right sides (film edge)
- Sprocket holes (black rectangles on one edge)
- Adjacent frame content at top/bottom (frame line visible)
- The actual frame content in the center

This module detects the single center frame by finding:
1. Left/right dark borders (sprocket edge and film edge)
2. Top/bottom frame lines (thin dark lines separating frames)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..core.models import BoundingBox

logger = logging.getLogger(__name__)


def _find_dark_valleys(profile: np.ndarray, dark_threshold: float = 50) -> list[int]:
    """Find positions where the profile dips into darkness (frame lines)."""
    valleys = []
    in_valley = False
    valley_start = 0

    for i, val in enumerate(profile):
        if val < dark_threshold and not in_valley:
            in_valley = True
            valley_start = i
        elif val >= dark_threshold and in_valley:
            in_valley = False
            # Record center of valley
            valleys.append((valley_start + i) // 2)

    return valleys


def detect_super8_frame(
    image: Image.Image | np.ndarray,
    dark_threshold: int = 40,
    min_valley_width: int = 5,
) -> Optional[BoundingBox]:
    """Detect single frame region in a Super 8 full gate scan.

    Uses brightness profiles to find:
    - Left border: sprocket hole area (dark on left edge)
    - Right border: film edge (dark on right)
    - Top/bottom: frame lines (local minima in vertical profile)

    Args:
        image: PIL Image or numpy array of the frame
        dark_threshold: Pixel values below this are "dark"
        min_valley_width: Minimum width for a valid frame line

    Returns:
        BoundingBox of the center frame, or None if detection fails
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Convert to grayscale if color
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    h, w = gray.shape

    # --- Find horizontal boundaries (left/right) ---
    # Use column-wise mean brightness
    h_profile = np.mean(gray, axis=0)

    # Find left edge: first column with mean > threshold
    left = 0
    for i in range(w):
        if h_profile[i] > dark_threshold:
            left = i
            break

    # Find right edge: last column with mean > threshold
    right = w
    for i in range(w - 1, -1, -1):
        if h_profile[i] > dark_threshold:
            right = i
            break

    # --- Find vertical boundaries using local minima ---
    # Use row-wise mean brightness in the content area only
    content_gray = gray[:, left:right]
    v_profile = np.mean(content_gray, axis=1)

    center_y = h // 2

    # Find local minima by looking for valleys in the brightness profile
    # Super 8 frame lines are thin dark bands between bright frame content

    # Search for top frame line (from center upward)
    # Look for where brightness drops significantly below neighbors
    top = 0
    min_brightness_top = float('inf')
    min_pos_top = 0

    # Search in the upper half but not too close to center or edge
    search_start = int(h * 0.05)  # Start 5% from top
    search_end = int(h * 0.35)    # End 35% from top (before center)

    for i in range(search_start, search_end):
        # Look for local minimum with significant depth
        window = 50  # Look at neighbors within this range
        local_mean = np.mean(v_profile[max(0, i-window):min(h, i+window)])
        if v_profile[i] < min_brightness_top and v_profile[i] < local_mean - 5:
            min_brightness_top = v_profile[i]
            min_pos_top = i

    if min_pos_top > 0:
        top = min_pos_top + 20  # Add margin after the frame line

    # Search for bottom frame line (from center downward)
    min_brightness_bottom = float('inf')
    min_pos_bottom = h

    search_start = int(h * 0.65)  # Start 65% from top (after center)
    search_end = int(h * 0.95)    # End 95% from top

    for i in range(search_start, search_end):
        window = 50
        local_mean = np.mean(v_profile[max(0, i-window):min(h, i+window)])
        if v_profile[i] < min_brightness_bottom and v_profile[i] < local_mean - 5:
            min_brightness_bottom = v_profile[i]
            min_pos_bottom = i

    if min_pos_bottom < h:
        bottom = min_pos_bottom - 20  # Add margin before the frame line

    # Add margins to exclude edge artifacts
    margin_x = max(10, int((right - left) * 0.01))

    left = left + margin_x
    right = right - margin_x

    content_width = right - left
    content_height = bottom - top

    if content_width <= 0 or content_height <= 0:
        logger.warning("Invalid Super 8 frame dimensions")
        return None

    # Validate reasonable aspect ratio for Super 8 (roughly 4:3 or 1.33:1)
    aspect = content_width / content_height
    if aspect < 0.8 or aspect > 2.0:
        logger.warning(f"Unusual aspect ratio {aspect:.2f}, detection may have failed")

    logger.debug(f"Detected Super 8 frame: {left},{top} {content_width}x{content_height}")

    return BoundingBox(x=left, y=top, width=content_width, height=content_height)


def detect_content_region(
    image: Image.Image | np.ndarray,
    dark_threshold: int = 30,
    min_content_ratio: float = 0.3,
) -> Optional[BoundingBox]:
    """Fallback detection using contour finding.

    Args:
        image: PIL Image or numpy array of the frame
        dark_threshold: Pixel values below this are considered "dark" (border)
        min_content_ratio: Minimum ratio of content area to total area

    Returns:
        BoundingBox of the content region, or None if detection fails
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Convert to grayscale if color
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    h, w = gray.shape

    # Create binary mask: bright (content) = 255, dark (border) = 0
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No content regions found in Super 8 frame")
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, cw, ch = cv2.boundingRect(largest_contour)

    # Validate: content should be at least min_content_ratio of frame
    content_area = cw * ch
    total_area = w * h
    if content_area / total_area < min_content_ratio:
        logger.warning(f"Content region too small: {content_area/total_area:.1%}")
        return None

    logger.debug(f"Detected Super 8 content region: {x},{y} {cw}x{ch}")

    return BoundingBox(x=x, y=y, width=cw, height=ch)


def detect_content_region_robust(
    image: Image.Image | np.ndarray,
    sample_rows: int = 100,
) -> Optional[BoundingBox]:
    """More robust content detection using edge analysis.

    This method analyzes horizontal and vertical profiles to find
    where content begins/ends, which is more reliable for varying
    exposure levels.

    Args:
        image: PIL Image or numpy array
        sample_rows: Number of rows to sample for edge detection

    Returns:
        BoundingBox of content region, or None if detection fails
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    h, w = gray.shape

    # Compute horizontal profile (mean brightness per column)
    h_profile = np.mean(gray, axis=0)

    # Compute vertical profile (mean brightness per row)
    v_profile = np.mean(gray, axis=1)

    # Find content boundaries using gradient
    # Left edge: where brightness increases significantly
    h_gradient = np.gradient(h_profile)
    v_gradient = np.gradient(v_profile)

    # Threshold for detecting edges
    h_threshold = np.std(h_gradient) * 2
    v_threshold = np.std(v_gradient) * 2

    # Find left boundary (first major increase)
    left_edges = np.where(h_gradient > h_threshold)[0]
    left = left_edges[0] if len(left_edges) > 0 else 0

    # Find right boundary (last major decrease)
    right_edges = np.where(h_gradient < -h_threshold)[0]
    right = right_edges[-1] if len(right_edges) > 0 else w

    # Find top boundary
    top_edges = np.where(v_gradient > v_threshold)[0]
    top = top_edges[0] if len(top_edges) > 0 else 0

    # Find bottom boundary
    bottom_edges = np.where(v_gradient < -v_threshold)[0]
    bottom = bottom_edges[-1] if len(bottom_edges) > 0 else h

    # Add small margin to avoid edge artifacts
    margin_x = int(w * 0.005)  # 0.5% margin
    margin_y = int(h * 0.005)

    left = max(0, left + margin_x)
    right = min(w, right - margin_x)
    top = max(0, top + margin_y)
    bottom = min(h, bottom - margin_y)

    content_width = right - left
    content_height = bottom - top

    if content_width <= 0 or content_height <= 0:
        logger.warning("Invalid content region dimensions")
        return None

    logger.debug(f"Robust detection: {left},{top} {content_width}x{content_height}")

    return BoundingBox(x=left, y=top, width=content_width, height=content_height)


def detect_super8_gate(
    image: Image.Image | np.ndarray,
    method: str = 'hybrid',
) -> Optional[BoundingBox]:
    """Main entry point for Super 8 gate detection.

    Tries multiple methods and returns the best result.

    Args:
        image: Frame image
        method: 'frame' (new), 'threshold', 'gradient', or 'hybrid' (tries all)

    Returns:
        BoundingBox of content region
    """
    if method == 'frame':
        return detect_super8_frame(image)
    elif method == 'threshold':
        return detect_content_region(image)
    elif method == 'gradient':
        return detect_content_region_robust(image)
    else:  # hybrid - try frame detection first (best for full gate scans)
        # Try specialized Super 8 frame detection first
        result = detect_super8_frame(image)
        if result is not None:
            return result

        # Fall back to threshold method
        result = detect_content_region(image)
        if result is not None:
            return result

        # Final fallback to gradient method
        return detect_content_region_robust(image)


def crop_to_content(
    image: Image.Image,
    content_box: BoundingBox = None,
) -> Image.Image:
    """Crop image to content region.

    Args:
        image: Source image
        content_box: Pre-computed content box, or None to auto-detect

    Returns:
        Cropped image containing only the film content
    """
    if content_box is None:
        content_box = detect_super8_gate(image)

    if content_box is None:
        logger.warning("Could not detect content region, returning original")
        return image

    return image.crop((
        content_box.x,
        content_box.y,
        content_box.x + content_box.width,
        content_box.y + content_box.height,
    ))
