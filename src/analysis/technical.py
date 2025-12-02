"""Technical quality scoring (blur, exposure, etc.)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TechnicalScorer:
    """Scores technical quality of images (sharpness, exposure, etc.)."""
    
    def __init__(self):
        self._cv2 = None
    
    def _load_cv2(self):
        """Lazy-load OpenCV."""
        if self._cv2 is not None:
            return
        
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.warning("OpenCV not installed, technical scoring limited")
    
    def score_sharpness(self, image: Image.Image) -> float:
        """Score image sharpness using Laplacian variance.
        
        Higher variance = sharper image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Sharpness score from 0.0 to 1.0
        """
        self._load_cv2()
        
        if self._cv2 is None:
            return 0.5  # Default if no OpenCV
        
        # Convert to grayscale numpy array
        gray = np.array(image.convert('L'))
        
        # Compute Laplacian variance
        laplacian = self._cv2.Laplacian(gray, self._cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range
        # Typical variance ranges from 0 (very blurry) to 1000+ (very sharp)
        # Use sigmoid-like normalization
        normalized = 1 - (1 / (1 + variance / 500))
        
        logger.debug(f"Sharpness: {normalized:.3f} (variance={variance:.1f})")
        return float(normalized)
    
    def score_exposure(self, image: Image.Image) -> float:
        """Score image exposure quality.
        
        Checks for under/over exposure by analyzing histogram.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Exposure score from 0.0 to 1.0 (1.0 = well exposed)
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))
        
        # Calculate histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()  # Normalize
        
        # Check for under-exposure (too many dark pixels)
        dark_ratio = hist[:50].sum()  # Pixels in 0-50 range
        
        # Check for over-exposure (too many bright pixels)
        bright_ratio = hist[205:].sum()  # Pixels in 205-255 range
        
        # Penalize extreme exposures
        under_penalty = max(0, dark_ratio - 0.3) * 2  # Penalty if >30% dark
        over_penalty = max(0, bright_ratio - 0.2) * 2  # Penalty if >20% bright
        
        score = 1.0 - under_penalty - over_penalty
        score = max(0.0, min(1.0, score))
        
        logger.debug(f"Exposure: {score:.3f} (dark={dark_ratio:.2f}, bright={bright_ratio:.2f})")
        return score
    
    def score_contrast(self, image: Image.Image) -> float:
        """Score image contrast.
        
        Good images have a wide tonal range.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Contrast score from 0.0 to 1.0
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))
        
        # Calculate standard deviation (measure of contrast)
        std = gray.std()
        
        # Typical std ranges from 0 (flat) to ~80 (high contrast)
        # Ideal is around 50-70
        if std < 30:
            score = std / 30 * 0.5  # Low contrast penalty
        elif std > 80:
            score = 1.0 - (std - 80) / 80 * 0.3  # Very high contrast slight penalty
        else:
            score = 0.5 + (std - 30) / 50 * 0.5  # Good range
        
        score = max(0.0, min(1.0, score))
        
        logger.debug(f"Contrast: {score:.3f} (std={std:.1f})")
        return score
    
    def score_noise(self, image: Image.Image) -> float:
        """Estimate image noise level.
        
        Lower noise = higher score.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Noise score from 0.0 to 1.0 (1.0 = low noise)
        """
        self._load_cv2()
        
        if self._cv2 is None:
            return 0.5
        
        # Convert to grayscale
        gray = np.array(image.convert('L')).astype(float)
        
        # Estimate noise using median absolute deviation
        # Apply high-pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = self._cv2.filter2D(gray, -1, kernel)
        
        # MAD-based noise estimate
        noise_estimate = np.median(np.abs(filtered)) / 0.6745
        
        # Normalize (typical noise ranges 0-20)
        score = 1.0 - min(1.0, noise_estimate / 30)
        
        logger.debug(f"Noise: {score:.3f} (estimate={noise_estimate:.1f})")
        return float(score)
    
    def score_composition(self, image: Image.Image, faces: list = None) -> float:
        """Score image composition using rule of thirds.
        
        Checks if key elements (faces) are positioned at power points.
        
        Args:
            image: PIL Image to analyze
            faces: Optional list of Face objects
            
        Returns:
            Composition score from 0.0 to 1.0
        """
        width, height = image.size
        
        # Power points (rule of thirds intersections)
        power_points = [
            (width / 3, height / 3),
            (2 * width / 3, height / 3),
            (width / 3, 2 * height / 3),
            (2 * width / 3, 2 * height / 3),
        ]
        
        if not faces:
            # Without faces, give neutral score
            return 0.5
        
        # Calculate distance from each face center to nearest power point
        total_score = 0.0
        
        for face in faces:
            face_center = (face.bbox.center_x, face.bbox.center_y)
            
            # Find minimum distance to any power point
            min_dist = float('inf')
            for pp in power_points:
                dist = np.sqrt((face_center[0] - pp[0])**2 + (face_center[1] - pp[1])**2)
                min_dist = min(min_dist, dist)
            
            # Normalize distance (closer = better)
            max_dist = np.sqrt(width**2 + height**2) / 3
            face_score = 1.0 - min(1.0, min_dist / max_dist)
            total_score += face_score
        
        score = total_score / len(faces) if faces else 0.5
        
        logger.debug(f"Composition: {score:.3f}")
        return score
    
    def score_all(self, image: Image.Image, faces: list = None) -> dict[str, float]:
        """Calculate all technical scores.
        
        Args:
            image: PIL Image to analyze
            faces: Optional list of Face objects
            
        Returns:
            Dictionary of score name to value
        """
        return {
            'sharpness': self.score_sharpness(image),
            'exposure': self.score_exposure(image),
            'contrast': self.score_contrast(image),
            'noise': self.score_noise(image),
            'composition': self.score_composition(image, faces),
        }
    
    def score_combined(self, image: Image.Image, faces: list = None,
                       weights: dict = None) -> float:
        """Calculate weighted combined technical score.
        
        Args:
            image: PIL Image to analyze
            faces: Optional list of Face objects
            weights: Optional weight dictionary
            
        Returns:
            Combined score from 0.0 to 1.0
        """
        if weights is None:
            weights = {
                'sharpness': 0.30,
                'exposure': 0.25,
                'contrast': 0.15,
                'noise': 0.15,
                'composition': 0.15,
            }
        
        scores = self.score_all(image, faces)
        
        combined = sum(scores[k] * weights.get(k, 0.2) for k in scores)
        return combined
