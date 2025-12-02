"""Aesthetic quality scoring."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from ..core.exceptions import ModelLoadError
from ..core.config import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


class AestheticScorer:
    """Scores images for visual aesthetics using LAION predictor."""
    
    WEIGHTS_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    
    def __init__(self):
        self._model = None
        self._clip_model = None
        self._preprocess = None
        self._device = None
        self._initialized = False
    
    def _load_model(self):
        """Load CLIP and aesthetic predictor."""
        if self._initialized:
            return
        
        logger.info("Loading aesthetic scorer...")
        
        try:
            import open_clip
            
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load CLIP for image embeddings
            self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained='laion2b_s32b_b82k',
                device=self._device,
            )
            
            # Load aesthetic predictor (MLP that takes CLIP embeddings)
            self._model = self._create_aesthetic_model()
            self._load_aesthetic_weights()
            self._initialized = True
            
            logger.info("Aesthetic scorer loaded successfully")
            
        except ImportError:
            raise ModelLoadError(
                "open_clip not installed. Run: pip install open-clip-torch"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load aesthetic scorer: {e}")
    
    def _create_aesthetic_model(self) -> nn.Module:
        """Create the aesthetic prediction MLP.
        
        This is the same architecture used by the LAION aesthetic predictor.
        """
        model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        return model.to(self._device)
    
    def _load_aesthetic_weights(self):
        """Load pre-trained weights for the aesthetic predictor."""
        cache_dir = DEFAULT_CACHE_DIR / 'aesthetic'
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / 'aesthetic_predictor.pth'

        if not weights_path.exists():
            logger.info(f"Downloading aesthetic predictor weights to {weights_path}...")
            try:
                urllib.request.urlretrieve(self.WEIGHTS_URL, weights_path)
            except Exception as e:
                raise ModelLoadError(f"Failed to download aesthetic weights: {e}")

        # Load and remap keys if needed (weights may have 'layers.' prefix)
        state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)

        # Check if keys have 'layers.' prefix and remap
        if any(k.startswith('layers.') for k in state_dict.keys()):
            state_dict = {k.replace('layers.', ''): v for k, v in state_dict.items()}

        self._model.load_state_dict(state_dict)
        self._model.eval()
    
    def score(self, image: Image.Image) -> float:
        """Score an image's aesthetic quality.
        
        Args:
            image: PIL Image to score
        
        Returns:
            Score from 1.0 to 10.0 (higher = more aesthetic)
        """
        self._load_model()
        
        # Preprocess and get CLIP embedding
        img_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            image_features = self._clip_model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Predict aesthetic score
            score = self._model(image_features).item()
        
        # Clamp to valid range
        clamped = max(1.0, min(10.0, score))
        
        logger.debug(f"Aesthetic score: {clamped:.2f}")
        return clamped
    
    def score_batch(self, images: list[Image.Image]) -> list[float]:
        """Score multiple images efficiently.
        
        Args:
            images: List of PIL Images to score
            
        Returns:
            List of scores (1.0 to 10.0)
        """
        self._load_model()
        
        if not images:
            return []
        
        img_tensors = torch.stack([
            self._preprocess(img) for img in images
        ]).to(self._device)
        
        with torch.no_grad():
            image_features = self._clip_model.encode_image(img_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            scores = self._model(image_features).squeeze().tolist()
        
        # Handle single image case
        if isinstance(scores, float):
            scores = [scores]
        
        return [max(1.0, min(10.0, s)) for s in scores]
    
    def score_normalized(self, image: Image.Image) -> float:
        """Score an image and normalize to 0.0-1.0 range.
        
        Args:
            image: PIL Image to score
            
        Returns:
            Normalized score from 0.0 to 1.0
        """
        score = self.score(image)
        return (score - 1.0) / 9.0  # Map 1-10 to 0-1
