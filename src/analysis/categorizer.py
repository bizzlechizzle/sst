"""Category classification using CLIP."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ..core.models import Category
from ..core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class CLIPCategorizer:
    """Classifies images into PEOPLE, B_ROLL, or ARTSY using CLIP."""

    # Wedding-specific prompts for each category
    # More prompts = better accuracy through averaging
    PROMPTS = {
        Category.PEOPLE: [
            "a photo of a bride and groom",
            "a wedding portrait of people",
            "wedding guests celebrating",
            "people at a wedding ceremony",
            "a couple dancing at their wedding",
            "a group photo at a wedding",
            "candid photo of people smiling at a wedding",
            "close-up portrait of a person at a wedding",
            "bridesmaids and groomsmen at wedding",
            "family portrait at wedding reception",
        ],
        Category.B_ROLL: [
            # Details
            "wedding rings on a surface",
            "a bouquet of wedding flowers",
            "wedding dress details and lace",
            "wedding table decorations",
            "a wedding cake",
            "wedding invitation and stationery",
            "wedding jewelry and accessories",
            "wedding food and champagne",
            # Venue
            "exterior of a wedding venue building",
            "empty wedding ceremony location",
            "wedding reception hall interior",
            "landscape at a wedding venue",
            "outdoor wedding location scenery",
            "church or chapel interior",
            # Nature / Environment
            "trees and grass at outdoor wedding",
            "sunset at wedding venue",
            "empty aisle with flower petals",
        ],
        Category.ARTSY: [
            "blurry artistic wedding photo with motion blur",
            "film light leak effect over wedding footage",
            "shallow depth of field bokeh lights wedding",
            "grainy vintage film wedding footage",
            "lens flare sunlight at wedding",
            "backlit silhouette of couple at sunset",
            "artistic reflection in mirror or water wedding",
            "abstract creative angle wedding photography",
            "intentionally blurred artistic wedding shot",
            "dreamy soft focus romantic wedding",
        ],
    }
    
    def __init__(self, model_name: str = 'ViT-L-14', pretrained: str = 'laion2b_s32b_b82k'):
        """Initialize the CLIP model.
        
        Args:
            model_name: CLIP architecture (ViT-L-14 is largest/most accurate)
            pretrained: Which pretrained weights to use
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._prompt_embeddings = None
        self._device = None
        self._initialized = False
    
    def _load_model(self):
        """Lazy-load the model."""
        if self._initialized:
            return
        
        logger.info(f"Loading CLIP model {self.model_name}...")
        
        try:
            import open_clip
            
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self._device}")
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self._device,
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Pre-compute prompt embeddings (this speeds up inference)
            self._precompute_prompts()
            self._initialized = True
            
            logger.info("CLIP model loaded successfully")
            
        except ImportError:
            raise ModelLoadError(
                "open_clip not installed. Run: pip install open-clip-torch"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load CLIP: {e}")
    
    def _precompute_prompts(self):
        """Pre-compute text embeddings for all prompts.
        
        Instead of computing text embeddings every time, we do it once
        and reuse them. This makes inference much faster.
        """
        self._prompt_embeddings = {}
        
        with torch.no_grad():
            for category, prompts in self.PROMPTS.items():
                # Tokenize all prompts for this category
                tokens = self._tokenizer(prompts).to(self._device)
                
                # Get text embeddings
                text_features = self._model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Average all prompts for this category
                avg_features = text_features.mean(dim=0)
                avg_features /= avg_features.norm()
                
                self._prompt_embeddings[category] = avg_features
        
        logger.debug("Pre-computed embeddings for all category prompts")
    
    def categorize(self, image: Image.Image) -> Tuple[Category, float]:
        """Classify an image into a category.
        
        Args:
            image: PIL Image to classify
        
        Returns:
            Tuple of (Category, confidence)
            Confidence is 0.0 to 1.0
        """
        self._load_model()
        
        # Preprocess image
        img_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            # Get image embedding
            image_features = self._model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compare to each category's prompts
            similarities = {}
            for category, text_features in self._prompt_embeddings.items():
                similarity = (image_features @ text_features).item()
                similarities[category] = similarity
        
        # Softmax to get probabilities
        values = list(similarities.values())
        exp_values = np.exp(np.array(values) * 100)  # Temperature scaling
        probs = exp_values / exp_values.sum()
        
        # Get best category
        best_idx = np.argmax(probs)
        best_category = list(similarities.keys())[best_idx]
        confidence = float(probs[best_idx])
        
        logger.debug(f"Categorized as {best_category.name} ({confidence:.2%})")
        
        return best_category, confidence
    
    def categorize_batch(self, images: list[Image.Image]) -> list[Tuple[Category, float]]:
        """Classify multiple images efficiently.
        
        Batching is faster than processing one at a time.
        
        Args:
            images: List of PIL Images to classify
            
        Returns:
            List of (Category, confidence) tuples
        """
        self._load_model()
        
        if not images:
            return []
        
        # Preprocess all images
        img_tensors = torch.stack([
            self._preprocess(img) for img in images
        ]).to(self._device)
        
        with torch.no_grad():
            # Get all image embeddings
            image_features = self._model.encode_image(img_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            results = []
            for i in range(len(images)):
                # Compare this image to each category
                similarities = {}
                for category, text_features in self._prompt_embeddings.items():
                    similarity = (image_features[i] @ text_features).item()
                    similarities[category] = similarity
                
                # Softmax
                values = list(similarities.values())
                exp_values = np.exp(np.array(values) * 100)
                probs = exp_values / exp_values.sum()
                
                best_idx = np.argmax(probs)
                best_category = list(similarities.keys())[best_idx]
                confidence = float(probs[best_idx])
                
                results.append((best_category, confidence))
        
        logger.debug(f"Batch categorized {len(images)} images")
        return results
    
    def get_all_scores(self, image: Image.Image) -> dict[Category, float]:
        """Get similarity scores for all categories.
        
        Useful for debugging or showing category breakdown.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary mapping Category to confidence score
        """
        self._load_model()
        
        img_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            image_features = self._model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarities = {}
            for category, text_features in self._prompt_embeddings.items():
                similarity = (image_features @ text_features).item()
                similarities[category] = similarity
        
        # Softmax normalize
        values = list(similarities.values())
        exp_values = np.exp(np.array(values) * 100)
        probs = exp_values / exp_values.sum()
        
        return {cat: float(prob) for cat, prob in zip(similarities.keys(), probs)}
