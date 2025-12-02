"""Face detection using InsightFace."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from ..core.models import Face, BoundingBox
from ..core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces in images using InsightFace RetinaFace model."""
    
    def __init__(self, gpu_id: int = 0):
        """Initialize the face detector.
        
        Args:
            gpu_id: Which GPU to use (0 = first GPU, -1 = CPU)
        """
        self.gpu_id = gpu_id
        self._model = None
        self._initialized = False
    
    def _load_model(self):
        """Lazy-load the model on first use.
        
        This saves memory if face detection isn't needed.
        """
        if self._initialized:
            return
        
        logger.info("Loading InsightFace model...")
        
        try:
            from insightface.app import FaceAnalysis
            
            # 'buffalo_l' is the large model - most accurate
            self._model = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self._model.prepare(ctx_id=self.gpu_id)
            self._initialized = True
            
            logger.info("InsightFace model loaded successfully")
            
        except ImportError:
            raise ModelLoadError(
                "InsightFace not installed. Run: pip install insightface onnxruntime-gpu"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load InsightFace: {e}")
    
    def detect(self, image: Image.Image, min_confidence: float = 0.5) -> list[Face]:
        """Detect all faces in an image.
        
        Args:
            image: PIL Image to analyze
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            List of Face objects, sorted by size (largest first)
        """
        self._load_model()
        
        # Convert PIL Image to numpy array (InsightFace needs BGR)
        img_array = np.array(image)
        
        # Handle different image modes
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # InsightFace expects BGR, PIL gives RGB
        img_bgr = img_array[:, :, ::-1].copy()
        
        # Detect faces
        results = self._model.get(img_bgr)
        
        faces = []
        for face_data in results:
            # face_data.bbox is [x1, y1, x2, y2]
            bbox = face_data.bbox.astype(int)
            confidence = float(face_data.det_score)
            
            if confidence < min_confidence:
                continue
            
            # Ensure bbox is within image bounds
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(image.width, int(bbox[2]))
            y2 = min(image.height, int(bbox[3]))
            
            face = Face(
                bbox=BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                ),
                confidence=confidence,
                landmarks=self._extract_landmarks(face_data),
            )
            faces.append(face)
        
        # Sort by face size (largest first)
        faces.sort(key=lambda f: f.bbox.area, reverse=True)
        
        logger.debug(f"Detected {len(faces)} faces (min_conf={min_confidence})")
        return faces
    
    def _extract_landmarks(self, face_data) -> Optional[dict]:
        """Extract facial landmarks if available.
        
        Args:
            face_data: InsightFace face detection result
            
        Returns:
            Dictionary of landmark positions or None
        """
        if not hasattr(face_data, 'kps') or face_data.kps is None:
            return None
        
        kps = face_data.kps
        return {
            'left_eye': (float(kps[0][0]), float(kps[0][1])),
            'right_eye': (float(kps[1][0]), float(kps[1][1])),
            'nose': (float(kps[2][0]), float(kps[2][1])),
            'left_mouth': (float(kps[3][0]), float(kps[3][1])),
            'right_mouth': (float(kps[4][0]), float(kps[4][1])),
        }
    
    def count_faces(self, image: Image.Image, min_confidence: float = 0.5) -> int:
        """Quick count of faces in an image.
        
        Args:
            image: PIL Image to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            Number of faces detected
        """
        return len(self.detect(image, min_confidence))
    
    def get_largest_face(self, image: Image.Image, min_confidence: float = 0.5) -> Optional[Face]:
        """Get the largest (most prominent) face.
        
        Args:
            image: PIL Image to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            Largest Face object or None if no faces found
        """
        faces = self.detect(image, min_confidence)
        return faces[0] if faces else None
    
    def get_face_centers(self, image: Image.Image, min_confidence: float = 0.5) -> list[tuple[int, int]]:
        """Get center points of all detected faces.
        
        Useful for smart cropping.
        
        Args:
            image: PIL Image to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (x, y) center coordinates
        """
        faces = self.detect(image, min_confidence)
        return [(f.bbox.center_x, f.bbox.center_y) for f in faces]
