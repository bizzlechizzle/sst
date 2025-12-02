# IMPLEMENTATION.md - Step-by-Step Coding Guide

**For developers who are new to Python ML projects or this codebase.**

This guide walks you through building each component in order. Don't skip ahead - each section builds on the previous ones.

---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Data Models](#2-data-models)
3. [Medium Detection](#3-medium-detection)
4. [Face Detection](#4-face-detection)
5. [Category Classification](#5-category-classification)
6. [Aesthetic Scoring](#6-aesthetic-scoring)
7. [Audio Analysis](#7-audio-analysis)
8. [Smart Cropping](#8-smart-cropping)
9. [Export Functions](#9-export-functions)
10. [GUI Basics](#10-gui-basics)
11. [Putting It Together](#11-putting-it-together)

---

## 1. Project Setup

### What We're Doing
Setting up the project structure and making sure Python can find all our code.

### Create the Files

```
social-screenshot-tool/
├── pyproject.toml
├── src/
│   ├── __init__.py          # Empty file - tells Python this is a package
│   ├── main.py
│   └── core/
│       ├── __init__.py      # Empty file
│       ├── models.py
│       ├── config.py
│       └── exceptions.py
```

### src/core/config.py

This file holds all our constants in one place. When you need to change a value, you change it here - not scattered across 20 files.

```python
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
        'technical': 0.20,
    },
    'DETAILS': {
        'aesthetic': 0.40,
        'sharpness': 0.30,
        'composition': 0.20,
        'technical': 0.10,
    },
    'VENUE': {
        'composition': 0.35,
        'aesthetic': 0.30,
        'technical': 0.20,
        'no_people': 0.15,
    },
}

# === DEFAULT QUOTAS ===
DEFAULT_QUOTAS = {
    'people_screenshots': 50,
    'people_clips': 10,
    'details_screenshots': 30,
    'details_clips': 5,
    'venue_screenshots': 20,
    'venue_clips': 5,
}

# === QUALITY SETTINGS ===
JPEG_QUALITY = 90  # 1-100, higher = better quality, larger file
VIDEO_CRF = 22     # 0-51, lower = better quality, larger file

# === PATHS ===
DEFAULT_CONFIG_DIR = Path.home() / '.sst'
DEFAULT_LOG_DIR = DEFAULT_CONFIG_DIR / 'logs'
```

### src/core/exceptions.py

Custom exceptions make debugging easier. When something goes wrong, you'll know exactly what kind of error it is.

```python
"""Custom exceptions for SST."""

class SSTError(Exception):
    """Base exception for all SST errors."""
    pass


class VideoNotFoundError(SSTError):
    """Raised when a video file doesn't exist."""
    pass


class UnsupportedFormatError(SSTError):
    """Raised when a video format isn't supported."""
    pass


class FFmpegError(SSTError):
    """Raised when FFmpeg fails."""
    pass


class ModelLoadError(SSTError):
    """Raised when an ML model fails to load."""
    pass


class NoFacesFoundError(SSTError):
    """Raised when no faces are found (and faces were expected)."""
    pass


class AudioExtractionError(SSTError):
    """Raised when audio can't be extracted from video."""
    pass
```

---

## 2. Data Models

### What We're Doing
Creating data structures to hold information about videos, frames, and clips. We use **dataclasses** because they're cleaner than dictionaries and give us type hints.

### Why Dataclasses?

Instead of:
```python
# Bad - easy to make typos, no autocomplete
video = {'path': 'video.mp4', 'widht': 1920}  # Oops, typo!
```

We use:
```python
# Good - typos become errors, IDE helps you
video = Video(path='video.mp4', width=1920)
```

### src/core/models.py

```python
"""Data models for SST."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class Medium(Enum):
    """The type of camera used to shoot the video."""
    SUPER_8 = auto()   # Film scans - high res, film frame rates, NO audio
    DAD_CAM = auto()   # Consumer cameras - SD to HD, HAS audio
    MODERN = auto()    # Professional cameras - 4K+, HAS audio


class Category(Enum):
    """What the frame/clip is about."""
    PEOPLE = auto()    # Humans are the main subject
    DETAILS = auto()   # Objects, decorations, close-ups
    VENUE = auto()     # Environment, establishing shots


class LogProfile(Enum):
    """LOG color profile for footage that needs a LUT."""
    NONE = auto()      # Standard/Rec.709 - no LUT needed
    SLOG3 = auto()     # Sony S-Log3
    CLOG3 = auto()     # Canon C-Log3
    VLOG = auto()      # Panasonic V-Log
    NLOG = auto()      # Nikon N-Log
    FLAT = auto()      # Generic flat profile


class AudioEventType(Enum):
    """Types of audio events we can detect."""
    SPEECH = "speech"
    LAUGHTER = "laughter"
    APPLAUSE = "applause"
    CHEER = "cheer"
    MUSIC = "music"
    SILENCE = "silence"


@dataclass
class BoundingBox:
    """A rectangle, used for face locations and crops.
    
    Coordinates are in pixels from top-left corner.
    """
    x: int       # Left edge
    y: int       # Top edge
    width: int
    height: int
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Face:
    """A detected face in a frame."""
    bbox: BoundingBox
    confidence: float          # 0.0 to 1.0
    landmarks: Optional[dict] = None  # Eye positions, etc.


@dataclass
class VideoMetadata:
    """Technical information about a video file."""
    width: int
    height: int
    fps: float
    duration_sec: float
    codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    color_space: Optional[str] = None
    gamma: Optional[str] = None  # For detecting LOG profiles


@dataclass
class Video:
    """A video file and its analysis results."""
    path: Path
    metadata: VideoMetadata
    medium: Medium = Medium.MODERN
    log_profile: LogProfile = LogProfile.NONE
    
    @property
    def has_audio(self) -> bool:
        return self.metadata.has_audio
    
    @property
    def needs_lut(self) -> bool:
        return self.log_profile != LogProfile.NONE
    
    @property
    def filename(self) -> str:
        return self.path.name


@dataclass 
class Frame:
    """A single frame extracted from a video."""
    video: Video
    timestamp_sec: float       # Position in video
    category: Category
    scores: dict = field(default_factory=dict)  # Various quality scores
    faces: list[Face] = field(default_factory=list)
    
    @property
    def frame_number(self) -> int:
        return int(self.timestamp_sec * self.video.metadata.fps)
    
    @property
    def total_score(self) -> float:
        """Combined score from all factors."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class AudioEvent:
    """An interesting moment detected in audio."""
    type: AudioEventType
    start_sec: float
    end_sec: float
    confidence: float
    energy: float = 0.0  # Loudness level
    
    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class ClipSuggestion:
    """A suggested video clip to export."""
    video: Video
    start_sec: float
    end_sec: float
    category: Category
    score: float
    source: str  # "audio" or "visual" or "scene"
    
    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class ExportSettings:
    """Settings for exporting screenshots and clips."""
    output_folder: Path
    crops: list[str] = field(default_factory=lambda: ['square', 'portrait', 'story'])
    jpeg_quality: int = 90
    video_crf: int = 22
    lut_path: Optional[Path] = None
    quotas: dict = field(default_factory=dict)
    clip_duration_short: float = 8.0
    clip_duration_long: float = 30.0
```

### Understanding the Models

Think of these like forms you fill out:

**Video**: "Here's a video file. What's its path? How big is it? Does it have audio?"

**Frame**: "Here's one frame. Which video is it from? What time? What category? How good is it?"

**Face**: "Here's a face. Where is it in the image? How confident are we it's a face?"

---

## 3. Medium Detection

### What We're Doing
Automatically figuring out if footage is from Super 8 film, a consumer camera, or a professional camera. This affects how we process it (especially audio).

### src/analysis/detector.py

```python
"""Auto-detect video medium and LOG profile."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from ..core.models import Medium, LogProfile, VideoMetadata
from ..core.exceptions import FFmpegError


class MediumDetector:
    """Detects the type of camera used to shoot a video."""
    
    # Frame rates typical of film (Super 8)
    FILM_FRAME_RATES = {16, 18, 24}
    
    # Frame rates typical of consumer video (NTSC/PAL)
    CONSUMER_FRAME_RATES = {25, 29.97, 30}
    
    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract technical metadata from a video file using FFprobe.
        
        FFprobe is part of FFmpeg and reads video file information.
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',                    # Don't print extra info
            '-print_format', 'json',          # Output as JSON
            '-show_streams',                  # Show stream info
            '-show_format',                   # Show container info
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"FFprobe failed: {e.stderr}")
        except json.JSONDecodeError:
            raise FFmpegError("FFprobe returned invalid JSON")
        
        # Find the video stream
        video_stream = None
        audio_stream = None
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and video_stream is None:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and audio_stream is None:
                audio_stream = stream
        
        if not video_stream:
            raise FFmpegError("No video stream found")
        
        # Parse frame rate (might be "24000/1001" or "24")
        fps_str = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        # Get duration
        duration = float(data.get('format', {}).get('duration', 0))
        
        return VideoMetadata(
            width=int(video_stream.get('width', 0)),
            height=int(video_stream.get('height', 0)),
            fps=round(fps, 3),
            duration_sec=duration,
            codec=video_stream.get('codec_name', 'unknown'),
            has_audio=audio_stream is not None,
            audio_codec=audio_stream.get('codec_name') if audio_stream else None,
            color_space=video_stream.get('color_space'),
            gamma=video_stream.get('color_transfer'),
        )
    
    def detect_medium(self, metadata: VideoMetadata) -> Medium:
        """Determine the medium based on technical characteristics.
        
        Logic:
        1. Super 8 = High resolution + film frame rates (16, 18, 24 fps)
        2. Modern = 4K+ at standard video rates
        3. Dad Cam = SD resolution OR consumer HD at consumer frame rates
        """
        width = metadata.width
        height = metadata.height
        fps = metadata.fps
        
        # Round fps to nearest common value for comparison
        fps_rounded = self._round_fps(fps)
        
        # Super 8: High-res film scans at film frame rates
        # Film scans are typically 2K, 4K, or 6K at exactly 16, 18, or 24 fps
        if width >= 2000 and fps_rounded in self.FILM_FRAME_RATES:
            return Medium.SUPER_8
        
        # Modern: 4K or higher at standard video frame rates
        if width >= 3840 and fps >= 23.976:
            return Medium.MODERN
        
        # Dad Cam: Standard definition (SD)
        if width <= 720 or height <= 576:
            return Medium.DAD_CAM
        
        # Dad Cam: HD at consumer frame rates with consumer codecs
        if width <= 1920 and fps_rounded in self.CONSUMER_FRAME_RATES:
            return Medium.DAD_CAM
        
        # Default: Assume modern camera
        return Medium.MODERN
    
    def detect_log_profile(self, metadata: VideoMetadata) -> LogProfile:
        """Detect LOG color profile from metadata.
        
        Cameras store their color profile in metadata tags.
        We check gamma (transfer function) and color space.
        """
        gamma = (metadata.gamma or '').lower()
        color_space = (metadata.color_space or '').lower()
        
        # Sony S-Log3
        if 's-log3' in gamma or 'slog3' in gamma:
            return LogProfile.SLOG3
        if 's-gamut' in color_space:
            return LogProfile.SLOG3
        
        # Canon C-Log3
        if 'c-log3' in gamma or 'clog3' in gamma:
            return LogProfile.CLOG3
        if 'cinema gamut' in color_space:
            return LogProfile.CLOG3
        
        # Panasonic V-Log
        if 'v-log' in gamma or 'vlog' in gamma:
            return LogProfile.VLOG
        if 'v-gamut' in color_space:
            return LogProfile.VLOG
        
        # Nikon N-Log
        if 'n-log' in gamma or 'nlog' in gamma:
            return LogProfile.NLOG
        
        # Generic flat/log profile
        if any(term in gamma for term in ['flat', 'log', 'neutral']):
            return LogProfile.FLAT
        
        return LogProfile.NONE
    
    def _round_fps(self, fps: float) -> float:
        """Round FPS to nearest common value.
        
        Video frame rates are often slightly off (23.976 vs 24).
        This rounds to the nearest "standard" frame rate.
        """
        common_fps = [16, 18, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]
        return min(common_fps, key=lambda x: abs(x - fps))
```

### How to Use It

```python
from pathlib import Path
from src.analysis.detector import MediumDetector

detector = MediumDetector()

# Get video info
metadata = detector.get_video_metadata(Path("wedding.mp4"))
print(f"Resolution: {metadata.width}x{metadata.height}")
print(f"FPS: {metadata.fps}")
print(f"Has audio: {metadata.has_audio}")

# Detect medium
medium = detector.detect_medium(metadata)
print(f"Medium: {medium}")  # Medium.MODERN or Medium.DAD_CAM or Medium.SUPER_8

# Detect LOG profile
log_profile = detector.detect_log_profile(metadata)
print(f"LOG profile: {log_profile}")  # LogProfile.SLOG3, etc.
```

---

## 4. Face Detection

### What We're Doing
Finding faces in frames. This is crucial for:
- Scoring PEOPLE shots (more/bigger faces = better)
- Smart cropping (keep faces in frame)
- Eventually, couple identification (V2)

### Why InsightFace?
It's the most accurate face detector that runs on GPU. It can find faces at different angles, sizes, and lighting conditions.

### src/analysis/face.py

```python
"""Face detection using InsightFace."""
from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Optional

from ..core.models import Face, BoundingBox
from ..core.exceptions import ModelLoadError


class FaceDetector:
    """Detects faces in images using InsightFace RetinaFace model."""
    
    def __init__(self, gpu_id: int = 0):
        """Initialize the face detector.
        
        Args:
            gpu_id: Which GPU to use (0 = first GPU, -1 = CPU)
        """
        self.gpu_id = gpu_id
        self._model = None
    
    def _load_model(self):
        """Lazy-load the model on first use.
        
        This saves memory if face detection isn't needed.
        """
        if self._model is not None:
            return
        
        try:
            from insightface.app import FaceAnalysis
            
            # 'buffalo_l' is the large model - most accurate
            self._model = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self._model.prepare(ctx_id=self.gpu_id)
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load InsightFace: {e}")
    
    def detect(self, image: Image.Image, min_confidence: float = 0.5) -> list[Face]:
        """Detect all faces in an image.
        
        Args:
            image: PIL Image to analyze
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            List of Face objects
        """
        self._load_model()
        
        # Convert PIL Image to numpy array (InsightFace needs BGR)
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # InsightFace expects BGR, PIL gives RGB
        img_bgr = img_array[:, :, ::-1]
        
        # Detect faces
        results = self._model.get(img_bgr)
        
        faces = []
        for face_data in results:
            # face_data.bbox is [x1, y1, x2, y2]
            bbox = face_data.bbox.astype(int)
            confidence = float(face_data.det_score)
            
            if confidence < min_confidence:
                continue
            
            face = Face(
                bbox=BoundingBox(
                    x=int(bbox[0]),
                    y=int(bbox[1]),
                    width=int(bbox[2] - bbox[0]),
                    height=int(bbox[3] - bbox[1]),
                ),
                confidence=confidence,
                landmarks=self._extract_landmarks(face_data),
            )
            faces.append(face)
        
        # Sort by face size (largest first)
        faces.sort(key=lambda f: f.bbox.area, reverse=True)
        return faces
    
    def _extract_landmarks(self, face_data) -> Optional[dict]:
        """Extract facial landmarks if available."""
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
    
    def count_faces(self, image: Image.Image) -> int:
        """Quick count of faces in an image."""
        return len(self.detect(image))
    
    def get_largest_face(self, image: Image.Image) -> Optional[Face]:
        """Get the largest (most prominent) face."""
        faces = self.detect(image)
        return faces[0] if faces else None
```

### How to Use It

```python
from PIL import Image
from src.analysis.face import FaceDetector

detector = FaceDetector(gpu_id=0)  # Use first GPU

# Load an image
image = Image.open("wedding_photo.jpg")

# Detect faces
faces = detector.detect(image)
print(f"Found {len(faces)} faces")

for i, face in enumerate(faces):
    print(f"Face {i+1}:")
    print(f"  Position: ({face.bbox.x}, {face.bbox.y})")
    print(f"  Size: {face.bbox.width}x{face.bbox.height}")
    print(f"  Confidence: {face.confidence:.2%}")
```

---

## 5. Category Classification

### What We're Doing
Using CLIP to classify each frame as PEOPLE, DETAILS, or VENUE.

### What is CLIP?
CLIP (Contrastive Language-Image Pre-training) is a model trained to understand images AND text together. We give it text descriptions like "a portrait photo of a wedding couple" and it tells us how well an image matches that description.

### Zero-Shot Classification
We don't train the model - we just give it prompts for each category and it figures out which one matches best.

### src/analysis/categorizer.py

```python
"""Category classification using CLIP."""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from typing import Tuple

from ..core.models import Category
from ..core.exceptions import ModelLoadError


class CLIPCategorizer:
    """Classifies images into PEOPLE, DETAILS, or VENUE using CLIP."""
    
    # Wedding-specific prompts for each category
    # More prompts = better accuracy
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
        ],
        Category.DETAILS: [
            "wedding rings on a surface",
            "a bouquet of wedding flowers",
            "wedding dress details and lace",
            "wedding table decorations",
            "a wedding cake",
            "wedding invitation and stationery",
            "wedding jewelry and accessories",
            "wedding food and drinks",
            "decorative wedding details",
        ],
        Category.VENUE: [
            "exterior of a wedding venue",
            "empty wedding ceremony location",
            "wedding reception hall interior",
            "landscape at a wedding venue",
            "architectural photo of a wedding venue",
            "outdoor wedding location scenery",
            "church or chapel interior",
            "establishing shot of a wedding location",
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
    
    def _load_model(self):
        """Lazy-load the model."""
        if self._model is not None:
            return
        
        try:
            import open_clip
            
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self._device,
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Pre-compute prompt embeddings (this speeds up inference)
            self._precompute_prompts()
            
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
        
        return best_category, confidence
    
    def categorize_batch(self, images: list[Image.Image]) -> list[Tuple[Category, float]]:
        """Classify multiple images efficiently.
        
        Batching is faster than processing one at a time.
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
        
        return results
```

### How to Use It

```python
from PIL import Image
from src.analysis.categorizer import CLIPCategorizer

categorizer = CLIPCategorizer()

# Classify a single image
image = Image.open("wedding_photo.jpg")
category, confidence = categorizer.categorize(image)
print(f"Category: {category.name}, Confidence: {confidence:.2%}")

# Batch classify multiple images
images = [Image.open(f"photo_{i}.jpg") for i in range(10)]
results = categorizer.categorize_batch(images)
for i, (cat, conf) in enumerate(results):
    print(f"Photo {i}: {cat.name} ({conf:.2%})")
```

---

## 6. Aesthetic Scoring

### What We're Doing
Scoring how "beautiful" or "Instagram-worthy" an image is. We use the LAION aesthetic predictor which was trained on millions of images with human ratings.

### src/analysis/aesthetic.py

```python
"""Aesthetic quality scoring."""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image

from ..core.exceptions import ModelLoadError


class AestheticScorer:
    """Scores images for visual aesthetics using LAION predictor."""
    
    def __init__(self):
        self._model = None
        self._clip_model = None
        self._preprocess = None
        self._device = None
    
    def _load_model(self):
        """Load CLIP and aesthetic predictor."""
        if self._model is not None:
            return
        
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
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load aesthetic scorer: {e}")
    
    def _create_aesthetic_model(self) -> nn.Module:
        """Create the aesthetic prediction MLP."""
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
        # Download weights if needed
        import urllib.request
        import os
        
        weights_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        cache_dir = os.path.expanduser("~/.cache/aesthetic")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "aesthetic_predictor.pth")
        
        if not os.path.exists(weights_path):
            urllib.request.urlretrieve(weights_url, weights_path)
        
        self._model.load_state_dict(torch.load(weights_path, map_location=self._device))
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
        return max(1.0, min(10.0, score))
    
    def score_batch(self, images: list[Image.Image]) -> list[float]:
        """Score multiple images efficiently."""
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
        
        if isinstance(scores, float):
            scores = [scores]
        
        return [max(1.0, min(10.0, s)) for s in scores]
```

---

## 7. Audio Analysis

### What We're Doing
Finding interesting audio moments (speech, laughter, applause) to suggest good clip boundaries.

### Why This Matters
The best wedding clips end on laughter, applause, or the end of a sentence - not mid-word.

### src/analysis/audio.py

```python
"""Audio analysis for moment detection."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional

from ..core.models import AudioEvent, AudioEventType
from ..core.exceptions import AudioExtractionError


class AudioAnalyzer:
    """Analyzes audio to find interesting moments."""
    
    def __init__(self):
        self._vad_model = None
        self._sample_rate = 16000  # Silero VAD needs 16kHz
    
    def _load_vad(self):
        """Load Silero VAD (Voice Activity Detection) model."""
        if self._vad_model is not None:
            return
        
        self._vad_model, _ = torch.hub.load(
            'snakers4/silero-vad',
            'silero_vad',
            force_reload=False,
        )
    
    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> np.ndarray:
        """Extract audio from video as numpy array.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save WAV file
        
        Returns:
            Audio samples as numpy array (mono, 16kHz)
        """
        import subprocess
        import tempfile
        import soundfile as sf
        
        # Create temp file if no output specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        
        # Extract audio with FFmpeg
        cmd = [
            'ffmpeg', '-y',           # Overwrite output
            '-i', str(video_path),    # Input video
            '-vn',                     # No video
            '-acodec', 'pcm_s16le',   # WAV format
            '-ar', str(self._sample_rate),  # 16kHz
            '-ac', '1',               # Mono
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise AudioExtractionError(f"FFmpeg audio extraction failed: {e.stderr}")
        
        # Load the audio
        audio, sr = sf.read(output_path)
        
        return audio.astype(np.float32)
    
    def detect_speech(self, audio: np.ndarray) -> list[AudioEvent]:
        """Detect speech segments using Silero VAD.
        
        Args:
            audio: Audio samples (mono, 16kHz)
        
        Returns:
            List of AudioEvent objects for speech segments
        """
        self._load_vad()
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio)
        
        # Get speech timestamps
        speech_timestamps = self._vad_model.get_speech_timestamps(
            audio_tensor,
            self._vad_model,
            sampling_rate=self._sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )
        
        events = []
        for segment in speech_timestamps:
            start_sec = segment['start'] / self._sample_rate
            end_sec = segment['end'] / self._sample_rate
            
            events.append(AudioEvent(
                type=AudioEventType.SPEECH,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=0.9,  # VAD is quite reliable
            ))
        
        return events
    
    def detect_energy_peaks(self, audio: np.ndarray, threshold: float = 0.7) -> list[AudioEvent]:
        """Detect high-energy moments (laughter, applause, cheers).
        
        Uses RMS (Root Mean Square) energy to find loud moments.
        """
        import librosa
        
        # Compute RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Convert to dB for easier thresholding
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find peaks above threshold
        threshold_db = np.max(rms_db) * threshold
        peak_frames = np.where(rms_db > threshold_db)[0]
        
        if len(peak_frames) == 0:
            return []
        
        # Group consecutive frames into segments
        events = []
        segment_start = peak_frames[0]
        prev_frame = peak_frames[0]
        
        for frame in peak_frames[1:]:
            # If there's a gap, end the segment
            if frame - prev_frame > 10:  # ~0.3 seconds gap
                start_sec = segment_start * hop_length / self._sample_rate
                end_sec = prev_frame * hop_length / self._sample_rate
                
                # Classify the event based on spectral features
                event_type = self._classify_energy_event(
                    audio, start_sec, end_sec
                )
                
                events.append(AudioEvent(
                    type=event_type,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    confidence=0.7,
                    energy=float(np.mean(rms_db[segment_start:prev_frame+1])),
                ))
                
                segment_start = frame
            
            prev_frame = frame
        
        # Don't forget the last segment
        start_sec = segment_start * hop_length / self._sample_rate
        end_sec = prev_frame * hop_length / self._sample_rate
        event_type = self._classify_energy_event(audio, start_sec, end_sec)
        
        events.append(AudioEvent(
            type=event_type,
            start_sec=start_sec,
            end_sec=end_sec,
            confidence=0.7,
            energy=float(np.mean(rms_db[segment_start:prev_frame+1])),
        ))
        
        return events
    
    def _classify_energy_event(self, audio: np.ndarray, start_sec: float, end_sec: float) -> AudioEventType:
        """Classify what type of energy event this is.
        
        Uses spectral features to distinguish:
        - Applause: Broadband noise (flat spectrum)
        - Laughter: Mid-frequency variations
        - Cheering: High energy, harmonic content
        """
        import librosa
        
        # Extract the segment
        start_sample = int(start_sec * self._sample_rate)
        end_sample = int(end_sec * self._sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < 1024:
            return AudioEventType.CHEER  # Default for short segments
        
        # Compute spectral features
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self._sample_rate))
        
        # Applause: Very flat spectrum (noise-like)
        if spectral_flatness > 0.3:
            return AudioEventType.APPLAUSE
        
        # Laughter: Mid-range centroid with variation
        if 1000 < spectral_centroid < 3000:
            return AudioEventType.LAUGHTER
        
        # Default to cheer
        return AudioEventType.CHEER
    
    def detect_music(self, audio: np.ndarray) -> list[AudioEvent]:
        """Detect music segments.
        
        Uses tempo and harmonic content to identify music.
        """
        import librosa
        
        # Compute chromagram (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self._sample_rate)
        
        # Music has consistent harmonic patterns
        chroma_std = np.std(chroma, axis=0)
        
        # Low variation = likely music
        music_frames = chroma_std < np.percentile(chroma_std, 30)
        
        # Convert to time segments
        hop_length = 512
        events = []
        
        # Group consecutive frames
        in_music = False
        start_frame = 0
        
        for i, is_music in enumerate(music_frames):
            if is_music and not in_music:
                start_frame = i
                in_music = True
            elif not is_music and in_music:
                # End of music segment
                if i - start_frame > 20:  # At least 0.5 seconds
                    events.append(AudioEvent(
                        type=AudioEventType.MUSIC,
                        start_sec=start_frame * hop_length / self._sample_rate,
                        end_sec=i * hop_length / self._sample_rate,
                        confidence=0.6,
                    ))
                in_music = False
        
        return events
    
    def analyze(self, video_path: Path) -> list[AudioEvent]:
        """Full audio analysis of a video.
        
        Returns all detected audio events sorted by time.
        """
        # Extract audio
        audio = self.extract_audio(video_path)
        
        # Detect all event types
        events = []
        events.extend(self.detect_speech(audio))
        events.extend(self.detect_energy_peaks(audio))
        events.extend(self.detect_music(audio))
        
        # Sort by start time
        events.sort(key=lambda e: e.start_sec)
        
        return events
```

---

## 8. Smart Cropping

### What We're Doing
Cropping images for different aspect ratios while keeping faces visible.

### The Challenge
A 16:9 video frame might have faces on the sides. When you crop to 1:1 (square), you might cut off faces. Smart cropping positions the crop to include as many faces as possible.

### src/crops/smart_crop.py

```python
"""Face-aware smart cropping."""
from __future__ import annotations

from PIL import Image
from typing import Optional

from ..core.models import Face, BoundingBox
from .presets import CROP_PRESETS


def smart_crop(
    image: Image.Image,
    target_width: int,
    target_height: int,
    faces: list[Face],
    bias: str = 'center',
) -> Image.Image:
    """Crop an image while keeping faces in frame.
    
    Args:
        image: Source image
        target_width: Desired output width
        target_height: Desired output height
        faces: List of detected faces
        bias: Where to position crop if no faces ('center', 'left', 'right')
    
    Returns:
        Cropped and resized image
    """
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
    """Find crop position that includes the most/largest faces."""
    
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
    
    return crop_x, crop_y


def _position_with_bias(
    img_width: int,
    img_height: int,
    crop_width: int,
    crop_height: int,
    bias: str,
) -> tuple[int, int]:
    """Position crop based on bias preference."""
    
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
    faces: list[Face],
) -> Image.Image:
    """Crop image to a named preset.
    
    Args:
        image: Source image
        preset: Preset name ('square', 'portrait', 'landscape', 'story', etc.)
        faces: Detected faces
    
    Returns:
        Cropped image
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
    faces: list[Face],
    presets: list[str] = None,
) -> dict[str, Image.Image]:
    """Crop image to multiple presets at once.
    
    Args:
        image: Source image
        faces: Detected faces
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
```

### src/crops/presets.py

```python
"""Crop preset definitions."""

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
    """Get detailed info about a preset."""
    if preset not in CROP_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    
    width, height = CROP_PRESETS[preset]
    
    return {
        'name': preset,
        'width': width,
        'height': height,
        'aspect_ratio': f"{width}:{height}",
        'orientation': 'square' if width == height else ('portrait' if height > width else 'landscape'),
    }
```

---

## 9. Export Functions

### What We're Doing
Saving screenshots (as JPEG) and video clips (as MP4).

### src/export/screenshot.py

```python
"""Screenshot export functions."""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import subprocess

from ..core.config import JPEG_QUALITY


def export_screenshot(
    image: Image.Image,
    output_path: Path,
    quality: int = JPEG_QUALITY,
) -> Path:
    """Save an image as JPEG.
    
    Args:
        image: PIL Image to save
        output_path: Where to save (should end in .jpg)
        quality: JPEG quality (1-100)
    
    Returns:
        Path to saved file
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to RGB if necessary (JPEG doesn't support alpha)
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    
    # Save with quality setting
    image.save(
        output_path,
        'JPEG',
        quality=quality,
        optimize=True,
    )
    
    return output_path


def extract_frame(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path = None,
) -> Image.Image:
    """Extract a single frame from a video.
    
    Args:
        video_path: Path to video file
        timestamp_sec: Time position in seconds
        output_path: Optional path to save the frame
    
    Returns:
        PIL Image of the frame
    """
    import tempfile
    
    # Create temp file for frame
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
    
    # Use FFmpeg to extract frame
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(timestamp_sec),  # Seek to position
        '-i', str(video_path),
        '-frames:v', '1',           # Extract 1 frame
        '-q:v', '2',                # High quality
        str(output_path)
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    
    # Load and return
    return Image.open(output_path)
```

### src/export/clip.py

```python
"""Video clip export functions."""
from __future__ import annotations

from pathlib import Path
import subprocess

from ..core.config import VIDEO_CRF
from ..core.models import ClipSuggestion


def export_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    crop: tuple[int, int, int, int] = None,  # x, y, w, h
    crf: int = VIDEO_CRF,
    lut_path: Path = None,
) -> Path:
    """Export a video clip.
    
    Args:
        video_path: Source video path
        output_path: Where to save clip
        start_sec: Start time in seconds
        end_sec: End time in seconds
        crop: Optional crop box (x, y, width, height)
        crf: Quality setting (lower = better, larger file)
        lut_path: Optional LUT file to apply
    
    Returns:
        Path to exported clip
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    duration = end_sec - start_sec
    
    # Build filter chain
    filters = []
    
    # Crop if specified
    if crop:
        x, y, w, h = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    
    # Apply LUT if specified
    if lut_path:
        filters.append(f"lut3d={lut_path}")
    
    # Scale to 1080p max
    filters.append("scale='min(1920,iw)':-2")
    
    # Sharpen slightly
    filters.append("unsharp=5:5:0.5:5:5:0.5")
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', str(video_path),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', str(crf),
        '-c:a', 'aac',
        '-b:a', '128k',
    ]
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    cmd.append(str(output_path))
    
    subprocess.run(cmd, capture_output=True, check=True)
    
    return output_path


def export_clip_suggestion(
    suggestion: ClipSuggestion,
    output_folder: Path,
    preset: str = 'landscape',
    **kwargs,
) -> Path:
    """Export a ClipSuggestion to a file.
    
    Args:
        suggestion: The clip to export
        output_folder: Where to save
        preset: Crop preset to use
        **kwargs: Additional arguments for export_clip
    
    Returns:
        Path to exported clip
    """
    from ..crops.presets import CROP_PRESETS
    
    # Generate filename
    filename = f"{suggestion.video.path.stem}_{suggestion.start_sec:.1f}_{suggestion.category.name.lower()}_{preset}.mp4"
    output_path = output_folder / filename
    
    # Get crop dimensions if preset specified
    crop = None
    if preset in CROP_PRESETS:
        # TODO: Implement proper video crop calculation
        pass
    
    return export_clip(
        suggestion.video.path,
        output_path,
        suggestion.start_sec,
        suggestion.end_sec,
        crop=crop,
        **kwargs,
    )
```

---

## 10. GUI Basics

### What We're Doing
Creating the user interface with PySide6 (Qt).

### src/gui/main_window.py

This is a simplified starting point. The full GUI will have more features.

```python
"""Main application window."""
from __future__ import annotations

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QListWidget,
    QListWidgetItem, QSplitter, QGroupBox, QCheckBox, QSpinBox,
)
from PySide6.QtCore import Qt, QThread, Signal


class AnalysisWorker(QThread):
    """Background thread for video analysis."""
    
    progress = Signal(int, str)  # percent, message
    finished = Signal(list)      # results
    error = Signal(str)          # error message
    
    def __init__(self, video_paths: list[Path]):
        super().__init__()
        self.video_paths = video_paths
    
    def run(self):
        """Run analysis in background."""
        try:
            # TODO: Implement actual analysis
            results = []
            total = len(self.video_paths)
            
            for i, path in enumerate(self.video_paths):
                self.progress.emit(
                    int((i + 1) / total * 100),
                    f"Analyzing {path.name}..."
                )
                # Placeholder for actual analysis
                results.append({'path': path, 'frames': []})
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Social Screenshot Tool")
        self.setMinimumSize(1200, 800)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        
        # Left panel: Video list
        left_panel = self._create_video_panel()
        
        # Right panel: Preview and settings
        right_panel = self._create_preview_panel()
        
        # Add to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        
        layout.addWidget(splitter)
    
    def _create_video_panel(self) -> QWidget:
        """Create the video list panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("Videos")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # Add button
        add_btn = QPushButton("Add Videos...")
        add_btn.clicked.connect(self._add_videos)
        layout.addWidget(add_btn)
        
        # Video list
        self.video_list = QListWidget()
        layout.addWidget(self.video_list)
        
        # Analyze button
        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self._start_analysis)
        layout.addWidget(analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)
        
        return panel
    
    def _create_preview_panel(self) -> QWidget:
        """Create the preview and settings panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Preview area (placeholder)
        preview = QLabel("Select a video to preview")
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet("background: #333; color: #888; min-height: 400px;")
        layout.addWidget(preview)
        
        # Settings group
        settings_group = QGroupBox("Export Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Crop checkboxes
        crops_label = QLabel("Crops to export:")
        settings_layout.addWidget(crops_label)
        
        self.crop_checkboxes = {}
        for crop in ['square', 'portrait', 'landscape', 'story']:
            cb = QCheckBox(crop.capitalize())
            cb.setChecked(True)
            self.crop_checkboxes[crop] = cb
            settings_layout.addWidget(cb)
        
        layout.addWidget(settings_group)
        
        # Export button
        export_btn = QPushButton("Export All")
        export_btn.clicked.connect(self._export_all)
        layout.addWidget(export_btn)
        
        return panel
    
    def _add_videos(self):
        """Open file dialog to add videos."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.tod *.mod *.mts);;All Files (*)",
        )
        
        for file in files:
            item = QListWidgetItem(Path(file).name)
            item.setData(Qt.UserRole, file)
            self.video_list.addItem(item)
    
    def _start_analysis(self):
        """Start analyzing videos."""
        # Get all video paths
        paths = []
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            paths.append(Path(item.data(Qt.UserRole)))
        
        if not paths:
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start worker
        self.worker = AnalysisWorker(paths)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()
    
    def _on_progress(self, percent: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)
    
    def _on_analysis_finished(self, results: list):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setText(f"Found {len(results)} results")
        # TODO: Display results in preview panel
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.progress_label.setText(f"Error: {error}")
    
    def _export_all(self):
        """Export all selected frames and clips."""
        # Get selected crops
        crops = [name for name, cb in self.crop_checkboxes.items() if cb.isChecked()]
        
        # TODO: Implement export
        print(f"Would export with crops: {crops}")


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

---

## 11. Putting It Together

### src/main.py

```python
"""Social Screenshot Tool - Main Entry Point."""
from __future__ import annotations

import sys
import logging
from pathlib import Path

from .core.config import DEFAULT_LOG_DIR


def setup_logging():
    """Configure logging for the application."""
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-5s [%(name)s] %(message)s',
        handlers=[
            logging.FileHandler(DEFAULT_LOG_DIR / 'sst.log'),
            logging.StreamHandler(),
        ]
    )


def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger('sst')
    logger.info("Starting Social Screenshot Tool")
    
    # Import GUI here to avoid slow startup
    from .gui.main_window import main as gui_main
    gui_main()


if __name__ == '__main__':
    main()
```

---

## What's Next?

After implementing these basics:

1. **Connect the pipeline**: Wire up the analysis components in `src/analysis/pipeline.py`
2. **Add the candidate grid**: Display frame thumbnails in the GUI
3. **Implement progress/resume**: Save state between runs
4. **Add LUT selection**: Let users choose and apply LUTs
5. **Polish the UI**: Better video preview, thumbnails, etc.

Each of these is a substantial piece of work. Take them one at a time, test as you go, and refer back to CLAUDE.md for design decisions.

Good luck! 🎬
