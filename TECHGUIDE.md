# TECHGUIDE.md - Technical Setup Guide

## Prerequisites

Before you start, make sure you have:

- **Python 3.10+** (3.11 recommended)
- **FFmpeg** installed and in your PATH
- **CUDA 11.8+** (for GPU acceleration - optional but recommended)
- **Git** for version control
- **8GB+ RAM** (16GB recommended for 4K video)

---

## Installation

### Step 1: Clone and Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/social-screenshot-tool.git
cd social-screenshot-tool

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install the project with all dependencies
pip install -e .

# OR install dependencies directly
pip install -r requirements.txt
```

### Step 3: Verify FFmpeg

```bash
# Check FFmpeg is installed
ffmpeg -version

# Should show something like:
# ffmpeg version 6.0 Copyright (c) 2000-2023...
```

If FFmpeg is not found:
- **Windows**: Download from https://ffmpeg.org/download.html, extract, add to PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### Step 4: First Run (Downloads ML Models)

```bash
# Run the app - first time will download ~3GB of models
python -m src.main

# Or just test model downloads
python -c "from src.analysis.categorizer import CLIPCategorizer; CLIPCategorizer()"
```

---

## Dependencies Explained

### pyproject.toml

```toml
[project]
name = "social-screenshot-tool"
version = "1.0.0"
requires-python = ">=3.10"

dependencies = [
    # GUI
    "PySide6>=6.5.0",
    
    # Image Processing
    "Pillow>=10.0.0",
    "opencv-python>=4.8.0",
    
    # ML Models
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "open-clip-torch>=2.20.0",
    "insightface>=0.7.3",
    "onnxruntime-gpu>=1.15.0",  # GPU support
    
    # Audio Analysis
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "silero-vad>=4.0.0",
    
    # Video Processing
    "scenedetect[opencv]>=0.6.1",
    "ffmpeg-python>=0.2.0",
    
    # Utilities
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.0.270",
]
```

### What Each Library Does

| Library | Purpose | Size |
|---------|---------|------|
| PySide6 | Qt-based GUI framework | ~100MB |
| torch | PyTorch for ML models | ~2GB |
| open-clip-torch | CLIP model for categorization | ~500MB (models) |
| insightface | Face detection | ~300MB |
| onnxruntime-gpu | GPU inference for InsightFace | ~200MB |
| librosa | Audio feature extraction | ~50MB |
| silero-vad | Voice activity detection | ~1MB |
| scenedetect | Scene boundary detection | ~10MB |
| ffmpeg-python | Python wrapper for FFmpeg | ~1MB |

---

## GPU Setup (NVIDIA)

### Check CUDA Version

```bash
nvidia-smi
# Look for "CUDA Version: XX.X" in the output
```

### Install Correct PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Is Working

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

import onnxruntime as ort
print(f"ONNX providers: {ort.get_available_providers()}")
# Should include 'CUDAExecutionProvider'
```

---

## Model Downloads

Models are downloaded automatically on first use. Here's where they go:

| Model | Location | Size |
|-------|----------|------|
| CLIP ViT-L-14 | `~/.cache/clip/` | ~1.7GB |
| LAION Aesthetic | `~/.cache/` | ~100MB |
| InsightFace | `~/.insightface/models/` | ~300MB |
| Silero VAD | Downloaded via torch.hub | ~1MB |

### Pre-download Models (Optional)

```python
# Download CLIP
import open_clip
open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')

# Download InsightFace
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# Download Silero VAD
import torch
torch.hub.load('snakers4/silero-vad', 'silero_vad')
```

---

## Configuration

### Environment Variables

```bash
# Optional: Set FFmpeg path explicitly
export FFMPEG_PATH=/usr/local/bin/ffmpeg

# Optional: Set model cache directory
export HF_HOME=/path/to/model/cache
export TORCH_HOME=/path/to/torch/cache
```

### Config File (~/.sst/config.json)

```json
{
    "ffmpeg_path": null,
    "gpu_enabled": true,
    "model_cache_dir": null,
    "default_output_dir": "~/Pictures/SST",
    "jpeg_quality": 90,
    "video_crf": 22,
    "quotas": {
        "people_screenshots": 50,
        "people_clips": 10,
        "details_screenshots": 30,
        "details_clips": 5,
        "venue_screenshots": 20,
        "venue_clips": 5
    },
    "lut_presets": {
        "slog3": "/path/to/slog3.cube",
        "clog3": "/path/to/clog3.cube",
        "vlog": "/path/to/vlog.cube"
    }
}
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'PySide6'"

```bash
pip install PySide6
```

### "CUDA out of memory"

Reduce batch size in config or process one frame at a time:

```python
# In src/analysis/batch.py, reduce BATCH_SIZE
BATCH_SIZE = 1  # Instead of 8 or 16
```

### "InsightFace model not found"

```python
# Force re-download
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', download=True)
app.prepare(ctx_id=0)
```

### ".tod files not showing in file browser"

Add to file dialog filter:
```python
filter = "Video Files (*.mp4 *.mov *.avi *.mkv *.tod *.mod *.mts);;All Files (*)"
```

### "FFmpeg not found"

```python
# Check FFmpeg path
import shutil
print(shutil.which('ffmpeg'))

# If None, FFmpeg is not in PATH
# On Windows, add to PATH or set explicitly:
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
```

### "No CUDA provider" (GPU not working)

```bash
# Check ONNX Runtime providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# If only 'CPUExecutionProvider':
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

### "Audio analysis failing"

```bash
# Install audio dependencies
pip install librosa soundfile

# On Linux, may need:
sudo apt install libsndfile1
```

### "Model download stuck/failing"

```bash
# Clear cache and retry
rm -rf ~/.cache/clip
rm -rf ~/.cache/huggingface

# Use a VPN if behind restrictive firewall
```

---

## Performance Tuning

### For Large Projects (100+ videos)

1. **Use SSD storage** for video files and output
2. **Increase RAM** to 32GB if processing 4K
3. **Use GPU** with 8GB+ VRAM for faster inference
4. **Process overnight** using batch mode

### Memory Management

```python
# In src/analysis/pipeline.py
import gc
import torch

def cleanup():
    """Call between videos to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Parallel Processing

```python
# CPU-bound tasks (image resizing, cropping)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_frame, frames))

# GPU-bound tasks (ML inference) - sequential is often faster
for frame in frames:
    result = model.predict(frame)
```

---

## Development Setup

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/

# With coverage
pytest --cov=src tests/
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

### Type Checking

```bash
# Install mypy
pip install mypy

# Run type checker
mypy src/
```

---

## Building Executable

### Windows (PyInstaller)

```bash
pip install pyinstaller

pyinstaller --name "SocialScreenshotTool" \
    --windowed \
    --icon=assets/icon.ico \
    --add-data "assets;assets" \
    src/main.py
```

### macOS (py2app)

```bash
pip install py2app

python setup.py py2app
```

---

## Logging

Logs are written to `~/.sst/logs/sst.log`:

```
2024-01-15 10:30:45 INFO  [pipeline] Starting analysis of 15 videos
2024-01-15 10:30:46 INFO  [detector] Video: wedding_001.mp4 -> MODERN (4K, 23.976fps)
2024-01-15 10:30:47 DEBUG [face] Found 3 faces at 00:01:30
```

### Log Levels

```python
import logging

# In development
logging.getLogger('sst').setLevel(logging.DEBUG)

# In production
logging.getLogger('sst').setLevel(logging.INFO)
```

---

## Need Help?

1. Check the logs at `~/.sst/logs/`
2. Run with `--verbose` flag for more output
3. Check GitHub Issues
4. Review IMPLEMENTATION.md for code details
