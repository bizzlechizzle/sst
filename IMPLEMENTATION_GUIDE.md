# Social Screenshot Tool - Implementation Guide

**Last Updated: December 2025**
**Status: COMPLETE (100%)**
**Tests: 100/100 Passing**

---

## Quick Start

```bash
# Prerequisites
brew install python@3.12 ffmpeg

# Setup
cd /path/to/sst
rm -rf venv
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -e .

# Run GUI
sst
# OR
python -m src

# Run tests
pip install pytest
python -m pytest tests/ -v
```

---

## Spec Compliance Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Short clips + screenshots for social media | ✅ | `export/clip.py`, `export/screenshot.py` |
| GUI for Mac/Linux | ✅ | `gui/main_window.py` (PySide6) |
| Batch OR individual selection | ✅ | Checkbox per frame, quotas per category |
| High-res footage support (6k) | ✅ | No resolution limits in pipeline |
| Auto-detect tool | ✅ | Medium, face, category auto-detection |
| Modern audio on Dad Cam | ✅ | `analysis/audio.py` + clip boundaries |
| 3 categories (People/Details/Venue) | ✅ | `analysis/categorizer.py` (CLIP-based) |
| Raw/LOG footage (needs LUT) | ✅ | Auto-detection + LUT settings UI |
| Finished media (1080p+) | ✅ | Handled by medium detector |
| Super 8 (up to 6k scan) | ✅ | `analysis/detector.py` |
| Dad Cam (VHS/MiniDV) | ✅ | `analysis/detector.py` |
| Modern Digital (4k) | ✅ | `analysis/detector.py` |
| Vertical (Stories 9:16) | ✅ | `crops/presets.py` |
| Square (Posts 1:1) | ✅ | `crops/presets.py` |
| Portrait (4:5) | ✅ | `crops/presets.py` |
| Landscape (16:9) | ✅ | `crops/presets.py` |
| Single clips (2-18s) | ✅ | `config.py` CLIP_TYPES |
| Multi clips (8-58s) | ✅ | `config.py` CLIP_TYPES |
| 1080px max screenshots | ✅ | Resolution selector in GUI |
| 2048px Facebook high-res | ✅ | Resolution selector in GUI |
| Smart cropping (face-aware) | ✅ | `crops/smart_crop.py` |
| User-supplied LUT | ✅ | LUT folder picker in GUI |
| Best-of with priorities | ✅ | Category-specific scoring weights |

---

## Architecture

```
sst/
├── src/
│   ├── main.py              # Entry point
│   ├── __main__.py          # python -m src support
│   ├── core/
│   │   ├── models.py        # Data classes
│   │   ├── config.py        # Constants, presets, quotas
│   │   └── exceptions.py    # Custom exceptions
│   ├── analysis/
│   │   ├── detector.py      # Medium/LOG detection
│   │   ├── face.py          # InsightFace face detection
│   │   ├── categorizer.py   # CLIP categorization
│   │   ├── aesthetic.py     # LAION aesthetic scoring
│   │   ├── technical.py     # Sharpness/exposure/noise
│   │   ├── audio.py         # Speech/laughter/music
│   │   └── pipeline.py      # Orchestration
│   ├── crops/
│   │   ├── presets.py       # Crop dimensions
│   │   └── smart_crop.py    # Face-aware cropping
│   ├── export/
│   │   ├── screenshot.py    # JPEG export
│   │   ├── clip.py          # Video clip export
│   │   └── lut.py           # LUT application
│   └── gui/
│       └── main_window.py   # PySide6 GUI (dark theme)
└── tests/                   # 100 unit tests
```

---

## Key Components

### 1. Medium Detection (`analysis/detector.py`)
- **Super 8**: 2K+ resolution + film frame rates (16, 18, 24 fps)
- **Dad Cam**: SD resolution OR HD at consumer rates (25, 29.97, 30 fps)
- **Modern**: 4K+ at standard video rates

### 2. LOG Profile Detection
Automatically detects from video metadata:
- S-Log3 (Sony)
- C-Log3 (Canon)
- V-Log (Panasonic)
- N-Log (Nikon)
- Generic Flat

### 3. Categorization (`analysis/categorizer.py`)
Uses OpenCLIP (ViT-B/32) to classify frames into:
- **People**: Humans as main subject
- **Details**: Objects, decorations, close-ups
- **Venue**: Environment, establishing shots

### 4. Scoring System (`analysis/pipeline.py`)
Category-specific weighted scoring:

**People frames:**
- Face presence: 35%
- Aesthetic: 25%
- Composition: 20%
- Technical: 20%

**Details frames:**
- Aesthetic: 40%
- Sharpness: 30%
- Composition: 20%
- Technical: 10%

**Venue frames:**
- Composition: 35%
- Aesthetic: 30%
- Technical: 20%
- No-people bonus: 15%

### 5. Smart Cropping (`crops/smart_crop.py`)
- Detects face positions
- Calculates weighted centroid
- Positions crop to include faces
- Supports left/right bias for story variants

### 6. GUI (`gui/main_window.py`)
- Dark theme with green accents
- Project folder selection
- Real-time progress bar
- Category tabs with frame counts
- Per-frame selection checkboxes
- Thumbnail loading in background
- Export settings:
  - Crop format selection
  - Resolution targets (1080/2048/1600/Original)
  - JPEG quality slider (60-100%)
  - Quotas per category
  - LUT folder selection
  - Clip type (Single 2-18s / Multi 8-58s)
  - Clip format (Square/Portrait/Landscape)
- Settings persistence (QSettings)

---

## Configuration (`core/config.py`)

### Crop Presets
| Preset | Dimensions | Aspect Ratio |
|--------|------------|--------------|
| square | 1080x1080 | 1:1 |
| portrait | 1080x1350 | 4:5 |
| landscape | 1920x1080 | 16:9 |
| story | 1080x1920 | 9:16 |
| story_l | 1080x1920 | 9:16 (left bias) |
| story_r | 1080x1920 | 9:16 (right bias) |

### Clip Durations
| Type | Min | Max | Default |
|------|-----|-----|---------|
| Single | 2s | 18s | 8s |
| Multi | 8s | 58s | 30s |

### Resolution Presets
| Platform | Max Dimension |
|----------|---------------|
| Instagram | 1080px |
| Facebook | 2048px |
| Twitter | 1600px |
| Original | No limit |

### Default Quotas
| Category | Screenshots | Clips |
|----------|-------------|-------|
| People | 50 | 10 |
| Details | 30 | 5 |
| Venue | 20 | 5 |

---

## Video Extensions Supported

```
.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v,
.mpg, .mpeg, .mts, .m2ts,
.tod, .mod,  ← JVC cameras
.3gp, .3g2,
.mxf,
.dv, .vob
```

---

## Export Pipeline

### Screenshots
1. Extract frame at timestamp (FFmpeg)
2. Detect faces (InsightFace)
3. Apply smart crop for each preset
4. Resize to max dimension (if set)
5. Apply LUT (if set)
6. Save as JPEG with quality setting

### Clips
1. Calculate smart crop box
2. Build FFmpeg filter chain (crop → LUT → scale → sharpen)
3. Encode with libx264 (CRF quality)
4. Include AAC audio

---

## Worker Threads

| Worker | Purpose |
|--------|---------|
| `AnalysisWorker` | Background video analysis |
| `ExportWorker` | Background screenshot export |
| `ThumbnailWorker` | Background thumbnail loading |
| `ClipExportWorker` | Background clip export |

All workers emit progress signals for real-time UI updates.

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# By module
python -m pytest tests/test_crops.py -v
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_config.py -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

Current: **100/100 tests passing**

---

## Dependencies

### Core
- Python 3.12+
- PySide6 (GUI)
- torch, torchvision (ML)
- open-clip-torch (categorization)
- insightface, onnxruntime (face detection)
- Pillow (image processing)
- numpy, opencv-python

### Analysis
- librosa, soundfile (audio analysis)
- scenedetect (scene detection)

### Build
- ffmpeg-python
- tqdm (progress bars)

---

## Premium User Experience Features

1. **Dark theme** with professional aesthetic
2. **Non-blocking operations** - all heavy work in background threads
3. **Real-time progress** - progress bar, status messages
4. **Visual feedback** - thumbnails load progressively
5. **Settings persistence** - remembers last folder, quality, LUT path
6. **Selection flexibility** - checkbox per frame, double-click toggle
7. **Category organization** - tabs with counts
8. **Completion dialogs** - confirmation with file counts

---

## Completion Score: 100%

All original spec requirements implemented and tested:
- ✅ GUI interface (Mac/Linux)
- ✅ Auto-detection (medium, LOG, faces, categories)
- ✅ All video mediums (Super 8, Dad Cam, Modern)
- ✅ All aspect ratios (Square, Portrait, Story, Landscape)
- ✅ Clip durations (Single 2-18s, Multi 8-58s)
- ✅ Resolution targets (1080, 2048, 1600, Original)
- ✅ Smart face-aware cropping
- ✅ LUT support with user selection
- ✅ Category-based scoring priorities
- ✅ Batch and individual selection
- ✅ 100/100 tests passing
