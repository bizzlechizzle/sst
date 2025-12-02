# Social Screenshot Tool (SST)

Automated screenshot and video clip extraction for wedding videographers, optimized for social media platforms.

## Mission

Extract the best screenshots and video clips from wedding footage across three content categories, with intelligent cropping for all social media aspect ratios. Handles archival formats (Super 8 film scans, HDV camcorders) alongside modern 4K footage.

---

## Quick Reference

```bash
cd /Users/bryant/Documents/sst
source venv/bin/activate
python -m src.main  # GUI mode
```

---

## Core Specification

### Three Content Categories

| Category | Definition | Detection Method |
|----------|------------|------------------|
| **PEOPLE** | Humans are the main subject | Face detection (InsightFace) + CLIP classification |
| **B_ROLL** | Non-people content: details, venue, nature | CLIP classification + face penalty |
| **ARTSY** | Stylized footage: blur, light leaks, grain | CLIP classification + blur bonus |

### Sub-Categories

**PEOPLE**: bride, groom, couple, family, friends, party, kids, solo

**B_ROLL**: details, dress, flowers, venue_interior, venue_exterior, nature, food, decor

**ARTSY**: motion_blur, light_leak, bokeh, grain, flare, silhouette, reflection, abstract

### Three Media Types

| Medium | Detection Criteria | Audio |
|--------|-------------------|-------|
| **SUPER_8** | Width ≥2000px AND fps in {16, 18, 24} | NO |
| **DAD_CAM** | Extension `.mts/.tod/.mod` OR codec `mpeg2video` OR (width ≤1920 AND fps in {25, 29.97, 30, 50, 59.94, 60}) | YES |
| **MODERN** | Width ≥3840px AND fps ≥23.976 (default fallback) | YES |

### Export Quotas

| Category | Screenshots | Clips |
|----------|-------------|-------|
| PEOPLE | 50 | 10 |
| B_ROLL | 30 | 5 |
| ARTSY | 20 | 5 |

### Crop Presets

| Preset | Dimensions | Platform |
|--------|------------|----------|
| `square` | 1080×1080 | Instagram feed |
| `portrait` | 1080×1350 | Instagram feed (tall) |
| `landscape` | 1920×1080 | YouTube, web |
| `story` | 1080×1920 | Stories (center bias) |
| `story_l` | 1080×1920 | Stories (left bias) |
| `story_r` | 1080×1920 | Stories (right bias) |

### Clip Types

| Type | Default Duration | Range |
|------|------------------|-------|
| `single` | 8 seconds | 2-18s |
| `multi` | 30 seconds | 8-58s |

---

## Architecture

```
sst/
├── CLAUDE.md              # This file - authoritative spec
├── src/
│   ├── core/
│   │   ├── config.py      # CROP_PRESETS, QUOTAS, CLIP_TYPES, VIDEO_EXTENSIONS
│   │   ├── models.py      # Medium, Category, LogProfile, Video, Frame, ClipSuggestion
│   │   └── exceptions.py  # Custom exceptions
│   │
│   ├── analysis/
│   │   ├── pipeline.py    # AnalysisPipeline - main orchestrator
│   │   ├── detector.py    # MediumDetector - auto-detect video type + LOG profile
│   │   ├── face.py        # FaceDetector - InsightFace buffalo_l wrapper
│   │   ├── categorizer.py # CLIPCategorizer - ViT-L-14 zero-shot classification
│   │   ├── aesthetic.py   # AestheticScorer - LAION predictor
│   │   ├── audio.py       # AudioAnalyzer - beat/speech detection for clips
│   │   └── technical.py   # TechnicalScorer - sharpness, composition, contrast
│   │
│   ├── crops/
│   │   ├── smart_crop.py    # Face-aware cropping + Super 8 gate handling
│   │   ├── super8_gate.py   # Full gate scan frame detection
│   │   └── presets.py       # Crop preset definitions
│   │
│   ├── export/
│   │   ├── screenshot.py  # JPEG export with quality settings
│   │   ├── clip.py        # H.264 video export with FFmpeg
│   │   └── lut.py         # LUT application for LOG footage
│   │
│   └── gui/
│       └── main_window.py # PySide6 interface
│
└── tests/
```

---

## ML Pipeline Details

### Face Detection
- **Model**: InsightFace `buffalo_l`
- **Providers**: CUDA > CPU
- **Min confidence**: 0.5
- **Output**: Sorted by face area (largest first)

### Category Classification
- **Model**: OpenCLIP `ViT-L-14` + `laion2b_s32b_b82k`
- **Method**: Wedding-specific prompts averaged per category
- **Prompts per category**:
  - PEOPLE: 10 prompts (bride/groom, portraits, guests, dancing, family, etc.)
  - B_ROLL: 17 prompts (rings, flowers, cake, venue interior/exterior, nature, etc.)
  - ARTSY: 10 prompts (motion blur, light leaks, bokeh, grain, silhouettes, etc.)

### Aesthetic Scoring
- **Model**: LAION aesthetic predictor MLP
- **Input**: CLIP embeddings (768-dim)
- **Output**: Score 1.0-10.0, normalized to 0.0-1.0

### Final Score Formula

**PEOPLE:**
```python
final = aesthetic*0.25 + sharpness*0.20 + composition*0.20 + min(1.0, num_faces*0.3)*0.35
```

**B_ROLL:**
```python
face_penalty = min(0.15, num_faces * 0.05)  # Penalty for faces in B-roll
final = aesthetic*0.35 + composition*0.30 + sharpness*0.20 + contrast*0.15 - face_penalty
```

**ARTSY:**
```python
blur_bonus = max(0, 0.15 - sharpness*0.15)  # Low sharpness is intentional
final = aesthetic*0.50 + composition*0.25 + blur_bonus + contrast*0.10
```

---

## FFmpeg Export Settings

### Video Clips (H.264)
```bash
ffmpeg -y \
  -ss {start} -i {input} -t {duration} \
  -c:v libx264 \
  -profile:v high \
  -level 4.0 \
  -pix_fmt yuv420p \
  -preset medium \
  -crf 23 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  -vf "{filters}" \
  {output}
```

**Why these settings:**
- `level 4.0`: TOD/HDV camera compatibility
- `yuv420p`: Forces 8-bit (10-bit source → High 10 profile → QuickTime fails)
- `faststart`: Web streaming without full download

### HDV Anamorphic Correction
```bash
# MUST be applied BEFORE cropping
scale=1920:1080:flags=lanczos,setsar=1:1
```
Applied when source is 1440×1080 (HDV pixel aspect ratio correction).

### Screenshots (JPEG)
- Quality: 95
- Optimize: True
- Mode: RGB (convert RGBA/P)

---

## Super 8 Gate Detection

Full gate scans include sprocket holes and frame lines. Detection algorithm in `super8_gate.py`:

1. **Horizontal boundaries**: Column-wise brightness profile → left/right dark borders
2. **Vertical boundaries**: Row-wise brightness profile → frame line local minima
3. **Fallback chain**: Frame detection → threshold → gradient

**Expected aspect ratio**: ~1.33 (4:3 Super 8 frame)
**Warning**: "Unusual aspect ratio X.XX" indicates detection may have failed

---

## LOG Profile Support

Auto-detected from FFprobe metadata (`color_transfer`, `color_space`):

| Profile | Vendor | Detection |
|---------|--------|-----------|
| SLOG3 | Sony | `s-log3` in gamma OR `s-gamut` in color_space |
| CLOG3 | Canon | `c-log3` in gamma OR `cinema gamut` in color_space |
| VLOG | Panasonic | `v-log` in gamma OR `v-gamut` in color_space |
| NLOG | Nikon | `n-log` in gamma |
| FLAT | Generic | `flat`, `log`, or `neutral` in gamma |

LUTs auto-applied from `luts/` directory if present.

---

## Known Issues & Required Fixes

### 1. ~~Clip Suggestions Biased Toward PEOPLE~~ (FIXED)
**Status**: RESOLVED in pipeline.py
**Fix**: Clip suggestions now generated per-category (7 clips each) to ensure balanced representation across PEOPLE, B_ROLL, and ARTSY.

### 2. Super 8 Gate Detection Edge Frame Errors
**Location**: `src/crops/super8_gate.py:125-143`
**Symptom**: "cannot access local variable 'bottom' where it is not associated with a value"
**Frames affected**: Last few seconds of video (286s+)
**Cause**: `bottom` variable not initialized when `min_pos_bottom` stays at default
**Fix**: Add default initialization:
```python
bottom = h  # Default to full height if no frame line found
```

### 3. Aspect Ratio Warning for Wide Scans
**Symptom**: "Unusual aspect ratio 2.16" for 3072×2304 source
**Impact**: Non-blocking warning, crops may be suboptimal
**Status**: Acceptable - Super 8 detection still functions

### 4. No Audio Analysis for Super 8
**Behavior**: Clip boundaries default to visual-only (center ± 4s)
**Reason**: Intentional - film scans have no audio track
**Status**: By design

---

## Export Output Structure

```
OUTPUT_DIR/
├── screenshots/
│   ├── people/
│   │   └── {stem}_{timestamp}s_{preset}_{WxH}.jpg
│   ├── broll/
│   └── artsy/
├── clips/
│   ├── people/
│   │   ├── single/   # 8s clips
│   │   └── multi/    # 30s clips
│   ├── broll/
│   │   ├── single/
│   │   └── multi/
│   └── artsy/
│       ├── single/
│       └── multi/
└── export_manifest.json
```

---

## Video Extensions (Complete List)

```python
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
    '.mpg', '.mpeg', '.mts', '.m2ts',
    '.tod', '.mod',  # JVC camcorders - CRITICAL
    '.3gp', '.3g2',
    '.mxf',
    '.dv', '.vob',
}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| torch 2.x | ML inference |
| open-clip-torch | CLIP model |
| insightface | Face detection |
| onnxruntime | Model runtime |
| pillow | Image processing |
| opencv-python | Super 8 gate detection |
| PySide6 | GUI |
| ffmpeg (system) | Video processing |

---

## Environment Setup

```bash
# Create venv
python3.12 -m venv venv
source venv/bin/activate

# Install deps
pip install torch torchvision
pip install open-clip-torch insightface onnxruntime
pip install pillow opencv-python PySide6

# Verify FFmpeg
ffmpeg -version
```

---

## Coding Standards

- Type hints everywhere (`from __future__ import annotations`)
- Dataclasses for models
- Enums for categories/mediums (not strings)
- Pathlib, not os.path
- Logging, not print
- Explicit error handling

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| H.264 Level 4.0 | TOD/HDV camera compatibility |
| yuv420p forced | 10-bit → High 10 profile → QuickTime fails |
| faststart enabled | Web playback without full download |
| InsightFace buffalo_l | Best accuracy for wedding portraits |
| CLIP ViT-L-14 | Largest model, best zero-shot |
| Wedding-specific prompts | Generic prompts misclassify wedding content |
| HDV stretch before crop | Pixel aspect ratio must be corrected first |
| 1s sample interval | Balance coverage vs processing time |

---

## Do Not

- Hardcode video paths (use Path objects)
- Skip HDV anamorphic correction (breaks aspect ratios)
- Use yuv444p or 10-bit (breaks QuickTime playback)
- Remove faststart (breaks web streaming)
- Apply LUTs to non-LOG footage (oversaturates)
- Trust Super 8 gate detection blindly (verify aspect ratio)
- Use strings instead of enums for categories/mediums

---

## Stop and Ask When

- Task requires changing scoring formulas
- Task affects clip suggestion algorithm
- Adding new crop presets
- Modifying FFmpeg export settings
- Changing medium detection logic
