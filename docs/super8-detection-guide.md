# Super 8 Detection Implementation Guide

## Overview

This guide explains how SST detects Super 8 raw film scans vs. edited exports. The detection is critical for applying correct crop behavior (overscan preservation vs. removal).

---

## The Problem

Super 8 film scanners produce raw scans with unique characteristics. However, users may also have edited exports derived from those scans. The challenge:

| Type | Example | Resolution | FPS | Looks Like Super 8? |
|------|---------|------------|-----|---------------------|
| Raw scan | `NealB_3908.mov` | 6464×4852 | 24 | YES - should be SUPER_8 |
| Edited export | `Super 8.mov` | 3072×2304 | 24 | YES - but is NOT raw |

**The old approach failed** because it used resolution + frame rate:
```python
# WRONG: This incorrectly classifies edited exports as SUPER_8
if width >= 2000 and fps_rounded in {16, 18, 24}:
    return Medium.SUPER_8
```

Edited exports can have the same resolution and frame rate as raw scans!

---

## The Solution: Codec + Audio Detection

After analyzing 8 real-world files, we found a **100% reliable signal**:

| Property | Raw Scans | Edited Exports |
|----------|-----------|----------------|
| **Codec** | ProRes (always) | H.264/HEVC |
| **Audio streams** | 0 (never) | 1+ (always) |

**Why this works:**
1. Film scanners output ProRes, DNxHD, or similar archival codecs
2. Raw scans have NO audio track (film has no sound)
3. Edited exports use H.264/HEVC and add music/audio tracks

---

## Detection Logic

Location: `src/analysis/detector.py`

### Film Scan Codecs (line ~42)

```python
FILM_SCAN_CODECS = {
    'prores',      # Apple ProRes (most common from scanners)
    'dnxhd',       # Avid DNxHD
    'dnxhr',       # Avid DNxHR
    'cfhd',        # CineForm
    'v210',        # Uncompressed 10-bit
    'r210',        # Uncompressed 10-bit RGB
}
```

### Detection Method (line ~132)

```python
def detect_medium(self, metadata: VideoMetadata, video_path: Path = None) -> Medium:
    # ... extension and codec checks for DAD_CAM ...

    # Super 8 RAW SCAN: Film scan codec + NO audio
    # This is DEFINITIVE based on factual data:
    # - Raw scans: prores codec, 0 audio streams (100% correlation)
    # - Edited exports: h264/hevc codec, 1+ audio streams
    if codec in self.FILM_SCAN_CODECS and not has_audio:
        logger.debug(
            f"Detected SUPER_8: film scan codec '{codec}' + no audio "
            f"({width}x{height} @ {fps_rounded}fps)"
        )
        return Medium.SUPER_8
```

---

## How FFprobe Extracts Metadata

The `get_video_metadata()` method calls FFprobe:

```bash
ffprobe -v quiet -print_format json -show_streams -show_format <video_path>
```

Key fields extracted:
- `codec_name` → `metadata.codec` (e.g., "prores", "h264", "hevc")
- Audio stream presence → `metadata.has_audio` (True if any audio stream exists)

---

## Validation Dataset

Detection was validated against 8 real files:

### Raw Scans (All detected as SUPER_8)

| File | Codec | Resolution | FPS | Audio |
|------|-------|------------|-----|-------|
| NealB_3908-3903 | prores | 6464×4852 | 24 | 0 |
| BryantN_s8_OCN_9167-9164 | prores | 6464×4852 | 24 | 0 |
| BryantN_s8_OCN_5326 | prores | 6464×4852 | 18 | 0 |
| NealB_5495-5493 | prores | 6464×4852 | 18 | 0 |
| BryantN_s8_Ekta_CR01,02 | prores | 6464×4852 | 24 | 0 |
| BryantN_s8_Ekta_5393 | prores | 6464×4852 | 18 | 0 |

### Edited Exports (All detected as MODERN)

| File | Codec | Resolution | FPS | Audio |
|------|-------|------------|-----|-------|
| Super 8.mov | h264 | 3072×2304 | 24 | 1 |
| Date Night.mp4 | hevc | 1600×1200 | 24 | 1 |

---

## Detection Priority Order

The `detect_medium()` method checks in this order:

1. **File extension** → `.mts`, `.tod`, `.mod` = DAD_CAM
2. **MPEG-2 codec** → `mpeg2video` = DAD_CAM (HDV camcorders)
3. **Film scan codec + no audio** → ProRes/DNx + no audio = SUPER_8
4. **4K+ resolution** → width ≥ 3840 = MODERN
5. **SD resolution** → width ≤ 720 OR height ≤ 576 = DAD_CAM
6. **HD at consumer rates** → 1080p at 30/60fps = DAD_CAM
7. **Default** → MODERN

---

## Why Audio Matters

Film scan audio state is definitive:

- **Raw scans** are direct digitizations of film strips. Film has no audio track, so raw scans have no audio stream.
- **Edited exports** are timeline renders. Editors add music, sound effects, or even silent audio tracks.

This means `has_audio == False` is a reliable indicator of unprocessed footage.

---

## Crop Behavior by Medium

Once detected, the medium affects crop handling:

| Medium | Original Preset | Square Preset | Vertical Preset |
|--------|-----------------|---------------|-----------------|
| SUPER_8 | Keep full gate with overscan | Keep full gate with overscan | Gate detection → crop from content area |
| DAD_CAM | Standard crop | Standard crop | Standard crop |
| MODERN | Standard crop | Standard crop | Standard crop |

**Super 8 overscan** includes sprocket holes and frame lines - the "film aesthetic" that users want to preserve in original/square crops, but remove for vertical social media crops.

---

## Testing Detection

To verify detection is working:

```python
from pathlib import Path
from src.analysis.detector import MediumDetector

detector = MediumDetector()

# Test a file
video_path = Path('/path/to/video.mov')
metadata = detector.get_video_metadata(video_path)
medium = detector.detect_medium(metadata, video_path)

print(f"Codec: {metadata.codec}")
print(f"Has audio: {metadata.has_audio}")
print(f"Detected medium: {medium.name}")
```

Expected output for raw scan:
```
Codec: prores
Has audio: False
Detected medium: SUPER_8
```

Expected output for edited export:
```
Codec: h264
Has audio: True
Detected medium: MODERN
```

---

## Troubleshooting

### "Edited export detected as SUPER_8"

This should no longer happen. If it does:
1. Check if the export truly has no audio track (`ffprobe -show_streams`)
2. Check if it uses ProRes codec (unusual for exports)
3. If both conditions are true, the export is indistinguishable from a raw scan

### "Raw scan detected as MODERN"

Check:
1. Does the scanner output ProRes? Some older scanners use different codecs.
2. Does the file have an audio track added? Some workflows add silent audio.

### Adding Support for New Scanner Codecs

If a scanner uses a codec not in `FILM_SCAN_CODECS`:
1. Verify the codec name with `ffprobe -show_streams`
2. Add to the `FILM_SCAN_CODECS` set in `detector.py`
3. Update this guide and `CLAUDE.md`

---

## Key Takeaways

1. **Never use resolution/FPS alone** to detect Super 8 - edited exports can match
2. **Codec + audio is definitive** for distinguishing raw vs. edited
3. **ProRes + no audio = raw scan** (100% correlation in validation dataset)
4. **H.264/HEVC + audio = edited export** (not Super 8)

---

## Reference

- Source file: `src/analysis/detector.py`
- Spec document: `CLAUDE.md` → "Three Media Types" section
- Validation data: 8 files tested (6 raw scans, 2 edited exports)

Last updated: 2025-12
