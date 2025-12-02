# Social Screenshot Tool (SST)

**Automatically extract Instagram-worthy screenshots and clips from wedding videos.**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## What It Does

Wedding videographers shoot HOURS of footage. Finding the perfect frames for Instagram is tedious. SST automates this by:

1. **Scanning your videos** for interesting moments
2. **Categorizing frames** as People, Details, or Venue
3. **Scoring quality** using AI aesthetic models
4. **Smart cropping** to keep faces in frame
5. **Batch exporting** in all social media formats

## Features

- ✅ Works with Super 8, consumer cameras, and professional cinema cameras
- ✅ Auto-detects LOG profiles (S-Log3, C-Log3, V-Log, N-Log)
- ✅ Audio-based clip suggestions (finds laughter, applause, speech endings)
- ✅ Face-aware smart cropping
- ✅ All social formats: Square, Portrait, Landscape, Stories
- ✅ Batch export with quotas
- ✅ Resume interrupted analysis

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/yourusername/social-screenshot-tool.git
cd social-screenshot-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .
```

### 2. Run

```bash
sst
```

Or:

```bash
python -m src.main
```

### 3. Use

1. Click **"Add Videos"** to select your footage
2. Click **"Analyze"** to scan for good frames
3. Review suggested frames in the grid
4. Select crops (Square, Portrait, Story, etc.)
5. Click **"Export All"** to save

## Supported Formats

### Video Files
- MP4, MOV, AVI, MKV, WMV, FLV, WebM, M4V
- MPG, MPEG, MTS, M2TS (AVCHD)
- **TOD, MOD** (JVC cameras)
- 3GP, 3G2 (mobile)
- MXF (professional)
- DV, VOB (legacy)

### Export Crops
| Name | Size | Use |
|------|------|-----|
| Square | 1080×1080 | Instagram feed |
| Portrait | 1080×1350 | Instagram feed (4:5) |
| Landscape | 1920×1080 | YouTube |
| Story | 1080×1920 | Instagram/TikTok Stories |

## Requirements

- Python 3.10 or newer
- FFmpeg installed and in PATH
- 8GB RAM minimum (16GB recommended for 4K)
- NVIDIA GPU recommended (works on CPU, just slower)

## Documentation

- [TECHGUIDE.md](TECHGUIDE.md) - Installation and troubleshooting
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Developer guide
- [CLAUDE.md](CLAUDE.md) - Project architecture

## License

MIT License - see LICENSE file for details.

## Credits

Built with:
- [PySide6](https://www.qt.io/qt-for-python) - GUI framework
- [CLIP](https://github.com/openai/CLIP) - Image classification
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [FFmpeg](https://ffmpeg.org/) - Video processing
