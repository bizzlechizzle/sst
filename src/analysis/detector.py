"""Auto-detect video medium and LOG profile."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from ..core.models import Medium, LogProfile, VideoMetadata
from ..core.exceptions import FFmpegError

logger = logging.getLogger(__name__)


class MediumDetector:
    """Detects the type of camera used to shoot a video."""

    # Frame rates typical of film (Super 8)
    FILM_FRAME_RATES = {16, 18, 24}

    # Frame rates typical of consumer video (NTSC/PAL)
    CONSUMER_FRAME_RATES = {25, 29.97, 30, 50, 59.94, 60}

    # File extensions that definitively indicate consumer camcorders
    # These are container formats only used by camcorders
    CAMCORDER_EXTENSIONS = {
        '.mts',   # AVCHD (Sony, Panasonic, Canon consumer HD camcorders)
        '.m2ts',  # AVCHD variant
        '.tod',   # JVC HDV camcorders
        '.mod',   # JVC SD camcorders
        '.moi',   # JVC metadata file (paired with .mod/.tod)
    }

    # Codecs that indicate consumer camcorder footage
    CAMCORDER_CODECS = {
        'mpeg2video',  # HDV format (older HD camcorders like GG1)
    }
    
    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract technical metadata from a video file using FFprobe.
        
        FFprobe is part of FFmpeg and reads video file information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata with all technical details
            
        Raises:
            FFmpegError: If FFprobe fails or returns invalid data
        """
        if not video_path.exists():
            raise FFmpegError(f"Video file not found: {video_path}")
        
        cmd = [
            'ffprobe',
            '-v', 'quiet',                    # Don't print extra info
            '-print_format', 'json',          # Output as JSON
            '-show_streams',                  # Show stream info
            '-show_format',                   # Show container info
            str(video_path)
        ]
        
        logger.debug(f"Running ffprobe on {video_path}")
        
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
            raise FFmpegError(f"No video stream found in {video_path}")
        
        # Parse frame rate (might be "24000/1001" or "24")
        fps_str = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 24.0
        else:
            fps = float(fps_str)
        
        # Get duration from format (more reliable) or stream
        duration = float(data.get('format', {}).get('duration', 0))
        if duration == 0:
            duration = float(video_stream.get('duration', 0))
        
        metadata = VideoMetadata(
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
        
        logger.info(
            f"Video: {video_path.name} - {metadata.width}x{metadata.height} "
            f"@ {metadata.fps}fps, {metadata.duration_sec:.1f}s, audio={metadata.has_audio}"
        )
        
        return metadata
    
    def detect_medium(
        self,
        metadata: VideoMetadata,
        video_path: Optional[Path] = None,
    ) -> Medium:
        """Determine the medium based on technical characteristics.

        Logic (in priority order):
        1. File extension check - camcorder formats (.mts, .tod) are always DAD_CAM
        2. Codec check - MPEG-2 video indicates HDV camcorder (DAD_CAM)
        3. Super 8 = High resolution + film frame rates (16, 18, 24 fps)
           Film scans are typically 2K, 4K, or 6K at exactly film frame rates
        4. Modern = 4K+ at standard video rates (23.976+)
        5. Dad Cam = SD resolution OR consumer HD at consumer frame rates

        Args:
            metadata: Video metadata from FFprobe
            video_path: Optional path for extension-based detection

        Returns:
            Detected Medium enum value
        """
        width = metadata.width
        height = metadata.height
        fps = metadata.fps
        codec = metadata.codec.lower() if metadata.codec else ''

        # Round fps to nearest common value for comparison
        fps_rounded = self._round_fps(fps)

        # Check file extension first - camcorder formats are definitive
        if video_path:
            ext = video_path.suffix.lower()
            if ext in self.CAMCORDER_EXTENSIONS:
                logger.debug(f"Detected DAD_CAM: camcorder extension {ext}")
                return Medium.DAD_CAM

        # Check codec - MPEG-2 video is HDV (consumer HD camcorders)
        if codec in self.CAMCORDER_CODECS:
            logger.debug(f"Detected DAD_CAM: HDV codec {codec}")
            return Medium.DAD_CAM

        # Super 8: High-res film scans at film frame rates
        # Film scans are typically 2K, 4K, or 6K at exactly 16, 18, or 24 fps
        if width >= 2000 and fps_rounded in self.FILM_FRAME_RATES:
            logger.debug(f"Detected SUPER_8: {width}px wide, {fps_rounded}fps")
            return Medium.SUPER_8

        # Modern: 4K or higher at standard video frame rates
        if width >= 3840 and fps >= 23.976:
            logger.debug(f"Detected MODERN: {width}px wide (4K+), {fps}fps")
            return Medium.MODERN

        # Dad Cam: Standard definition (SD)
        if width <= 720 or height <= 576:
            logger.debug(f"Detected DAD_CAM: SD resolution {width}x{height}")
            return Medium.DAD_CAM

        # Dad Cam: HD at consumer frame rates (interlaced or progressive)
        if width <= 1920 and fps_rounded in self.CONSUMER_FRAME_RATES:
            logger.debug(f"Detected DAD_CAM: HD at consumer {fps_rounded}fps")
            return Medium.DAD_CAM

        # Default: Assume modern camera
        logger.debug(f"Detected MODERN (default): {width}x{height} @ {fps}fps")
        return Medium.MODERN
    
    def detect_log_profile(self, metadata: VideoMetadata) -> LogProfile:
        """Detect LOG color profile from metadata.
        
        Cameras store their color profile in metadata tags.
        We check gamma (transfer function) and color space.
        
        Args:
            metadata: Video metadata from FFprobe
            
        Returns:
            Detected LogProfile enum value
        """
        gamma = (metadata.gamma or '').lower()
        color_space = (metadata.color_space or '').lower()
        
        # Sony S-Log3
        if 's-log3' in gamma or 'slog3' in gamma:
            logger.debug("Detected S-Log3 from gamma tag")
            return LogProfile.SLOG3
        if 's-gamut' in color_space:
            logger.debug("Detected S-Log3 from S-Gamut color space")
            return LogProfile.SLOG3
        
        # Canon C-Log3
        if 'c-log3' in gamma or 'clog3' in gamma:
            logger.debug("Detected C-Log3 from gamma tag")
            return LogProfile.CLOG3
        if 'cinema gamut' in color_space:
            logger.debug("Detected C-Log3 from Cinema Gamut")
            return LogProfile.CLOG3
        
        # Panasonic V-Log
        if 'v-log' in gamma or 'vlog' in gamma:
            logger.debug("Detected V-Log from gamma tag")
            return LogProfile.VLOG
        if 'v-gamut' in color_space:
            logger.debug("Detected V-Log from V-Gamut")
            return LogProfile.VLOG
        
        # Nikon N-Log
        if 'n-log' in gamma or 'nlog' in gamma:
            logger.debug("Detected N-Log from gamma tag")
            return LogProfile.NLOG
        
        # Generic flat/log profile
        if any(term in gamma for term in ['flat', 'log', 'neutral']):
            logger.debug("Detected generic flat profile")
            return LogProfile.FLAT
        
        return LogProfile.NONE
    
    def _round_fps(self, fps: float) -> float:
        """Round FPS to nearest common value.
        
        Video frame rates are often slightly off (23.976 vs 24).
        This rounds to the nearest "standard" frame rate.
        
        Args:
            fps: Actual frame rate
            
        Returns:
            Nearest standard frame rate
        """
        common_fps = [16, 18, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]
        return min(common_fps, key=lambda x: abs(x - fps))
    
    def analyze_video(self, video_path: Path) -> tuple[VideoMetadata, Medium, LogProfile]:
        """Complete analysis of a video file.

        Convenience method that runs all detection.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (metadata, medium, log_profile)
        """
        metadata = self.get_video_metadata(video_path)
        medium = self.detect_medium(metadata, video_path)
        log_profile = self.detect_log_profile(metadata)

        return metadata, medium, log_profile
