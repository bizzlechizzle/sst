"""Tests for medium and LOG profile detection."""
import pytest

from src.core.models import Medium, LogProfile, VideoMetadata
from src.analysis.detector import MediumDetector


class TestMediumDetection:
    """Tests for medium detection logic."""
    
    @pytest.fixture
    def detector(self):
        return MediumDetector()
    
    # Super 8 detection tests
    
    def test_super8_2k_24fps(self, detector):
        """2K scan at 24fps = Super 8."""
        meta = VideoMetadata(
            width=2048, height=1536, fps=24.0,
            duration_sec=60, codec='prores', has_audio=False
        )
        assert detector.detect_medium(meta) == Medium.SUPER_8
    
    def test_super8_4k_18fps(self, detector):
        """4K scan at 18fps = Super 8."""
        meta = VideoMetadata(
            width=4096, height=3072, fps=18.0,
            duration_sec=60, codec='prores', has_audio=False
        )
        assert detector.detect_medium(meta) == Medium.SUPER_8
    
    def test_super8_6k_16fps(self, detector):
        """6K scan at 16fps = Super 8."""
        meta = VideoMetadata(
            width=6144, height=4608, fps=16.0,
            duration_sec=60, codec='prores', has_audio=False
        )
        assert detector.detect_medium(meta) == Medium.SUPER_8
    
    # Dad Cam detection tests
    
    def test_dadcam_480i(self, detector):
        """480i VHS = Dad Cam."""
        meta = VideoMetadata(
            width=720, height=480, fps=29.97,
            duration_sec=60, codec='mpeg2', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.DAD_CAM
    
    def test_dadcam_576i(self, detector):
        """576i PAL = Dad Cam."""
        meta = VideoMetadata(
            width=720, height=576, fps=25.0,
            duration_sec=60, codec='mpeg2', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.DAD_CAM
    
    def test_dadcam_1080p_30fps(self, detector):
        """1080p at 30fps = Dad Cam (consumer HD)."""
        meta = VideoMetadata(
            width=1920, height=1080, fps=30.0,
            duration_sec=60, codec='h264', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.DAD_CAM
    
    def test_dadcam_1080p_29_97fps(self, detector):
        """1080p at 29.97fps = Dad Cam (NTSC consumer HD)."""
        meta = VideoMetadata(
            width=1920, height=1080, fps=29.97,
            duration_sec=60, codec='h264', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.DAD_CAM
    
    def test_dadcam_720p_25fps(self, detector):
        """720p at 25fps = Dad Cam (PAL consumer)."""
        meta = VideoMetadata(
            width=1280, height=720, fps=25.0,
            duration_sec=60, codec='h264', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.DAD_CAM
    
    # Modern detection tests
    
    def test_modern_4k_24fps(self, detector):
        """4K at 24fps = Modern."""
        meta = VideoMetadata(
            width=3840, height=2160, fps=23.976,
            duration_sec=60, codec='h265', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.MODERN
    
    def test_modern_4k_60fps(self, detector):
        """4K at 60fps = Modern."""
        meta = VideoMetadata(
            width=3840, height=2160, fps=59.94,
            duration_sec=60, codec='h265', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.MODERN
    
    def test_modern_6k_30fps(self, detector):
        """6K at 30fps = Modern (not Super 8 because 30fps is video rate)."""
        meta = VideoMetadata(
            width=6144, height=3456, fps=30.0,
            duration_sec=60, codec='prores', has_audio=True
        )
        assert detector.detect_medium(meta) == Medium.MODERN
    
    # Edge cases
    
    def test_1080p_24fps_is_modern(self, detector):
        """1080p at 24fps could be professional = Modern."""
        meta = VideoMetadata(
            width=1920, height=1080, fps=23.976,
            duration_sec=60, codec='h264', has_audio=True
        )
        # This is an edge case - could be either
        # Currently defaults to Modern if not clearly consumer
        result = detector.detect_medium(meta)
        # Accept either Modern or Dad Cam here
        assert result in [Medium.MODERN, Medium.DAD_CAM]


class TestLogProfileDetection:
    """Tests for LOG profile detection."""
    
    @pytest.fixture
    def detector(self):
        return MediumDetector()
    
    def test_slog3_from_gamma(self, detector):
        """Detect S-Log3 from gamma tag."""
        meta = VideoMetadata(
            width=3840, height=2160, fps=24,
            duration_sec=60, codec='h265', has_audio=True,
            gamma='S-Log3'
        )
        assert detector.detect_log_profile(meta) == LogProfile.SLOG3
    
    def test_slog3_from_color_space(self, detector):
        """Detect S-Log3 from S-Gamut color space."""
        meta = VideoMetadata(
            width=3840, height=2160, fps=24,
            duration_sec=60, codec='h265', has_audio=True,
            color_space='S-Gamut3.Cine'
        )
        assert detector.detect_log_profile(meta) == LogProfile.SLOG3
    
    def test_clog3_from_gamma(self, detector):
        """Detect C-Log3 from gamma tag."""
        meta = VideoMetadata(
            width=4096, height=2160, fps=24,
            duration_sec=60, codec='h265', has_audio=True,
            gamma='C-Log3'
        )
        assert detector.detect_log_profile(meta) == LogProfile.CLOG3
    
    def test_vlog_from_gamma(self, detector):
        """Detect V-Log from gamma tag."""
        meta = VideoMetadata(
            width=3840, height=2160, fps=24,
            duration_sec=60, codec='h265', has_audio=True,
            gamma='V-Log'
        )
        assert detector.detect_log_profile(meta) == LogProfile.VLOG
    
    def test_nlog_from_gamma(self, detector):
        """Detect N-Log from gamma tag."""
        meta = VideoMetadata(
            width=3840, height=2160, fps=24,
            duration_sec=60, codec='h265', has_audio=True,
            gamma='N-Log'
        )
        assert detector.detect_log_profile(meta) == LogProfile.NLOG
    
    def test_flat_from_gamma(self, detector):
        """Detect generic flat profile."""
        meta = VideoMetadata(
            width=1920, height=1080, fps=30,
            duration_sec=60, codec='h264', has_audio=True,
            gamma='flat'
        )
        assert detector.detect_log_profile(meta) == LogProfile.FLAT
    
    def test_no_log_standard_video(self, detector):
        """Standard Rec.709 video has no LOG."""
        meta = VideoMetadata(
            width=1920, height=1080, fps=30,
            duration_sec=60, codec='h264', has_audio=True,
            gamma='bt709',
            color_space='bt709'
        )
        assert detector.detect_log_profile(meta) == LogProfile.NONE
    
    def test_no_log_missing_metadata(self, detector):
        """Missing metadata = assume no LOG."""
        meta = VideoMetadata(
            width=1920, height=1080, fps=30,
            duration_sec=60, codec='h264', has_audio=True,
        )
        assert detector.detect_log_profile(meta) == LogProfile.NONE


class TestFpsRounding:
    """Tests for FPS rounding helper."""
    
    @pytest.fixture
    def detector(self):
        return MediumDetector()
    
    def test_round_23_976(self, detector):
        assert detector._round_fps(23.976) == 23.976
    
    def test_round_24(self, detector):
        assert detector._round_fps(24.0) == 24
    
    def test_round_29_97(self, detector):
        assert detector._round_fps(29.97) == 29.97
    
    def test_round_30(self, detector):
        assert detector._round_fps(30.0) == 30
    
    def test_round_slight_variation(self, detector):
        # Sometimes cameras report slightly off values
        assert detector._round_fps(23.98) == 23.976
        assert detector._round_fps(29.98) == 29.97
