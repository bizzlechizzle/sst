"""Tests for data models."""
import pytest
from pathlib import Path

from src.core.models import (
    Medium, Category, LogProfile, AudioEventType,
    BoundingBox, Face, VideoMetadata, Video, Frame, AudioEvent,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""
    
    def test_create_bbox(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
    
    def test_center_x(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.center_x == 60  # 10 + 100/2
    
    def test_center_y(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.center_y == 45  # 20 + 50/2
    
    def test_area(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        assert bbox.area == 5000
    
    def test_right(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.right == 110
    
    def test_bottom(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.bottom == 70
    
    def test_as_tuple(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.as_tuple() == (10, 20, 100, 50)
    
    def test_as_xyxy(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.as_xyxy() == (10, 20, 110, 70)


class TestFace:
    """Tests for Face dataclass."""
    
    def test_create_face(self):
        bbox = BoundingBox(x=100, y=100, width=50, height=60)
        face = Face(bbox=bbox, confidence=0.95)
        assert face.confidence == 0.95
        assert face.size == 3000  # 50 * 60
    
    def test_face_with_landmarks(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        landmarks = {'left_eye': (30, 40), 'right_eye': (70, 40)}
        face = Face(bbox=bbox, confidence=0.9, landmarks=landmarks)
        assert face.landmarks['left_eye'] == (30, 40)


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""
    
    def test_create_metadata(self):
        meta = VideoMetadata(
            width=1920,
            height=1080,
            fps=29.97,
            duration_sec=120.5,
            codec='h264',
            has_audio=True,
        )
        assert meta.width == 1920
        assert meta.fps == 29.97
    
    def test_resolution_4k(self):
        meta = VideoMetadata(
            width=3840, height=2160, fps=24,
            duration_sec=60, codec='h265', has_audio=True
        )
        assert meta.resolution == "4K"
    
    def test_resolution_1080p(self):
        meta = VideoMetadata(
            width=1920, height=1080, fps=30,
            duration_sec=60, codec='h264', has_audio=True
        )
        assert meta.resolution == "1080p"
    
    def test_resolution_720p(self):
        meta = VideoMetadata(
            width=1280, height=720, fps=30,
            duration_sec=60, codec='h264', has_audio=True
        )
        assert meta.resolution == "720p"
    
    def test_resolution_sd(self):
        meta = VideoMetadata(
            width=720, height=480, fps=29.97,
            duration_sec=60, codec='mpeg2', has_audio=True
        )
        assert meta.resolution == "SD"
    
    def test_frame_count(self):
        meta = VideoMetadata(
            width=1920, height=1080, fps=30,
            duration_sec=10.0, codec='h264', has_audio=True
        )
        assert meta.frame_count == 300


class TestAudioEvent:
    """Tests for AudioEvent dataclass."""
    
    def test_duration(self):
        event = AudioEvent(
            type=AudioEventType.SPEECH,
            start_sec=10.0,
            end_sec=15.5,
            confidence=0.9
        )
        assert event.duration == 5.5
    
    def test_midpoint(self):
        event = AudioEvent(
            type=AudioEventType.LAUGHTER,
            start_sec=10.0,
            end_sec=14.0,
            confidence=0.8
        )
        assert event.midpoint == 12.0


class TestEnums:
    """Tests for enum values."""
    
    def test_medium_values(self):
        assert Medium.SUPER_8.name == 'SUPER_8'
        assert Medium.DAD_CAM.name == 'DAD_CAM'
        assert Medium.MODERN.name == 'MODERN'
    
    def test_category_values(self):
        assert Category.PEOPLE.name == 'PEOPLE'
        assert Category.DETAILS.name == 'DETAILS'
        assert Category.VENUE.name == 'VENUE'
    
    def test_log_profile_values(self):
        assert LogProfile.NONE.name == 'NONE'
        assert LogProfile.SLOG3.name == 'SLOG3'
        assert LogProfile.CLOG3.name == 'CLOG3'
    
    def test_audio_event_type_values(self):
        assert AudioEventType.SPEECH.value == 'speech'
        assert AudioEventType.LAUGHTER.value == 'laughter'
        assert AudioEventType.APPLAUSE.value == 'applause'
