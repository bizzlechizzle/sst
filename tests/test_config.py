"""Tests for configuration - CRITICAL: includes .tod extension test."""
import pytest

from src.core.config import (
    VIDEO_EXTENSIONS,
    CROP_PRESETS,
    CATEGORY_WEIGHTS,
    DEFAULT_QUOTAS,
    JPEG_QUALITY,
    VIDEO_CRF,
)


class TestVideoExtensions:
    """Tests for video extension configuration.
    
    CRITICAL: .tod and .mod must be included for JVC cameras!
    """
    
    def test_tod_extension_included(self):
        """CRITICAL: .tod must be in extensions for JVC cameras."""
        assert '.tod' in VIDEO_EXTENSIONS, ".tod extension missing! JVC cameras won't work!"
    
    def test_mod_extension_included(self):
        """CRITICAL: .mod must be in extensions for JVC cameras."""
        assert '.mod' in VIDEO_EXTENSIONS, ".mod extension missing! JVC cameras won't work!"
    
    def test_common_formats(self):
        """Common video formats should be supported."""
        common = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']
        for ext in common:
            assert ext in VIDEO_EXTENSIONS, f"{ext} missing from extensions"
    
    def test_avchd_formats(self):
        """AVCHD formats should be supported."""
        assert '.mts' in VIDEO_EXTENSIONS
        assert '.m2ts' in VIDEO_EXTENSIONS
    
    def test_professional_formats(self):
        """Professional formats should be supported."""
        assert '.mxf' in VIDEO_EXTENSIONS
    
    def test_legacy_formats(self):
        """Legacy formats should be supported."""
        assert '.dv' in VIDEO_EXTENSIONS
        assert '.vob' in VIDEO_EXTENSIONS
    
    def test_all_lowercase(self):
        """All extensions should be lowercase."""
        for ext in VIDEO_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} should be lowercase"
    
    def test_all_have_dot(self):
        """All extensions should start with a dot."""
        for ext in VIDEO_EXTENSIONS:
            assert ext.startswith('.'), f"Extension {ext} should start with ."


class TestCropPresets:
    """Tests for crop preset configuration."""
    
    def test_all_presets_have_dimensions(self):
        """All presets should have (width, height) tuple."""
        for name, dims in CROP_PRESETS.items():
            assert isinstance(dims, tuple), f"{name} should be tuple"
            assert len(dims) == 2, f"{name} should have 2 elements"
            assert isinstance(dims[0], int), f"{name} width should be int"
            assert isinstance(dims[1], int), f"{name} height should be int"
    
    def test_square_is_square(self):
        """Square preset should have equal dimensions."""
        w, h = CROP_PRESETS['square']
        assert w == h, "Square should be square!"
    
    def test_portrait_is_taller(self):
        """Portrait preset should be taller than wide."""
        w, h = CROP_PRESETS['portrait']
        assert h > w, "Portrait should be taller than wide"
    
    def test_landscape_is_wider(self):
        """Landscape preset should be wider than tall."""
        w, h = CROP_PRESETS['landscape']
        assert w > h, "Landscape should be wider than tall"
    
    def test_story_is_9_16(self):
        """Story preset should be 9:16 ratio."""
        w, h = CROP_PRESETS['story']
        assert w == 1080 and h == 1920, "Story should be 1080x1920"
    
    def test_story_variants_same_size(self):
        """Story variants should have same dimensions."""
        assert CROP_PRESETS['story'] == CROP_PRESETS['story_l']
        assert CROP_PRESETS['story'] == CROP_PRESETS['story_r']


class TestCategoryWeights:
    """Tests for category scoring weights."""
    
    def test_all_categories_defined(self):
        """All three categories should have weights."""
        assert 'PEOPLE' in CATEGORY_WEIGHTS
        assert 'DETAILS' in CATEGORY_WEIGHTS
        assert 'VENUE' in CATEGORY_WEIGHTS
    
    def test_weights_sum_to_one(self):
        """Weights for each category should sum to 1.0."""
        for category, weights in CATEGORY_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{category} weights sum to {total}, not 1.0"
    
    def test_people_has_face_weight(self):
        """PEOPLE category should weight face_presence heavily."""
        assert 'face_presence' in CATEGORY_WEIGHTS['PEOPLE']
        assert CATEGORY_WEIGHTS['PEOPLE']['face_presence'] >= 0.3
    
    def test_details_has_sharpness_weight(self):
        """DETAILS category should weight sharpness heavily."""
        assert 'sharpness' in CATEGORY_WEIGHTS['DETAILS']
        assert CATEGORY_WEIGHTS['DETAILS']['sharpness'] >= 0.2
    
    def test_venue_has_composition_weight(self):
        """VENUE category should weight composition heavily."""
        assert 'composition' in CATEGORY_WEIGHTS['VENUE']
        assert CATEGORY_WEIGHTS['VENUE']['composition'] >= 0.3


class TestQuotas:
    """Tests for default export quotas."""
    
    def test_people_quotas(self):
        """People category should have generous quotas."""
        assert 'people_screenshots' in DEFAULT_QUOTAS
        assert 'people_clips' in DEFAULT_QUOTAS
        assert DEFAULT_QUOTAS['people_screenshots'] >= 50
    
    def test_details_quotas(self):
        """Details category should have quotas."""
        assert 'details_screenshots' in DEFAULT_QUOTAS
        assert 'details_clips' in DEFAULT_QUOTAS
    
    def test_venue_quotas(self):
        """Venue category should have quotas."""
        assert 'venue_screenshots' in DEFAULT_QUOTAS
        assert 'venue_clips' in DEFAULT_QUOTAS
    
    def test_people_gets_most(self):
        """People should have highest screenshot quota."""
        assert DEFAULT_QUOTAS['people_screenshots'] > DEFAULT_QUOTAS['details_screenshots']
        assert DEFAULT_QUOTAS['people_screenshots'] > DEFAULT_QUOTAS['venue_screenshots']


class TestQualitySettings:
    """Tests for quality settings."""
    
    def test_jpeg_quality_reasonable(self):
        """JPEG quality should be high but not max."""
        assert 80 <= JPEG_QUALITY <= 100
    
    def test_video_crf_reasonable(self):
        """Video CRF should be reasonable quality."""
        assert 18 <= VIDEO_CRF <= 28  # 18=high quality, 28=lower quality
