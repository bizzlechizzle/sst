"""Tests for cropping functionality."""
import pytest
from PIL import Image

from src.core.models import BoundingBox, Face
from src.crops.presets import CROP_PRESETS, get_preset_info, get_all_presets
from src.crops.smart_crop import (
    smart_crop, crop_to_preset, crop_all_presets, calculate_crop_box
)


class TestPresets:
    """Tests for crop presets."""
    
    def test_all_presets_exist(self):
        expected = ['square', 'portrait', 'landscape', 'story', 'story_l', 'story_r']
        for preset in expected:
            assert preset in CROP_PRESETS
    
    def test_square_dimensions(self):
        assert CROP_PRESETS['square'] == (1080, 1080)
    
    def test_portrait_dimensions(self):
        assert CROP_PRESETS['portrait'] == (1080, 1350)
    
    def test_landscape_dimensions(self):
        assert CROP_PRESETS['landscape'] == (1920, 1080)
    
    def test_story_dimensions(self):
        assert CROP_PRESETS['story'] == (1080, 1920)
    
    def test_get_preset_info(self):
        info = get_preset_info('square')
        assert info['name'] == 'square'
        assert info['width'] == 1080
        assert info['height'] == 1080
        assert info['orientation'] == 'square'
        assert info['aspect_ratio'] == '1:1'
    
    def test_get_preset_info_portrait(self):
        info = get_preset_info('portrait')
        assert info['orientation'] == 'portrait'
        assert info['aspect_ratio'] == '4:5'
    
    def test_get_preset_info_landscape(self):
        info = get_preset_info('landscape')
        assert info['orientation'] == 'landscape'
        assert info['aspect_ratio'] == '16:9'
    
    def test_get_preset_info_invalid(self):
        with pytest.raises(ValueError):
            get_preset_info('invalid_preset')
    
    def test_get_all_presets(self):
        presets = get_all_presets()
        assert len(presets) == 6
        assert 'square' in presets


class TestSmartCrop:
    """Tests for smart cropping."""
    
    @pytest.fixture
    def wide_image(self):
        """Create a wide test image (16:9)."""
        return Image.new('RGB', (1920, 1080), color='red')
    
    @pytest.fixture
    def tall_image(self):
        """Create a tall test image (9:16)."""
        return Image.new('RGB', (1080, 1920), color='blue')
    
    @pytest.fixture
    def square_image(self):
        """Create a square test image."""
        return Image.new('RGB', (1000, 1000), color='green')
    
    def test_smart_crop_square_from_wide(self, wide_image):
        """Test cropping wide image to square."""
        result = smart_crop(wide_image, 1080, 1080)
        assert result.size == (1080, 1080)
    
    def test_smart_crop_portrait_from_wide(self, wide_image):
        """Test cropping wide image to portrait."""
        result = smart_crop(wide_image, 1080, 1350)
        assert result.size == (1080, 1350)
    
    def test_smart_crop_landscape_from_tall(self, tall_image):
        """Test cropping tall image to landscape."""
        result = smart_crop(tall_image, 1920, 1080)
        assert result.size == (1920, 1080)
    
    def test_smart_crop_with_face(self, wide_image):
        """Test that cropping considers face position."""
        # Create a face on the right side
        face = Face(
            bbox=BoundingBox(x=1500, y=400, width=200, height=250),
            confidence=0.95
        )
        
        result = smart_crop(wide_image, 1080, 1080, faces=[face])
        assert result.size == (1080, 1080)
        # The crop should be positioned toward the right
    
    def test_smart_crop_with_bias_left(self, wide_image):
        """Test left bias cropping."""
        result = smart_crop(wide_image, 1080, 1080, bias='left')
        assert result.size == (1080, 1080)
    
    def test_smart_crop_with_bias_right(self, wide_image):
        """Test right bias cropping."""
        result = smart_crop(wide_image, 1080, 1080, bias='right')
        assert result.size == (1080, 1080)


class TestCropToPreset:
    """Tests for preset-based cropping."""
    
    @pytest.fixture
    def test_image(self):
        return Image.new('RGB', (1920, 1080), color='white')
    
    def test_crop_to_square(self, test_image):
        result = crop_to_preset(test_image, 'square')
        assert result.size == (1080, 1080)
    
    def test_crop_to_portrait(self, test_image):
        result = crop_to_preset(test_image, 'portrait')
        assert result.size == (1080, 1350)
    
    def test_crop_to_landscape(self, test_image):
        result = crop_to_preset(test_image, 'landscape')
        assert result.size == (1920, 1080)
    
    def test_crop_to_story(self, test_image):
        result = crop_to_preset(test_image, 'story')
        assert result.size == (1080, 1920)
    
    def test_crop_to_invalid_preset(self, test_image):
        with pytest.raises(ValueError):
            crop_to_preset(test_image, 'invalid')


class TestCropAllPresets:
    """Tests for batch cropping."""
    
    @pytest.fixture
    def test_image(self):
        return Image.new('RGB', (1920, 1080), color='gray')
    
    def test_crop_all_default(self, test_image):
        results = crop_all_presets(test_image)
        assert len(results) == 6  # All presets
        assert 'square' in results
        assert results['square'].size == (1080, 1080)
    
    def test_crop_selected_presets(self, test_image):
        results = crop_all_presets(test_image, presets=['square', 'portrait'])
        assert len(results) == 2
        assert 'square' in results
        assert 'portrait' in results
        assert 'landscape' not in results


class TestCalculateCropBox:
    """Tests for crop box calculation."""
    
    def test_calculate_for_square(self):
        box = calculate_crop_box(1920, 1080, 1080, 1080)
        assert box.width == 1080
        assert box.height == 1080
        # Should be centered horizontally
        assert box.x == (1920 - 1080) // 2
        assert box.y == 0
    
    def test_calculate_for_portrait(self):
        box = calculate_crop_box(1920, 1080, 1080, 1350)
        # Portrait is taller than 16:9 source allows
        # So we crop full height and calculate width
        assert box.height == 1080
        expected_width = int(1080 * (1080 / 1350))
        assert box.width == expected_width
    
    def test_calculate_with_face(self):
        face = Face(
            bbox=BoundingBox(x=1600, y=300, width=200, height=250),
            confidence=0.9
        )
        box = calculate_crop_box(1920, 1080, 1080, 1080, faces=[face])
        # Box should be shifted toward the face on the right
        assert box.width == 1080
        assert box.x > (1920 - 1080) // 2  # Should be right of center
    
    def test_calculate_with_left_bias(self):
        box = calculate_crop_box(1920, 1080, 1080, 1080, bias='left')
        assert box.x == 0
    
    def test_calculate_with_right_bias(self):
        box = calculate_crop_box(1920, 1080, 1080, 1080, bias='right')
        assert box.x == 1920 - 1080  # Right edge
