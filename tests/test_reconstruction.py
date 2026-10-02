"""Pixel tests that run without ComfyUI or PyTorch."""
import importlib.util
from pathlib import Path
import unittest
from PIL import Image

spec = importlib.util.spec_from_file_location('reconstruction_masks', Path(__file__).resolve().parents[1] / 'reconstruction.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
background_removal_mask = module.background_removal_mask


class BackgroundRemovalTests(unittest.TestCase):
    def test_fractional_foreground_is_fully_removed_without_editing_cutout(self):
        alpha = Image.new('L', (128, 128))
        alpha.paste(180, (48, 48, 80, 80))
        alpha.putpixel((47, 60), 1)
        before = alpha.tobytes()
        result = background_removal_mask(alpha, 12, 4)
        self.assertEqual(result.getpixel((47, 60)), 255)
        self.assertEqual(result.getpixel((60, 60)), 255)
        self.assertEqual(result.getpixel((36, 48)), 255)
        self.assertEqual(alpha.tobytes(), before)
        self.assertEqual(result.getpixel((0, 0)), 0)
        self.assertTrue(any(0 < x < 255 for x in result.tobytes()))

    def test_empty_mask_stays_empty(self):
        self.assertIsNone(background_removal_mask(Image.new('L', (64, 64))).getbbox())

    def test_zero_margin_and_feather_preserve_support_not_partial_alpha(self):
        alpha = Image.new('L', (16, 16))
        alpha.putpixel((0, 0), 128)
        result = background_removal_mask(alpha, 0, 0)
        self.assertEqual(result.getpixel((0, 0)), 255)
        self.assertEqual(result.getpixel((1, 0)), 0)
        self.assertEqual(result.size, alpha.size)

    def test_invalid_controls_rejected(self):
        for margin, feather in [(-1, 4), (129, 4), (12, -1), (12, 33)]:
            with self.subTest(margin=margin, feather=feather), self.assertRaises(ValueError):
                background_removal_mask(Image.new('L', (16, 16)), margin, feather)
