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


class CompletionGeometryTests(unittest.TestCase):
    def test_silhouette_envelope_avoids_unrelated_box_corner(self):
        from PIL import ImageDraw
        visible=Image.new('L',(64,64));draw=ImageDraw.Draw(visible)
        draw.polygon([(8,32),(32,8),(56,32),(32,56)],fill=255)
        occluders=Image.new('L',visible.size)
        occluders.paste(255,(28,20,36,44))
        occluders.paste(255,(8,8,16,16))
        visible.paste(0,(28,20,36,44))
        hole=module.completion_region(visible,occluders,0)
        self.assertEqual(hole.getpixel((32,32)),255)
        self.assertEqual(hole.getpixel((9,9)),0)
        self.assertEqual(hole.getpixel((20,32)),0)

    def test_crop_contains_target_and_manual_hole_at_image_edge(self):
        visible=Image.new('L',(640,384));visible.paste(255,(100,40,200,90))
        hole=Image.new('L',visible.size);hole.paste(255,(140,65,165,90))
        self.assertEqual(module.completion_crop(visible,hole,16),(22,0,278,256))
        hole.paste(255,(630,375,640,384))
        x0,y0,x1,y1=module.completion_crop(visible,hole,16)
        self.assertLessEqual(x0,100);self.assertLessEqual(y0,40)
        self.assertEqual((x1,y1),(640,384))

    def test_empty_foreground_never_invents_completion(self):
        visible=Image.new('L',(32,32));visible.paste(255,(8,8,20,20))
        self.assertIsNone(module.completion_region(visible,Image.new('L',visible.size),32).getbbox())
        with self.assertRaises(ValueError):module.completion_crop(Image.new('L',(8,8)),Image.new('L',(8,8)))


class CompletionBlendTests(unittest.TestCase):
    def test_transition_is_inside_target_and_core_is_opaque(self):
        visible=Image.new('L',(96,96));visible.paste(255,(8,8,88,88))
        hole=Image.new('L',visible.size);hole.paste(128,(40,40,56,56))
        visible.paste(0,(40,40,56,56))
        before=visible.tobytes()
        blend=module.completion_blend_mask(visible,hole,4)
        self.assertEqual(blend.getpixel((40,40)),255)
        self.assertTrue(0<blend.getpixel((39,48))<255)
        self.assertEqual(blend.getpixel((0,0)),0)
        self.assertEqual(blend.getpixel((12,12)),0)
        self.assertEqual(visible.tobytes(),before)
        self.assertEqual(module.completion_blend_mask(visible,hole,0).getpixel((39,48)),0)

    def test_blend_never_grows_object_into_empty_background(self):
        visible=Image.new('L',(64,64));visible.paste(255,(16,16,32,48))
        hole=Image.new('L',visible.size);hole.paste(255,(32,24,40,40))
        blend=module.completion_blend_mask(visible,hole,4)
        self.assertEqual(blend.getpixel((40,32)),0)
        self.assertGreater(blend.getpixel((31,32)),0)
