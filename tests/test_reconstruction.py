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


class QualityRegressionTests(unittest.TestCase):
    def test_color_matching_preserves_texture_and_corrects_offset(self):
        import numpy as np
        yy,xx=np.mgrid[:128,:160]
        texture=.02*np.sin(xx*1.7)+.01*np.cos(yy*2.1)
        source=np.stack([.3+texture,.5+texture,.6+texture],-1).astype('float32')
        generated=source+np.array([.06,-.04,.025],dtype='float32')
        known=np.ones((128,160),dtype='float32');known[24:104,32:128]=0
        # The erased original is deliberately unrelated to the desired output.
        damaged=source.copy();damaged[known==0]=[1,0,0]
        matched=module.match_boundary_colors(damaged,generated,known)
        self.assertLess(float(np.abs(matched-source).max()),2e-5)
        self.assertGreater(float(matched[40:80,50:100,0].std()),.01)

    def test_color_matching_uses_spatial_context_for_lighting_gradient(self):
        import numpy as np
        yy,xx=np.mgrid[:64,:64]
        source=np.full((64,64,3),.5,dtype='float32')
        offset=(xx/63*.1-.05).astype('float32')
        generated=source+offset[...,None]
        known=np.ones((64,64),dtype='float32');known[16:48,16:48]=0
        matched=module.match_boundary_colors(source,generated,known)
        self.assertLess(float(np.abs(matched-source).max()),.002)

    def test_no_reference_does_not_invent_a_palette_and_large_shifts_are_bounded(self):
        import numpy as np
        source=np.zeros((32,32,3),dtype='float32');gen=np.full_like(source,.8)
        self.assertTrue(np.array_equal(module.match_boundary_colors(source,gen,np.zeros((32,32))),gen))
        self.assertTrue(np.array_equal(module.match_boundary_colors(source,gen,np.ones((32,32))),gen))

    def test_automatic_completion_keeps_round_shape_beyond_estimated_hole(self):
        from PIL import ImageDraw
        visible=Image.new('L',(80,64));ImageDraw.Draw(visible).ellipse((10,20,40,44),fill=255)
        predicted=visible.copy();ImageDraw.Draw(predicted).ellipse((25,12,60,48),fill=255)
        # Disconnected second instance must not be adopted.
        ImageDraw.Draw(predicted).ellipse((66,2,78,12),fill=255)
        hole=Image.new('L',visible.size);hole.paste(255,(34,24,44,40))
        allowed=Image.new('L',visible.size,255)
        merged=module.merge_completion_alpha(visible,predicted,hole,allowed)
        self.assertEqual(merged.getpixel((53,30)),255)
        self.assertEqual(merged.getpixel((72,7)),0)
        self.assertEqual(merged.getpixel((3,30)),0)
        manual=module.merge_completion_alpha(visible,predicted,hole,allowed,manual=True)
        self.assertEqual(manual.getpixel((53,30)),0)
        self.assertEqual(manual.getpixel((40,30)),255)

    def test_narrow_join_is_repaired_but_distant_gap_and_manual_region_are_preserved(self):
        visible=Image.new('L',(80,64));visible.paste(255,(8,8,40,56))
        hole=Image.new('L',visible.size);hole.paste(255,(41,8,60,56))
        predicted=Image.new('L',visible.size);predicted.paste(255,(8,8,60,56))
        allowed=Image.new('L',visible.size);allowed.paste(255,(38,6,64,58))
        merged=module.merge_completion_alpha(visible,predicted,hole,allowed)
        self.assertEqual(merged.getpixel((40,30)),255)
        self.assertEqual(merged.getpixel((65,30)),0)
        manual=module.merge_completion_alpha(visible,predicted,hole,allowed,manual=True)
        self.assertEqual(manual.getpixel((40,30)),0)

    def test_manual_completion_can_join_across_original_segmentation_gap(self):
        visible=Image.new('L',(32,32));visible.paste(255,(4,4,12,28))
        hole=Image.new('L',visible.size);hole.paste(255,(12,4,20,28))
        predicted=Image.new('L',visible.size,255)
        merged=module.merge_completion_alpha(visible,predicted,hole,predicted,manual=True)
        self.assertEqual(merged.getpixel((15,15)),255)
        self.assertEqual(merged.getpixel((21,15)),0)


class MaterialBoundaryTests(unittest.TestCase):
    def test_new_white_shape_does_not_pick_up_blue_from_old_background(self):
        import numpy as np
        source=np.full((96,96,3),[.35,.55,.95],dtype='float32')
        generated=source.copy();source[25:50,40:65]=1;generated[15:70,30:75]=1
        known=np.ones((96,96),dtype='float32');known[40:65,40:65]=0
        matched=module.match_boundary_colors(source,generated,known)
        self.assertTrue(np.allclose(matched[15:70,30:75],1))

    def test_disagreement_rejection_still_corrects_small_valid_color_offset(self):
        import numpy as np
        src=np.full((64,64,3),.6,dtype='float32')
        gen=np.full_like(src,.55);src[20:25,20:25]=0
        known=np.ones((64,64),dtype='float32');known[28:50,28:50]=0
        matched=module.match_boundary_colors(src,gen,known)
        self.assertLess(float(abs(matched[32:45,32:45]-.6).max()),1e-5)
