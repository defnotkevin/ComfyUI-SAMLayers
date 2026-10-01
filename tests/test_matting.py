import unittest
from unittest.mock import patch
import types
import sys
import torch
import test_layers as support
nodes = support.nodes
from test_layers_package.matting import refine_layer


class MattingTests(unittest.TestCase):
    def fixture(self):
        a=torch.zeros(32,48)
        a[:,16:32]=1
        a[:,14]=.2;a[:,15]=.65;a[:,32]=.65;a[:,33]=.2
        fore=torch.tensor([.85,.1,.05])
        back=torch.tensor([.1,.5,.8])
        image=a[...,None]*fore+(1-a[...,None])*back
        binary=(a>=.5).float()
        return image,a,binary,fore

    def test_recovers_fractional_edge_and_reduces_halo(self):
        rgb,truth,binary,fore=self.fixture()
        clean,alpha=refine_layer(rgb,truth,3,1.,1.)
        edge=(truth>0)&(truth<1)
        self.assertLess(float((alpha[edge]-truth[edge]).abs().mean()),.02)
        self.assertLess(float((clean[edge]-fore).abs().mean()),float((rgb[edge]-fore).abs().mean())*.15)
        # Opaque center and distant background are preserved exactly.
        self.assertTrue(torch.equal(clean[:,20:26],rgb[:,20:26]))
        self.assertTrue(torch.equal(alpha[:,:8],binary[:,:8]))
        self.assertTrue(torch.isfinite(clean).all())
        self.assertTrue(torch.all((clean>=0)&(clean<=1)))

    def test_color_refinement_cannot_punch_holes_or_grow_opaque_background(self):
        # A patterned opaque object, disconnected thin part and matching-color
        # background reproduce the failure mode of local color classification.
        torch.manual_seed(7)
        rgb=torch.rand(48,64,3)
        mask=torch.zeros(48,64);mask[8:40,10:40]=1;mask[5:42,50]=1
        rgb[12:38:2,10:14]=rgb[0,0]
        _,result=refine_layer(rgb,mask,8,1,1)
        self.assertTrue(torch.all(result[mask==1]>=.9))
        self.assertTrue(torch.all(result[mask==0]<=.1))
        self.assertTrue(torch.equal(result>.5,mask.bool()))

    def test_strength_zero_and_cleanup_only(self):
        rgb,truth,binary,_=self.fixture()
        unchanged,a=refine_layer(rgb,truth,3,0,0)
        self.assertTrue(torch.equal(unchanged,rgb));self.assertTrue(torch.equal(a,truth))
        clean,a=refine_layer(rgb,truth,3,0,1)
        self.assertTrue(torch.equal(a,truth));self.assertFalse(torch.equal(clean,rgb))
        original=rgb.clone();refine_layer(rgb,binary)
        self.assertTrue(torch.equal(original,rgb))

    def test_empty_full_thin_and_low_contrast_do_not_invent_alpha(self):
        image=torch.ones(16,24,3)*.5
        thin=torch.zeros(16,24);thin[:,10]=1
        for alpha in [torch.zeros(16,24),torch.ones(16,24),thin]:
            clean,out=refine_layer(image,alpha)
            self.assertTrue(torch.equal(out,alpha));self.assertTrue(torch.equal(clean,image))
        alpha=torch.zeros(16,24);alpha[:,8:18]=1
        clean,out=refine_layer(image,alpha)
        self.assertTrue(torch.equal(out,alpha))

    def test_project_keeps_transforms_and_changes_only_selected_layer(self):
        p=support.LayersTests().project();p['state']['layers'][0]['x']=10
        before=p['masks'].clone()
        fake=types.ModuleType('comfy');fake.model_management=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        with patch.dict(sys.modules,{'comfy':fake,'comfy.model_management':fake.model_management}):
            result,alpha=nodes.LayersMatte().run(p,1,1,.75,0)
        self.assertEqual(result['state']['layers'][0]['x'],10)
        self.assertEqual(result['state']['origin'],p['state']['origin'])
        self.assertTrue(torch.equal(alpha[1],before[1]))
        self.assertTrue(torch.equal(p['masks'],before))
        self.assertIsNot(result['state'],p['state'])
        self.assertIs(nodes.LayersMatte().run(p,3,0,0,-1)[0],p)

if __name__=='__main__':unittest.main()
