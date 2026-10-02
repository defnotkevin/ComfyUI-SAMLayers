import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('test_layers_package')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location('test_layers_package.nodes', ROOT/'nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)
from test_layers_package.project import matrix, inverse_pillow, parse_state


class LayersTests(unittest.TestCase):
    def project(self):
        image = torch.zeros(1, 16, 24, 3)
        image[..., 0] = .8
        masks = torch.zeros(2, 16, 24)
        masks[0, 2:8, 2:8] = 1
        masks[1, 4:10, 10:16] = 1
        return nodes.new_project(image, masks, ['chair', 'person'])

    def test_rgba_and_masks_preserve_source(self):
        p = self.project()
        rgba, alpha, name = nodes.LayersGetLayer().run(p, 0, False)
        self.assertEqual(name, 'chair')
        self.assertEqual(tuple(rgba.shape), (1,16,24,4))
        self.assertTrue(torch.equal(alpha[0], p['masks'][0]))
        self.assertTrue(torch.equal(rgba[...,:3], p['image']))

    def test_draw_erase_and_reorder(self):
        p = self.project()
        state = json.loads(json.dumps(p['state']))
        mask = torch.zeros(16,24); mask[0,0] = 1
        state['layers'][0]['mask'] = nodes.data_url(nodes.pil(mask,'L'))
        state['layers'].reverse()
        out = nodes.edited_project(p, state)
        self.assertEqual(out['state']['layers'][1]['name'], 'chair')
        self.assertEqual(out['masks'][1].sum(), 1)
        self.assertTrue(torch.equal(out['masks'][0], p['masks'][1]))

    def test_source_invalidation_and_validation(self):
        p = self.project();state=p['state'];ids=[x['id'] for x in state['layers']]
        self.assertIsNone(parse_state(json.dumps(state),'different',ids))
        state['layers'][0]['scale']=0
        with self.assertRaises(ValueError):parse_state(json.dumps(state),p['source'],ids)

    def test_transform_and_group_visibility(self):
        p = self.project();p['state']['layers'][0]['x']=4
        composite,layers=nodes.render(p)
        self.assertEqual(layers[0,3,3,3],0)
        self.assertEqual(layers[0,3,7,3],1)
        p['state']['layers'][0]['group']='g'
        p['state']['groups']['g']={'visible':False}
        composite,_=nodes.render(p)
        self.assertEqual(composite[0,3,7,3],0)

    def test_affine_inverse(self):
        m=matrix({'x':10,'y':-4,'scale':1.5,'angle':37},100,80)
        a,b,c,d,e,f=inverse_pillow(m)
        x,y=21,32;u=m[0]*x+m[2]*y+m[4];v=m[1]*x+m[3]*y+m[5]
        self.assertAlmostEqual(a*u+b*v+c,x);self.assertAlmostEqual(d*u+e*v+f,y)

    def test_reject_wrong_mask_size_and_batch(self):
        with self.assertRaises(ValueError):nodes.read_mask(nodes.data_url(Image.new('L',(2,2))), (24,16))
        with self.assertRaises(ValueError):nodes.new_project(torch.zeros(2,16,24,3),torch.zeros(1,16,24),['x'])

    def test_save_reload_preserves_order_groups_pixels(self):
        p=self.project();p['state']['layers'].reverse();p['masks']=p['masks'].flip(0)
        p['state']['layers'][0]['group']='g';p['state']['groups']['g']={'x':3,'scale':1.3,'angle':12}
        p['background']=torch.ones_like(p['image'])*.2
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_output_directory=lambda:tmp)}):
            root=nodes.LayersSave().run(p,'../test')['result'][0]
            loaded=nodes.LayersLoad().run(root)[0]
            self.assertEqual(loaded['state']['layers'][0]['id'],'layer-1')
            self.assertEqual(loaded['state']['groups'],p['state']['groups'])
            self.assertTrue(torch.allclose(loaded['image'],p['image'],atol=1/255))
            self.assertTrue(torch.equal(nodes.render(loaded)[0],nodes.render(p)[0]))
            with self.assertRaises(ValueError):nodes.LayersLoad().run('../outside')

    def test_editor_blocks_first_run_and_refines_selected_mask(self):
        class Blocker:
            def __init__(self, value):pass
        p=self.project()
        mods={'comfy_execution.graph':types.SimpleNamespace(ExecutionBlocker=Blocker)}
        with patch.dict(sys.modules,mods):
            out=nodes.LayersEditor().run(p,'')
            self.assertIsInstance(out['result'][0],Blocker)
            state=json.loads(json.dumps(p['state']));state['refine']='layer-0'
            state['layers'][0]['positive']=[{'x':3,'y':3}]
            with patch.object(nodes,'detect',return_value=p['masks']):
                out=nodes.LayersEditor().run(p,json.dumps(state),object())
            self.assertIsInstance(out['result'][0],Blocker)
            self.assertNotIn('refine',out['ui']['layers_project'][0]['state'])

    def test_sam3_requests_multiple_instances_by_default(self):
        seen = {}
        class Clip:
            def tokenize(self, text): seen['prompt'] = text; return text
            def encode_from_tokens_scheduled(self, text): return text
        class Detect:
            @classmethod
            def execute(cls, **kwargs):
                seen.update(kwargs)
                return types.SimpleNamespace(args=(torch.ones(2,16,24), []))
        with patch.dict(sys.modules, {'comfy_extras.nodes_sam3':types.SimpleNamespace(SAM3_Detect=Detect)}):
            masks = nodes.detect(None, self.project()['image'], Clip(), 'chair')
        self.assertEqual(seen['prompt'], 'chair:64')
        self.assertTrue(seen['individual_masks'])
        self.assertEqual(len(masks), 2)

    def test_reconstruction_refresh_keeps_transforms_not_stale_masks(self):
        class Blocker:
            def __init__(self, value): pass
        p = self.project(); old = json.loads(json.dumps(p['state']))
        old['layers'][0]['x'] = 42
        p['source'] = 'regenerated'; p['state']['source'] = 'regenerated'
        new_mask = nodes.data_url(Image.new('L',(24,16),255))
        p['state']['layers'][0]['mask'] = new_mask
        with patch.dict(sys.modules, {'comfy_execution.graph':types.SimpleNamespace(ExecutionBlocker=Blocker)}):
            result = nodes.LayersEditor().run(p,json.dumps(old))
        state = result['ui']['layers_project'][0]['state']
        self.assertEqual(state['layers'][0]['x'], 42)
        self.assertEqual(state['layers'][0]['mask'], new_mask)

    def test_automatic_discovery_filters_duplicates_and_size(self):
        image = torch.zeros(1,16,24,3)
        a = torch.zeros(16,24); a[2:6,2:6] = 1
        b = torch.zeros(16,24); b[8:14,12:20] = 1
        tiny = torch.zeros(16,24); tiny[0,0] = 1
        results = [torch.stack([a,a.clone(),tiny]), torch.stack([b]),
                   torch.ones(1,16,24), torch.stack([a])]
        fake = types.ModuleType('comfy')
        fake.model_management = types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        fake.utils = types.SimpleNamespace(ProgressBar=lambda total:types.SimpleNamespace(update=lambda n:None))
        with patch.dict(sys.modules, {'comfy':fake,'comfy.model_management':fake.model_management,'comfy.utils':fake.utils}), patch.object(nodes,'detect',side_effect=results) as detector:
            masks=nodes.automatic_masks(image,None,2,.01,.9,.8,64)
        self.assertEqual(len(masks),2)
        self.assertTrue(torch.equal(masks[0],b))
        self.assertTrue(torch.equal(masks[1],a))
        self.assertEqual(detector.call_count,4)
        self.assertTrue(all('positive' in call.kwargs for call in detector.call_args_list))

    def test_automatic_discovery_stops_at_limit(self):
        image=torch.zeros(1,16,24,3); mask=torch.zeros(1,16,24);mask[:,2:8,2:8]=1
        fake=types.ModuleType('comfy')
        fake.model_management=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        fake.utils=types.SimpleNamespace(ProgressBar=lambda total:types.SimpleNamespace(update=lambda n:None))
        with patch.dict(sys.modules, {'comfy':fake,'comfy.model_management':fake.model_management,'comfy.utils':fake.utils}), patch.object(nodes,'detect',return_value=mask) as detector:
            result=nodes.LayersSAM3Auto().run(image,None,8,.002,.95,.8,1)[0]
        self.assertEqual(detector.call_count,1)
        self.assertEqual(result['state']['layers'][0]['discovery'],'automatic')
        self.assertEqual(result['state']['layers'][0]['name'],'Region 1')

    def test_local_refinement_preserves_mask_outside_crop(self):
        class Blocker:
            def __init__(self, value): pass
        p=self.project(); state=json.loads(json.dumps(p['state']))
        state['refine']='layer-0'
        state['refine_region']={'x':1,'y':1,'right':14,'bottom':15}
        state['layers'][0]['positive']=[{'x':6,'y':6},{'x':20,'y':8}]
        state['layers'][0]['negative']=[{'x':8,'y':9}]
        candidate=torch.ones(1,14,13)
        with patch.dict(sys.modules, {'comfy_execution.graph':types.SimpleNamespace(ExecutionBlocker=Blocker)}),patch.object(nodes,'detect',return_value=candidate) as detector:
            out=nodes.LayersEditor().run(p,json.dumps(state),object())
        kwargs=detector.call_args.kwargs
        self.assertEqual(kwargs['positive'],[{'x':5,'y':5}])
        self.assertEqual(kwargs['negative'],[{'x':7,'y':8}])
        self.assertEqual(tuple(detector.call_args.args[1].shape),(1,14,13,3))
        result=out['ui']['layers_project'][0]['state']
        mask=nodes.read_mask(result['layers'][0]['mask'],(24,16))
        self.assertTrue(torch.equal(mask[:,14:],p['masks'][0,:,14:]))
        self.assertEqual(float(mask[9,9]),1)
        self.assertNotIn('refine_region',result)

    def test_native_inpaint_padding_and_unchanged_pixels(self):
        image=torch.ones(1,13,19,3)*.2;mask=torch.zeros(1,13,19);mask[:,3:6,4:8]=1
        seen={}
        class Clip:
            def tokenize(self,s):return s
            def encode_from_tokens_scheduled(self,s,**kwargs):
                seen.setdefault('encoded',[]).append((s,kwargs));return s
        class Conditioning:
            def encode(self,**kwargs):seen.update(kwargs);return ('p','n',{})
        class Sampler:
            def sample(self,*args,**kwargs):seen['sampler']=args;return ({},)
        class Decode:
            def decode(self,*args):return (torch.ones(1,16,32,3),)
        fake=types.SimpleNamespace(InpaintModelConditioning=Conditioning,KSampler=Sampler,VAEDecode=Decode)
        with patch.dict(sys.modules,{'nodes':fake}):
            model=types.SimpleNamespace(model=types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config={'image_model':'flux','in_channels':96})))
            out=nodes.inpaint(image,mask,model,Clip(),types.SimpleNamespace(downscale_ratio=8),'p','n',0,2,6)
        self.assertEqual(tuple(seen['pixels'].shape),(1,16,32,3))
        self.assertEqual(seen['sampler'][3],1.0)
        self.assertFalse(seen['noise_mask'])
        self.assertEqual(seen['encoded'][0][1],{'add_dict':{'guidance':6}})
        self.assertEqual(seen['encoded'][0][0],'p')
        self.assertEqual(seen['encoded'][1][0],'')
        self.assertEqual(tuple(out.shape),tuple(image.shape))
        self.assertTrue(torch.equal(out[mask==0],image[mask==0]))
        self.assertTrue(torch.all(out[mask==1]==1))

    def test_inpaint_rejects_sd_and_non_fill_flux(self):
        for config in ({}, {'image_model':'flux','in_channels':16}, {'image_model':'sd15','in_channels':96}):
            model=types.SimpleNamespace(model=types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config=config)))
            with self.assertRaisesRegex(ValueError,'FLUX.1 Fill'):
                nodes.validate_flux_fill(model)

    def test_hidden_completion_resegments_and_preserves_visible(self):
        p=self.project();p['state']['layers'][0]['completion']=nodes.data_url(Image.new('L',(24,16),255))
        fake_comfy=types.ModuleType('comfy');fake_comfy.model_management=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        def fake_fill(image,mask,*args):return image*(1-mask[...,None])+torch.ones_like(image)*mask[...,None]
        with patch.dict(sys.modules,{'comfy':fake_comfy,'comfy.model_management':fake_comfy.model_management}),patch.object(nodes,'inpaint',side_effect=fake_fill),patch.object(nodes,'detect',return_value=torch.ones(1,16,24)):
            result,_=nodes.LayersReconstruct().run(p,None,None,None,None,None,'empty','',True,32,0,2,6)
        old=p['masks'][0]>0
        self.assertTrue(torch.equal(result['rgbs'][0][old],p['image'][0][old]))
        self.assertTrue(torch.all(result['masks'][0]==1))
        self.assertIn('background',result)
        self.assertNotEqual(result['source'],p['source'])

    def test_independent_scene_completes_cloud_and_has_no_duplicate_backdrop(self):
        image=torch.zeros(1,16,24,3);image[...,2]=1
        cloud=torch.zeros(16,24);cloud[2:6,4:12]=1
        person=torch.zeros(16,24);person[4:14,7:10]=1
        visible_cloud=cloud*(1-person)
        sky=1-torch.maximum(cloud,person)
        image[0][cloud.bool()]=1;image[0][person.bool()]=torch.tensor([1.,0.,0.])
        p=nodes.new_project(image,torch.stack([sky,visible_cloud,person]),['sky','cloud','person'])
        for layer in p['state']['layers'][:2]:layer['kind']='background'
        p['state']['layers'][1]['x']=3
        calls=[]
        def fill(image,mask,model,clip,vae,prompt,*args):
            calls.append((mask.clone(),prompt))
            color=torch.tensor([0.,0.,1.]) if prompt.startswith('Continuous sky') else torch.ones(3)
            return image*(1-mask[...,None])+color*mask[...,None]
        mm=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        comfy=types.ModuleType('comfy');comfy.model_management=mm
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':mm}),patch.object(nodes,'inpaint',side_effect=fill),patch.object(nodes,'detect',return_value=cloud[None]):
            result,base=nodes.LayersReconstruct().run(p,None,None,None,None,None,'sky with clouds','',True,0,0,2,6,0,0,'independent scene layers')
        self.assertEqual(len(calls),2)
        self.assertTrue(torch.all(calls[0][0][0][torch.maximum(cloud,person).bool()]==1))
        self.assertNotIn('sky with clouds',calls[0][1])
        self.assertTrue(torch.equal(result['masks'][1],cloud))
        self.assertTrue(torch.equal(result['rgbs'][1][visible_cloud.bool()],image[0][visible_cloud.bool()]))
        self.assertTrue(torch.all(result['masks'][0]==1))
        self.assertIsNone(result['background'])
        self.assertEqual(result['state']['layers'][1]['x'],3)
        self.assertTrue(torch.equal(p['masks'][1],visible_cloud))
        self.assertTrue(nodes.ui_payload(result,result['state'])['layers_project'][0]['reconstructed'])
        self.assertNotIn('background',nodes.ui_payload(result,result['state'])['layers_project'][0])
        # Hiding other layers reveals the clean base; no stationary cloud copy remains.
        self.assertTrue(torch.all(base[...,0]==0))
        self.assertTrue(torch.all(base[...,2]==1))
        result['state']['layers'][1]['visible']=False
        result['state']['layers'][2]['visible']=False
        rendered,_=nodes.render(result)
        self.assertTrue(torch.equal(rendered[...,:3],base))
        result['state']['layers'][0]['visible']=False
        transparent,_=nodes.render(result)
        self.assertTrue(torch.all(transparent[...,3]==0))

    def test_independent_scene_requires_completion_and_backmost_surface(self):
        mm=types.SimpleNamespace();comfy=types.ModuleType('comfy');comfy.model_management=mm
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':mm}),patch.object(nodes,'inpaint') as fill:
            for enabled in (False,True):
                with self.assertRaisesRegex(ValueError,'backmost layer'):
                    nodes.LayersReconstruct().run(self.project(),None,None,None,None,None,'','',enabled,32,0,2,6,reconstruction_mode='independent scene layers')
            base_project=self.project();base_project['state']['layers'][0]['kind']='background'
            with self.assertRaisesRegex(ValueError,'complete_hidden'):
                nodes.LayersReconstruct().run(base_project,None,None,None,None,None,'','',False,32,0,2,6,reconstruction_mode='independent scene layers')
            fill.assert_not_called()

    def test_local_completion_maps_crop_and_preserves_source_pixels(self):
        image=torch.full((1,384,640,3),.2)
        cloud=torch.zeros(384,640);cloud[40:90,100:200]=1
        person=torch.zeros_like(cloud);person[65:140,140:165]=1
        visible=cloud*(1-person)
        p=nodes.new_project(image,torch.stack([1-torch.maximum(cloud,person),visible,person]),['sky','cloud on the right','person'])
        p['state']['layers'][0]['kind']='background'
        p['state']['layers'][1]['kind']='background'
        calls=[]
        def fill(image,mask,*args):
            calls.append((tuple(image.shape),args[3],args[4]))
            return torch.ones_like(image)
        mm=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        comfy=types.ModuleType('comfy');comfy.model_management=mm
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':mm}),patch.object(nodes,'inpaint',side_effect=fill),patch.object(nodes,'detect',return_value=torch.ones(1,256,256)) as detect:
            result,_=nodes.LayersReconstruct().run(p,None,None,None,None,None,'background','person, clouds',True,0,0,2,1,0,0,'independent scene layers',30,'flat colors',16,512)
        self.assertEqual(calls[1][0],(1,512,512,3))
        self.assertEqual(calls[1][1],'A complete cloud. flat colors')
        self.assertEqual(calls[1][2],'')
        self.assertEqual(tuple(detect.call_args.args[1].shape),(1,256,256,3))
        hole=(cloud*person).bool()
        self.assertTrue(torch.equal(result['masks'][1],cloud))
        self.assertTrue(torch.equal(result['rgbs'][1][~hole],image[0][~hole]))
        self.assertTrue(torch.all(result['rgbs'][1][hole]==1))

    def test_surface_completion_keeps_fill_without_sam_rejection(self):
        p=self.project();p['state']['layers'][0].update(name='grass',kind='background')
        region=Image.new('L',(24,16));region.paste(255,(8,4,12,8))
        p['state']['layers'][0]['completion']=nodes.data_url(region)
        mm=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        comfy=types.ModuleType('comfy');comfy.model_management=mm
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':mm}),patch.object(nodes,'inpaint',side_effect=lambda image,*a:torch.ones_like(image)),patch.object(nodes,'detect') as detect:
            result,_=nodes.LayersReconstruct().run(p,None,None,None,None,None,'background','',True,0,0,2,1)
        detect.assert_not_called()
        self.assertTrue(torch.all(result['masks'][0,4:8,8:12]==1))

    def test_unchanged_object_completion_is_reported(self):
        p=self.project();p['state']['layers'][0]['completion']=nodes.data_url(Image.new('L',(24,16),255))
        mm=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        comfy=types.ModuleType('comfy');comfy.model_management=mm
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':mm}),patch.object(nodes,'inpaint',side_effect=lambda image,*a:image),patch.object(nodes,'detect',return_value=p['masks'][:1]):
            with self.assertRaisesRegex(ValueError,'added no pixels'):
                nodes.LayersReconstruct().run(p,None,None,None,None,None,'background','',True,0,0,2,1)


if __name__=='__main__':unittest.main()
