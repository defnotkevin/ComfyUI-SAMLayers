import json
import tempfile
import types
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import test_layers as support
from test_layers_package.discovery import parse_objects, normalize_discovery, prepare_discovery_image, discovery_prompt, run_vision
nodes=support.nodes


def objects():
    return [{'name':'person','prompt':'person','bbox':[100,100,600,950],'kind':'object'},
            {'name':'floor','prompt':'floor','bbox':[0,800,1000,1000],'kind':'background'}]


class DiscoveryTests(unittest.TestCase):
    def test_pixel_boxes_use_explicit_dimensions_even_below_1000(self):
        raw='```json [ {"bbox_2d": [0, 468, 430, 672], "label": "floor"} ] ```'
        result=normalize_discovery(raw,pixel_size=(1036,672))[0]
        self.assertEqual(result['bbox'],[0,1000*468/672,1000*430/1036,1000])
        self.assertFalse(result['enabled'])
        self.assertEqual(normalize_discovery('{"label":"person","bbox":[0,0,250,400]}',
                         pixel_size=(500,800))[0]['bbox'],[0,0,500,500])
        with self.assertRaises(ValueError):
            normalize_discovery(raw,pixel_size=(500,500))

    def test_discovery_image_has_bounded_aligned_dimensions(self):
        for size in [(1036,672),(4000,3000),(64,64)]:
            result=prepare_discovery_image(Image.new('RGB',size))
            self.assertEqual(result.width%28,0)
            self.assertEqual(result.height%28,0)
            self.assertLessEqual(result.width*result.height,1024*28*28)
        prompt=discovery_prompt('whole objects',24,(1036,672))
        self.assertIn('absolute pixels',prompt)
        self.assertNotIn('ceiling and sky',prompt)

    def test_model_aliases_do_not_relax_review_validation(self):
        raw=json.dumps({'detections':[{'label':'person','bbox_2d':[100,100,600,950]},
            {'object_name':'floor','bounding_box':{'left':0,'top':800,'right':1000,'bottom':1000}}]})
        result=normalize_discovery('Here is the JSON:\n'+raw)
        self.assertEqual(result,parse_objects(json.dumps(objects())))
        with self.assertRaises(ValueError):parse_objects(raw)
        self.assertEqual(normalize_discovery(json.dumps(objects())),result)

    def test_model_missing_names_and_bad_boxes_are_not_invented(self):
        for raw in ['```', '{"objects":[{"category":42,"bbox":[0,0,100,100]}]}',
                    '{"label":"person","bbox_2d":[0,0,2000,100]}',
                    '{"label":"person"}', '{"objects":[]}']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):normalize_discovery(raw)

    def test_vision_repairs_once_and_saves_failed_responses(self):
        for succeeds in (True,False):
            events=[]
            raw_bad='{"objects":[{"bbox":[0,0,100,100]}]}'
            test_image=Image.new('RGB',(1036,672))
            pixel_objects=objects()
            for obj in pixel_objects:
                obj['bbox']=[v*d/1000 for v,d in zip(obj['bbox'],test_image.size*2)]
            responses=iter([raw_bad,json.dumps(pixel_objects) if succeeds else raw_bad])
            mm=types.SimpleNamespace(unload_all_models=lambda:None,
                soft_empty_cache=lambda:events.append('clear'),get_torch_device=lambda:'cpu',
                throw_exception_if_processing_interrupted=lambda:None)
            fake=types.ModuleType('comfy');fake.model_management=mm
            class Inputs(dict):
                def to(self,device):return self
            class Processor:
                @classmethod
                def from_pretrained(cls,*args,**kwargs):return cls()
                def apply_chat_template(self,messages,**kwargs):
                    events.append(('messages',len(messages)))
                    return 'prompt'
                def __call__(self,**kwargs):return Inputs(input_ids=torch.zeros(1,3,dtype=torch.long))
                def batch_decode(self,*args,**kwargs):return [next(responses)]
            class Model:
                @classmethod
                def from_pretrained(cls,*args,**kwargs):
                    events.append('load');return cls()
                def to(self,device):return self
                def eval(self):return self
                def generate(self,**kwargs):return torch.zeros(1,4,dtype=torch.long)
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory)/'config.json').write_text('{}')
                modules={'comfy':fake,'comfy.model_management':mm,
                    'transformers':types.SimpleNamespace(AutoProcessor=Processor,Qwen2_5_VLForConditionalGeneration=Model),
                    'folder_paths':types.SimpleNamespace(get_temp_directory=lambda:directory)}
                with patch.dict(sys.modules,modules),patch('torch.cuda.is_available',return_value=True):
                    if succeeds:
                        self.assertEqual(run_vision(Image.new('RGB',(1036,672)),directory,'whole objects',24),parse_objects(json.dumps(objects())))
                    else:
                        with self.assertRaisesRegex(ValueError,'Raw responses saved to'):
                            run_vision(Image.new('RGB',(1036,672)),directory,'whole objects',24)
                        paths=list((Path(directory)/'samlayers_discovery').glob('failed_*.json'))
                        self.assertEqual(len(paths),1)
                        saved=json.loads(paths[0].read_text())['attempts']
                        self.assertEqual([x['response'] for x in saved],[raw_bad,raw_bad])
                self.assertEqual(events.count('load'),1)
                self.assertIn(('messages',3),events)
                self.assertEqual(events[-1],'clear')

    def test_normalization_background_and_invalid_boxes(self):
        result=parse_objects('```json\n'+json.dumps({'objects':objects()})+'\n```')
        self.assertTrue(result[0]['enabled']);self.assertFalse(result[1]['enabled'])
        for box in [[-1,0,10,10],[10,0,1,10],[0,0,1001,10],[0,0,float('nan'),10]]:
            data=objects();data[0]['bbox']=box
            with self.assertRaises(ValueError):parse_objects(json.dumps(data))
        with self.assertRaises(ValueError):parse_objects('Not JSON')

    def test_whole_object_prompt_discourages_body_parts(self):
        self.assertIn('hands, clothing and shoes',discovery_prompt('whole objects',24))
        self.assertIn('normalized from 0 to 1000',discovery_prompt('whole objects',24))

    def test_review_pauses_new_source_and_applies_selection(self):
        class Blocker:
            def __init__(self,value):pass
        catalog={'source':'abc','image':torch.zeros(1,16,24,3),'objects':parse_objects(json.dumps(objects()))}
        with patch.dict(sys.modules,{'comfy_execution.graph':types.SimpleNamespace(ExecutionBlocker=Blocker)}):
            first=nodes.LayersReviewObjects().run(catalog,'')
            self.assertIsInstance(first['result'][0],Blocker)
            state=first['ui']['object_catalog'][0]['state'];state['objects'][0]['name']='Edited person'
            applied=nodes.LayersReviewObjects().run(catalog,json.dumps(state))
            self.assertEqual(applied['result'][0]['objects'][0]['name'],'Edited person')
            catalog['source']='changed'
            self.assertIsInstance(nodes.LayersReviewObjects().run(catalog,json.dumps(state))['result'][0],Blocker)

    def test_segmentation_uses_scaled_box_and_filters_duplicates(self):
        catalog={'image':torch.zeros(1,100,200,3),'objects':objects()+[objects()[0]]}
        mask=torch.zeros(1,100,200);mask[:,10:95,20:120]=1
        with patch.object(nodes,'detect',return_value=mask) as detector:
            project=nodes.LayersSegmentObjects().run(catalog,None,None,.5,.9)[0]
        self.assertEqual(len(project['masks']),1)
        self.assertEqual(detector.call_count,2) # background unchecked
        self.assertEqual(detector.call_args.kwargs['bboxes'],[{'x':20.,'y':10.,'width':100.,'height':85.}])
        self.assertEqual(detector.call_args.args[3],'person:1')
        self.assertEqual(project['state']['layers'][0]['name'],'person')

    def test_missing_detection_is_actionable(self):
        catalog={'image':torch.zeros(1,16,24,3),'objects':objects()}
        with patch.object(nodes,'detect',return_value=torch.zeros(0,16,24)):
            with self.assertRaisesRegex(ValueError,'Correct its box'):
                nodes.LayersSegmentObjects().run(catalog,None,None,.5,.9)

    def test_vision_releases_memory_on_generation_error(self):
        events=[]
        mm=types.SimpleNamespace(unload_all_models=lambda:events.append('unload'),
            soft_empty_cache=lambda:events.append('clear'),get_torch_device=lambda:'cpu',
            throw_exception_if_processing_interrupted=lambda:None)
        fake=types.ModuleType('comfy');fake.model_management=mm
        class Inputs(dict):
            def to(self,device):return self
        class Processor:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):return cls()
            def apply_chat_template(self,*args,**kwargs):return 'prompt'
            def __call__(self,**kwargs):return Inputs(input_ids=torch.zeros(1,3,dtype=torch.long))
        class Model:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):return cls()
            def to(self,device):return self
            def eval(self):return self
            def generate(self,**kwargs):raise RuntimeError('simulated failure')
        transformers=types.SimpleNamespace(AutoProcessor=Processor,Qwen2_5_VLForConditionalGeneration=Model)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'config.json').write_text('{}')
            with patch.dict(sys.modules,{'comfy':fake,'comfy.model_management':mm,'transformers':transformers}),patch('torch.cuda.is_available',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'simulated failure'):
                    run_vision(Image.new('RGB',(1036,672)),directory,'whole objects',24)
        self.assertEqual(events,['unload','clear','clear'])

if __name__=='__main__':unittest.main()
