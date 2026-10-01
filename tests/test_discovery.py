import json
import tempfile
import types
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import test_layers as support
from test_layers_package.discovery import parse_objects, discovery_prompt, run_vision
nodes=support.nodes


def objects():
    return [{'name':'person','prompt':'person','bbox':[100,100,600,950],'kind':'object'},
            {'name':'floor','prompt':'floor','bbox':[0,800,1000,1000],'kind':'background'}]


class DiscoveryTests(unittest.TestCase):
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
                    run_vision(None,directory,'whole objects',24)
        self.assertEqual(events,['unload','clear','clear'])

if __name__=='__main__':unittest.main()
