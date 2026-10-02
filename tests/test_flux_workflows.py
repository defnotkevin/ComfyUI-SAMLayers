import json
from pathlib import Path
import unittest


class FluxWorkflowTests(unittest.TestCase):
    def test_all_reconstruction_examples_wire_flux_and_keep_sam_separate(self):
        root = Path(__file__).resolve().parents[1] / 'examples'
        count = 0
        for path in root.glob('*.json'):
            graph = json.loads(path.read_text())
            nodes = {n['id']: n for n in graph['nodes']}
            links = {link[0]: link for link in graph['links']}
            for node in nodes.values():
                for slot, inp in enumerate(node.get('inputs', [])):
                    if inp.get('link') is not None:
                        link = links[inp['link']]
                        self.assertEqual((link[3], link[4]), (node['id'], slot), path.name)
                        self.assertIn(link[0], nodes[link[1]]['outputs'][link[2]]['links'])
                if node['type'] != 'LayersReconstruct':
                    continue
                count += 1
                def source(name):
                    inp = next(i for i in node['inputs'] if i['name'] == name)
                    link = links[inp['link']]
                    return nodes[link[1]], link[2]
                for name, kind in [('model','UNETLoader'),('clip','DualCLIPLoader'),('vae','VAELoader')]:
                    loader, slot = source(name)
                    self.assertEqual(loader['type'], kind, path.name)
                    self.assertEqual(slot, 0)
                self.assertEqual(source('model')[0]['widgets_values'][0], 'flux1-fill-dev.safetensors')
                self.assertEqual(source('clip')[0]['widgets_values'][2], 'flux')
                self.assertEqual(source('sam_model')[0]['widgets_values'][0], 'sam3.1_multiplex_fp16.safetensors')
                self.assertEqual(node['widgets_values'][7], 1)
        self.assertEqual(count, 4)
