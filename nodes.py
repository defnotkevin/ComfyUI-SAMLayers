"""SAM3 layers, non-destructive editor, native ComfyUI inpainting and exports."""
import base64
import hashlib
import io
import json
import re
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from .project import (VERSION, MAX_LAYERS, parse_state, layer_matrix,
                      inverse_pillow, visible)
from .reconstruction import background_removal_mask, completion_region, completion_crop

CATEGORY = 'Layers'


def pil(tensor, mode=None):
    arr = tensor.detach().cpu().numpy()
    return Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).convert(mode) if mode else Image.fromarray((np.clip(arr, 0, 1)*255).round().astype(np.uint8))


def tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255)


def data_url(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def read_mask(value, size):
    if not isinstance(value, str) or not value.startswith('data:image/png;base64,') or len(value) > 48_000_000:
        raise ValueError('Invalid or oversized mask PNG')
    raw = base64.b64decode(value.split(',', 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as img:
        if img.size != size:
            raise ValueError('Mask dimensions do not match the source')
        # Masks are exported as grayscale RGB, not transparent alpha.
        return tensor(img.convert('L'))


def outputs(result):
    if isinstance(result, tuple):
        return result
    if hasattr(result, 'args'):
        return result.args
    raise RuntimeError('Unsupported ComfyUI node output API; update ComfyUI.')


def detect(model, image, clip=None, prompt='', positive=None, negative=None, threshold=.5, bboxes=None):
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except ImportError as exc:
        raise RuntimeError('This package needs ComfyUI with native SAM3_Detect support.') from exc
    conditioning = None
    if clip is not None and prompt:
        # Native SAM3 defaults to one detection per text prompt. Explicitly
        # request multiple instances unless the user supplied a :N limit.
        detection_prompt = prompt if re.search(r':\d+\s*$', prompt) else f'{prompt}:{MAX_LAYERS}'
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(detection_prompt))
    result = SAM3_Detect.execute(model=model, image=image, conditioning=conditioning, bboxes=bboxes,
        positive_coords=json.dumps(positive) if positive else None,
        negative_coords=json.dumps(negative) if negative else None,
        threshold=threshold, refine_iterations=1 if positive or negative else 2, individual_masks=True)
    masks = outputs(result)[0].detach().cpu().float()
    if masks.ndim != 3 or tuple(masks.shape[-2:]) != tuple(image.shape[1:3]):
        raise ValueError('Unexpected SAM3 mask shape')
    return masks


def hash_project(image, masks, names):
    h = hashlib.sha256()
    h.update(image.detach().cpu().contiguous().numpy().tobytes())
    h.update(masks.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps(names).encode())
    return h.hexdigest()


def new_project(image, masks, names):
    if len(image) != 1:
        raise ValueError('Use one source image per layer project.')
    if not 1 <= len(masks) <= MAX_LAYERS:
        raise ValueError(f'Expected 1–{MAX_LAYERS} object masks.')
    if tuple(masks.shape[1:]) != tuple(image.shape[1:3]):
        raise ValueError('Image and masks must have matching dimensions.')
    image = image[..., :3].detach().cpu().float()
    masks = masks.detach().cpu().float().clamp(0, 1)
    source = hash_project(image, masks, names)
    h, w = image.shape[1:3]
    layers = [dict(id=f'layer-{i}', name=names[i] if i < len(names) else f'Object {i+1}',
                   x=0, y=0, scale=1, angle=0, visible=True, group=None,
                   mask=data_url(pil(mask, 'L')), positive=[], negative=[])
              for i, mask in enumerate(masks)]
    state = dict(version=VERSION, source=source, origin=source, width=w, height=h, layers=layers, groups={})
    return dict(source=source, image=image, masks=masks, state=state)


def ui_payload(project, state, review_required=False):
    payload = dict(state=state, review_required=review_required, image=data_url(pil(project['image'][0], 'RGB')),
                   reconstructed=bool(project.get('rgbs') is not None or project.get('background') is not None))
    if project.get('background') is not None:
        payload['background'] = data_url(pil(project['background'][0], 'RGB'))
    if project.get('rgbs') is not None:
        by_id = {x['id']: i for i, x in enumerate(project['state']['layers'])}
        payload['rgbs'] = [data_url(pil(project['rgbs'][by_id[x['id']]], 'RGB')) for x in state['layers']]
    return {'layers_project': [payload]}


def edited_project(project, state):
    w, h = state['width'], state['height']
    original = {x['id']: i for i, x in enumerate(project['state']['layers'])}
    result = dict(project)
    result['masks'] = torch.stack([read_mask(x['mask'], (w, h)) for x in state['layers']])
    if project.get('rgbs') is not None:
        result['rgbs'] = torch.stack([project['rgbs'][original[x['id']]] for x in state['layers']])
    result['state'] = state
    return result


class LayersSAM3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'sam_model': ('MODEL',), 'sam_clip': ('CLIP',),
                'objects': ('STRING', {'multiline': True, 'default': 'person\nchair'}),
                'threshold': ('FLOAT', {'default': .5, 'min': 0, 'max': 1, 'step': .01})}}
    RETURN_TYPES = ('LAYERS_PROJECT',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, image, sam_model, sam_clip, objects, threshold):
        if len(image) != 1:
            raise ValueError('Select one source image.')
        prompts = [p.strip() for p in objects.splitlines() if p.strip()]
        if not prompts or len(prompts) > MAX_LAYERS:
            raise ValueError('Named detection needs one object description per line (maximum 64). For detection without names, open sam3_layers_auto_edit.json or use Layers • SAM3 Automatic Regions.')
        masks, names = [], []
        for prompt in prompts:
            found = detect(sam_model, image, sam_clip, prompt, threshold=threshold)
            found = [m for m in found if m.max().item() > 0]
            for i, mask in enumerate(found):
                masks.append(mask)
                label = re.sub(r':\d+\s*$', '', prompt).strip()
                names.append(label if len(found) == 1 else f'{label} #{i+1}')
        if not masks:
            raise ValueError('SAM3 found no objects. Change the descriptions or threshold.')
        return (new_project(image, torch.stack(masks), names),)


def automatic_masks(image, sam_model, points_per_side=8, min_area=.002,
                    max_area=.95, duplicate_iou=.8, max_layers=64):
    """Sample independent point prompts, retaining distinct foreground regions.

    This is a discovery heuristic, not SAM3 semantic all-object enumeration.
    Native SAM3_Detect does not expose a cached/batched point-grid generator.
    """
    import comfy.model_management
    import comfy.utils
    if len(image) != 1:
        raise ValueError('Select one source image.')
    if not 2 <= points_per_side <= 16 or not 1 <= max_layers <= MAX_LAYERS:
        raise ValueError('Invalid automatic discovery limits.')
    if not 0 <= min_area < max_area <= 1 or not 0 < duplicate_iou <= 1:
        raise ValueError('Invalid automatic mask filters.')
    h, w = image.shape[1:3]
    found, signatures = [], []
    progress = comfy.utils.ProgressBar(points_per_side**2)
    for row in range(points_per_side):
        for col in range(points_per_side):
            comfy.model_management.throw_exception_if_processing_interrupted()
            point = {'x': min(w-1, int((col+.5)*w/points_per_side)),
                     'y': min(h-1, int((row+.5)*h/points_per_side))}
            candidates = detect(sam_model, image, positive=[point])
            for candidate in candidates:
                mask = candidate > .5
                area = float(mask.float().mean())
                if not min_area <= area <= max_area:
                    continue
                # Compact signatures limit CPU/memory cost of duplicate comparisons.
                small = torch.nn.functional.interpolate(mask[None,None].float(),
                        size=(128,128), mode='nearest')[0,0] > .5
                if not small.any():
                    continue
                if any(float((small & old).sum()) / max(1, int((small | old).sum())) >= duplicate_iou
                       for old in signatures):
                    continue
                found.append(candidate)
                signatures.append(small)
                if len(found) == max_layers:
                    break
            progress.update(1)
            if len(found) == max_layers:
                break
        if len(found) == max_layers:
            break
    if not found:
        raise ValueError('No automatic masks passed the filters. Increase grid density or lower min_area; named detection is also available.')
    # Larger regions behind smaller parts makes overlapping proposals inspectable.
    return torch.stack(sorted(found, key=lambda mask: float(mask.sum()), reverse=True))


class LayersSAM3Auto:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'sam_model': ('MODEL',),
            'points_per_side': ('INT', {'default': 8, 'min': 2, 'max': 16}),
            'min_area': ('FLOAT', {'default': .002, 'min': 0, 'max': .5, 'step': .001}),
            'max_area': ('FLOAT', {'default': .95, 'min': .01, 'max': 1, 'step': .01}),
            'duplicate_iou': ('FLOAT', {'default': .8, 'min': .1, 'max': 1, 'step': .05}),
            'max_layers': ('INT', {'default': 64, 'min': 1, 'max': MAX_LAYERS})}}
    RETURN_TYPES = ('LAYERS_PROJECT',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, image, sam_model, points_per_side, min_area, max_area, duplicate_iou, max_layers):
        masks = automatic_masks(image, sam_model, points_per_side, min_area, max_area, duplicate_iou, max_layers)
        project = new_project(image, masks, [f'Region {i+1}' for i in range(len(masks))])
        for layer in project['state']['layers']:
            layer['discovery'] = 'automatic'
        return (project,)


class LayersDiscoverObjects:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',),
            'model_folder': ('STRING', {'default': 'Qwen2.5-VL-3B-Instruct'}),
            'detail': (['whole objects', 'detailed parts'],),
            'max_objects': ('INT', {'default': 24, 'min': 1, 'max': 64})},
            'optional': {'scene_scope': (['full scene', 'foreground objects'], {'default': 'full scene'})}}
    RETURN_TYPES = ('LAYERS_OBJECTS',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, image, model_folder, detail, max_objects, scene_scope='full scene'):
        import folder_paths
        from .discovery import run_vision
        if len(image) != 1:
            raise ValueError('Use one image for object discovery.')
        root = (Path(folder_paths.models_dir)/'LLM').resolve()
        path = (root/model_folder).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Model folder must be inside ComfyUI/models/LLM/.')
        objects = run_vision(pil(image[0], 'RGB'), path, detail, max_objects, scene_scope)
        cpu = image[...,:3].detach().cpu().float()
        signature = hashlib.sha256(cpu.contiguous().numpy().tobytes()+json.dumps(objects).encode()+scene_scope.encode()).hexdigest()
        return ({'image':cpu,'objects':objects,'source':signature},)


class LayersReviewObjects:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'catalog': ('LAYERS_OBJECTS',),
            'object_state': ('STRING', {'default':'','multiline':True})},
            'optional': {'sam_model': ('MODEL',)}}
    RETURN_TYPES = ('LAYERS_OBJECTS',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, catalog, object_state, sam_model=None):
        from comfy_execution.graph import ExecutionBlocker
        from .discovery import parse_objects
        saved = json.loads(object_state or '{}')
        if not isinstance(saved, dict):
            raise ValueError('Invalid object review state.')
        waiting = saved.get('source') != catalog['source']
        objects = catalog['objects'] if waiting else (parse_objects(json.dumps(saved.get('objects'))) if saved.get('objects') != [] else [])
        state = {'source':catalog['source'],'objects':objects}
        payload = {'state':state,'image':data_url(pil(catalog['image'][0], 'RGB')),'review_required':waiting}
        request = saved.get('detect_object') if not waiting else None
        if request:
            if sam_model is None:
                raise ValueError('Connect the SAM3 checkpoint MODEL output to Review Objects sam_model for click detection.')
            h,w = catalog['image'].shape[1:3]
            def points(key):
                values = request.get(key, [])
                if not isinstance(values, list) or len(values) > 128:
                    raise ValueError('Use at most 128 clicks per type.')
                for point in values:
                    if not isinstance(point, dict) or any(isinstance(point.get(k), bool) or
                        not isinstance(point.get(k), (float,int)) or not np.isfinite(point[k]) for k in ('x','y')):
                        raise ValueError('Invalid click coordinates.')
                    if not (0 <= point['x'] < w and 0 <= point['y'] < h):
                        raise ValueError('Click lies outside the source image.')
                return values
            positive, negative = points('positive'), points('negative')
            if not positive:
                raise ValueError('Place at least one positive click inside the object.')
            found = detect(sam_model, catalog['image'], positive=positive, negative=negative)
            if not len(found) or not found.max().item():
                raise ValueError('SAM found no object. Adjust the include/exclude clicks and try again.')
            scores = sum(found[:,int(p['y']),int(p['x'])] for p in positive)
            scores -= sum((found[:,int(p['y']),int(p['x'])] for p in negative), torch.zeros(len(found)))
            mask = found[scores.argmax()]
            ys,xs = torch.where(mask > .5)
            if not len(xs):
                raise ValueError('SAM returned an empty object mask. Adjust your clicks.')
            payload['object_preview'] = {'id':request.get('id'), 'mask':data_url(pil(mask,'L')),
                'bbox':[float(xs.min())/w*1000,float(ys.min())/h*1000,
                        (float(xs.max())+1)/w*1000,(float(ys.max())+1)/h*1000]}
            payload['review_required'] = True
            waiting = True
        result = ExecutionBlocker(None) if waiting else dict(catalog,objects=objects)
        return {'ui':{'object_catalog':[payload]},'result':(result,)}


class LayersSegmentObjects:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'catalog': ('LAYERS_OBJECTS',), 'sam_model': ('MODEL',), 'sam_clip': ('CLIP',),
            'threshold': ('FLOAT', {'default':.5,'min':0,'max':1,'step':.01}),
            'duplicate_iou': ('FLOAT', {'default':.9,'min':.5,'max':1,'step':.01})}}
    RETURN_TYPES = ('LAYERS_PROJECT',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, catalog, sam_model, sam_clip, threshold, duplicate_iou):
        from .discovery import parse_objects
        selected = [item for item in parse_objects(json.dumps(catalog['objects'])) if item['enabled']]
        if not selected:
            raise ValueError('Select at least one object in Review Objects, then Apply & Run.')
        # Preserve relative depth within each kind, but keep scene surfaces behind subjects.
        selected.sort(key=lambda item: item['kind'] != 'background')
        image = catalog['image']; h,w = image.shape[1:3]
        masks, names, kinds = [], [], []
        for item in selected:
            x0,y0,x1,y1 = item['bbox']
            box = {'x':x0*w/1000,'y':y0*h/1000,'width':(x1-x0)*w/1000,'height':(y1-y0)*h/1000}
            # Keep candidates until their overlap with this instance's box is scored.
            # A :1 text limit can return the same highest-confidence cloud for both boxes.
            prompt = re.sub(r':\d+\s*$', '', item['prompt'])+f':{MAX_LAYERS}'
            if item.get('confirmed_mask'):
                mask = read_mask(item['confirmed_mask'], (w,h))
            else:
                found = detect(sam_model,image,sam_clip,prompt,threshold=threshold,bboxes=[box])
                if not len(found) or not found.max().item():
                    raise ValueError(f"SAM3 found no mask for {item['name']}. Correct its box/description or uncheck it in Review Objects.")
                # One prompt + box describes one instance. Never union unrelated candidates.
                area = torch.zeros((h,w));area[int(y0*h/1000):max(int(y0*h/1000)+1,int(y1*h/1000)),int(x0*w/1000):max(int(x0*w/1000)+1,int(x1*w/1000))]=1
                score = (found*area).sum((1,2))/(found+area-found*area).sum((1,2)).clamp_min(1)
                mask = found[score.argmax()]
            binary = mask>.5
            if any(float((binary & (old>.5)).sum())/max(1,int((binary | (old>.5)).sum())) >= duplicate_iou for old in masks):
                continue
            masks.append(mask);names.append(item['name']);kinds.append(item['kind'])
        project = new_project(image,torch.stack(masks),names)
        for layer,kind in zip(project['state']['layers'],kinds):
            layer['kind']=kind
        return (project,)


class LayersFromMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'masks': ('MASK',),
                'names': ('STRING', {'default': '', 'multiline': True})}}
    RETURN_TYPES = ('LAYERS_PROJECT',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, image, masks, names):
        return (new_project(image, masks, names.splitlines()),)


class LayersEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',),
                'editor_state': ('STRING', {'default': '', 'multiline': True})},
                'optional': {'sam_model': ('MODEL',)}}
    RETURN_TYPES = ('LAYERS_PROJECT',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, project, editor_state, sam_model=None):
        from comfy_execution.graph import ExecutionBlocker
        state = parse_state(editor_state, project['source'], [x['id'] for x in project['state']['layers']])
        if state is None:
            fresh = json.loads(json.dumps(project['state']))
            previous = json.loads(editor_state or '{}')
            # Regenerated content from the same original scene keeps arrangement,
            # but never reuses obsolete masks or generated pixels.
            if previous.get('origin') == fresh.get('origin') and previous.get('origin'):
                old = {x['id']: x for x in previous.get('layers', [])}
                if set(old) == {x['id'] for x in fresh['layers']}:
                    fresh['groups'] = previous.get('groups', {})
                    by_id = {x['id']: x for x in fresh['layers']}
                    fresh['layers'] = [by_id[x['id']] for x in previous['layers']]
                    for layer in fresh['layers']:
                        for key in ('x', 'y', 'scale', 'angle', 'group', 'visible'):
                            if key in old[layer['id']]: layer[key] = old[layer['id']][key]
                    parse_state(json.dumps(fresh), fresh['source'], list(old))
            return {'ui': ui_payload(project, fresh, review_required=True), 'result': (ExecutionBlocker(None),)}
        if (state.get('width'), state.get('height')) != (project['state']['width'], project['state']['height']):
            raise ValueError('Editor dimensions changed; reset editor.')
        request = state.pop('refine', None)
        region = state.pop('refine_region', None)
        if request:
            if sam_model is None:
                raise ValueError('Connect sam_model to use positive/negative click refinement.')
            layer = next(x for x in state['layers'] if x['id'] == request)
            index = next(i for i, x in enumerate(project['state']['layers']) if x['id'] == request)
            refine_image = project['rgbs'][index:index+1] if project.get('rgbs') is not None else project['image']
            old = read_mask(layer['mask'], (state['width'], state['height']))
            x, y, right, bottom = 0, 0, state['width'], state['height']
            if region is not None:
                x, y, right, bottom = (int(region[k]) for k in ('x', 'y', 'right', 'bottom'))
                if not (0 <= x < right <= state['width'] and 0 <= y < bottom <= state['height']):
                    raise ValueError('Invalid refinement crop. Zoom onto the image and retry.')
            def local_points(points):
                return [{'x': p['x']-x, 'y': p['y']-y} for p in points or []
                        if x <= p['x'] < right and y <= p['y'] < bottom]
            positive, negative = local_points(layer.get('positive')), local_points(layer.get('negative'))
            if region is not None and not positive:
                raise ValueError('Place a positive point on the object inside the visible refinement area.')
            candidates = detect(sam_model, refine_image[:, y:bottom, x:right], positive=positive, negative=negative)
            old_crop = old[y:bottom, x:right]
            if len(candidates) == 0 or not candidates.max().item():
                raise ValueError('SAM3 refinement found no object. Add a positive point inside it.')
            # Prefer the candidate overlapping this layer, not an unrelated detection.
            intersection = (candidates * old_crop).sum(dim=(1, 2))
            union = (candidates + old_crop - candidates * old_crop).sum(dim=(1, 2)).clamp_min(1)
            chosen = candidates[(intersection/union).argmax()]
            refined = old.clone()
            if region is not None:
                # Blend only at crop borders to avoid introducing a rectangular seam.
                ch, cw = chosen.shape
                yy, xx = torch.arange(ch), torch.arange(cw)
                weight = torch.minimum(torch.minimum(yy, ch-1-yy)[:,None],
                                       torch.minimum(xx, cw-1-xx)[None,:]).float().div(4).clamp(0,1)
                chosen = chosen*weight + old_crop*(1-weight)
            refined[y:bottom, x:right] = chosen
            layer['mask'] = data_url(pil(refined, 'L'))
            return {'ui': ui_payload(project, state, review_required=True), 'result': (ExecutionBlocker(None),)}
        return {'ui': ui_payload(project, state), 'result': (edited_project(project, state),)}


def validate_flux_fill(model):
    config = getattr(getattr(getattr(model, 'model', None), 'model_config', None), 'unet_config', {})
    if config.get('image_model') != 'flux' or config.get('in_channels') != 96:
        raise ValueError('Reconstruction requires FLUX.1 Fill dev. Connect UNETLoader flux1-fill-dev.safetensors, DualCLIPLoader (type flux, CLIP-L + T5XXL), and VAELoader ae.safetensors. SD/SDXL and ordinary FLUX dev are not supported.')


def inpaint(image, mask, model, clip, vae, prompt, negative, seed, steps, guidance):
    import nodes
    validate_flux_fill(model)
    # FLUX has 2x2 latent patches, so align pixels to 16 without cropping the source.
    h, w = image.shape[1:3]
    multiple = 16
    ph, pw = (-h) % multiple, (-w) % multiple
    pixels = torch.nn.functional.pad(image.movedim(-1, 1), (0, pw, 0, ph), mode='replicate').movedim(1, -1)
    padded_mask = torch.nn.functional.pad(mask, (0, pw, 0, ph))
    # FLUX Fill uses embedded guidance at CFG 1. Keep the instruction positive;
    # listing excluded scene objects can cause them to be generated again.
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt), add_dict={'guidance': guidance})
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(''))
    positive, neg, latent = nodes.InpaintModelConditioning().encode(
        positive=positive, negative=neg, pixels=pixels, vae=vae, mask=padded_mask, noise_mask=False)
    sampled = nodes.KSampler().sample(model, seed, steps, 1.0, 'euler', 'normal', positive, neg, latent, denoise=1.0)[0]
    generated = nodes.VAEDecode().decode(vae, sampled)[0][:, :h, :w, :3].cpu()
    alpha = mask.cpu().unsqueeze(-1)
    return image.cpu() * (1-alpha) + generated * alpha


class LayersReconstruct:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',), 'model': ('MODEL',), 'clip': ('CLIP',), 'vae': ('VAE',),
            'sam_model': ('MODEL',), 'sam_clip': ('CLIP',),
            'background_prompt': ('STRING', {'default': 'Continuous background matching the visible scene', 'multiline': True}),
            'negative_prompt': ('STRING', {'default': '', 'multiline': True, 'tooltip': 'Legacy field, unused by FLUX at CFG 1. Describe the desired content positively in layer names and style_prompt.'}),
            'complete_hidden': ('BOOLEAN', {'default': True}),
            'expand_pixels': ('INT', {'default': 32, 'min': 0, 'max': 256}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'steps': ('INT', {'default': 25, 'min': 1, 'max': 100}),
            'cfg': ('FLOAT', {'default': 1, 'min': 1, 'max': 1, 'tooltip': 'Legacy saved-workflow field. FLUX sampler CFG is always 1; use flux_guidance.'})},
            'optional': {
                'removal_margin': ('INT', {'default': 12, 'min': 0, 'max': 128,
                    'tooltip': 'Extra background pixels to regenerate around cutouts; reduces residual outlines.'}),
                'removal_feather': ('INT', {'default': 4, 'min': 0, 'max': 32,
                    'tooltip': 'Soft transition outside the fully removed area. Original cutout masks stay unchanged.'}),
                'reconstruction_mode': (['foreground removal', 'independent scene layers'],
                    {'default': 'foreground removal', 'tooltip': 'Completes the backmost background surface as an editable base, then occluded layers. Requires complete_hidden. No flattened duplicate backdrop.'}),
                'flux_guidance': ('FLOAT', {'default': 30, 'min': 0, 'max': 100, 'step': 0.5}),
                'style_prompt': ('STRING', {'default': 'Match the visible source image style, colors, lighting and texture. Preserve its level of detail.', 'multiline': True}),
                'completion_context': ('INT', {'default': 64, 'min': 0, 'max': 512, 'tooltip': 'Context around each local completion crop in source pixels.'}),
                'completion_resolution': ('INT', {'default': 768, 'min': 256, 'max': 1536, 'step': 16, 'tooltip': 'Longest side for local FLUX completion; results return to source coordinates.'})}}
    RETURN_TYPES = ('LAYERS_PROJECT', 'IMAGE')
    RETURN_NAMES = ('project', 'background')
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, project, model, clip, vae, sam_model, sam_clip, background_prompt, negative_prompt,
            complete_hidden, expand_pixels, seed, steps, cfg, removal_margin=12, removal_feather=4,
            reconstruction_mode='foreground removal', flux_guidance=30,
            style_prompt='Match the visible source image style, colors, lighting and texture. Preserve its level of detail.',
            completion_context=64, completion_resolution=768):
        import comfy.model_management
        image, masks = project['image'], project['masks'].clone()
        state = json.loads(json.dumps(project['state']))
        h, w = image.shape[1:3]
        if reconstruction_mode not in ('foreground removal', 'independent scene layers'):
            raise ValueError('Unknown reconstruction mode.')
        scene = reconstruction_mode == 'independent scene layers'
        if scene and (not complete_hidden or state['layers'][0].get('kind') != 'background'):
            raise ValueError('Independent scene layers requires complete_hidden enabled and a background surface (such as sky) as the backmost layer. Review layer order first.')
        # Keep original occluders immutable while completing layers back to front.
        source_masks = masks.clone()
        # Legacy mode removes foreground subjects only.
        removable = [i for i, layer in enumerate(state['layers']) if layer.get('kind') != 'background']
        union = masks[removable].amax(dim=0) if removable else torch.zeros_like(masks[0])
        if scene:
            # Fill every gap in the base surface, including all other scene elements.
            # The completed surface is itself editable, not duplicated below the stack.
            union = (1-source_masks[0]).clamp(0, 1)
            base_name = state['layers'][0]['name']
            background_prompt = f'Continuous {base_name} filling the image'
            base_negative = ''
        else:
            base_negative = negative_prompt
        removal = tensor(background_removal_mask(pil(union, 'L'), removal_margin, removal_feather)).unsqueeze(0)
        # Native model management loads/evicts models as each stage needs them.
        background = (inpaint(image, removal, model, clip, vae, f'{background_prompt}. Style: {style_prompt}', base_negative, seed, steps, flux_guidance)
                      if removal.max().item() > 0 else image.clone())
        rgbs = image.repeat(len(masks), 1, 1, 1)
        if scene:
            rgbs[0] = background[0]
            masks[0] = torch.ones_like(masks[0])
            state['layers'][0]['mask'] = data_url(pil(masks[0], 'L'))
        # State order is back-to-front. Restrict completion to foreground occluders
        # within its silhouette envelope; user can override with a painted region.
        if complete_hidden:
            for i, layer in enumerate(state['layers']):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if scene and i == 0:
                    continue
                original = source_masks[i].clone()
                front = source_masks[i+1:].amax(0) if i+1<len(masks) else torch.zeros_like(original)
                region = layer.get('completion')
                if region:
                    hole = read_mask(region, (w, h)) * (original <= 0)
                else:
                    hole = tensor(completion_region(pil(original,'L'),pil(front,'L'),expand_pixels))
                if hole.max().item() <= 0:
                    continue
                x0,y0,x1,y1 = completion_crop(pil(original,'L'),pil(hole,'L'),completion_context)
                visible_crop=original[y0:y1,x0:x1]
                hole_crop=hole[y0:y1,x0:x1]
                # Erase occluder context as well as the requested hole, so a partial
                # person does not remain in the crop conditioning. Only hole pixels
                # are allowed into the final layer; visible target pixels are protected.
                context_mask=torch.maximum(front,hole)
                fill=tensor(background_removal_mask(pil(context_mask,'L'),removal_margin,removal_feather))
                fill=fill[y0:y1,x0:x1]*(visible_crop<=0)
                crop=image[:,y0:y1,x0:x1]
                ch,cw=crop.shape[1:3]
                scale=completion_resolution/max(ch,cw)
                target=(max(16,round(ch*scale/16)*16),max(16,round(cw*scale/16)*16))
                pixels=torch.nn.functional.interpolate(crop.movedim(-1,1),size=target,mode='bilinear',align_corners=False).movedim(1,-1)
                fill_scaled=torch.nn.functional.interpolate(fill[None,None],size=target,mode='nearest')[0]
                description=re.sub(r' #\d+$','',layer['name'])
                description=re.sub(r'\s+on the (left|right)\b','',description,flags=re.I)
                if layer.get('discovery')=='automatic' and re.fullmatch(r'Region \d+',description):
                    description='visible object'
                surface=layer.get('kind')=='background' and bool(re.search(
                    r'\b(sky|grass|hill|hills|ground|floor|wall|ceiling|water)\b',description,re.I))
                prompt=(f'Continuous {description}. ' if surface else f'A complete {description}. ')+style_prompt
                generated=inpaint(pixels,fill_scaled,model,clip,vae,prompt,'',
                                  (seed+i+1) % (2**64),steps,flux_guidance)
                generated=torch.nn.functional.interpolate(generated.movedim(-1,1),size=(ch,cw),mode='bilinear',align_corners=False).movedim(1,-1)
                # Restore original context exactly after resizing for reliable SAM matching.
                completed=crop*(1-fill[None,...,None])+generated*fill[None,...,None]
                if surface:
                    # A continuous surface fills its bounded occlusion region. SAM
                    # must not punch the old subject silhouette back out of this fill.
                    alpha=torch.ones_like(hole_crop)
                else:
                    if layer.get('discovery')=='automatic':
                        interior=torch.nn.functional.avg_pool2d(visible_crop[None,None],7,stride=1,padding=3)[0,0]*visible_crop
                        index=int(interior.argmax())
                        candidates=detect(sam_model,completed,positive=[{'x':index % cw,'y':index // cw}])
                    else:
                        candidates=detect(sam_model,completed,sam_clip,description)
                    if not len(candidates) or not candidates.max().item():
                        raise ValueError(f'Could not segment reconstructed {layer["name"]}; adjust its completion region or description.')
                    score=(candidates*visible_crop).sum((1,2))/(candidates+visible_crop-candidates*visible_crop).sum((1,2)).clamp_min(1)
                    alpha=candidates[score.argmax()]
                    if not ((alpha>.5)&(hole_crop>.5)).any():
                        raise ValueError(f'Completion added no pixels to {layer["name"]}. Review its hidden-area mask or description; the layer is not complete.')
                masks[i,y0:y1,x0:x1]=torch.maximum(visible_crop,alpha*hole_crop)
                rgbs[i,y0:y1,x0:x1]=crop[0]*(1-hole_crop[...,None])+completed[0]*hole_crop[...,None]
                layer['mask'] = data_url(pil(masks[i], 'L'))
        result = dict(project, background=None if scene else background, masks=masks, rgbs=rgbs, state=state)
        # A new editor source signature invalidates only stale downstream edit sessions.
        result['source'] = hash_project(rgbs, masks, [x['name'] for x in state['layers']]) + hash_project(background, masks, [])
        state['source'] = result['source']
        return (result, background)


def render(project):
    state = project['state']
    w, h = state['width'], state['height']
    background = project.get('background')
    canvas = pil(background[0], 'RGBA') if background is not None else Image.new('RGBA', (w,h))
    rgbs = project.get('rgbs', project['image'].repeat(len(project['masks']),1,1,1))
    rendered = []
    for layer, rgb, mask in zip(state['layers'], rgbs, project['masks']):
        obj = pil(rgb, 'RGBA')
        obj.putalpha(pil(mask, 'L'))
        obj = obj.transform((w,h), Image.Transform.AFFINE,
                inverse_pillow(layer_matrix(layer, state.get('groups',{}), w,h)), Image.Resampling.BICUBIC)
        rendered.append(tensor(obj))
        if visible(layer, state.get('groups',{})):
            canvas = Image.alpha_composite(canvas, obj)
    return tensor(canvas).unsqueeze(0), torch.stack(rendered)


class LayersMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',),
            'edge_radius': ('INT', {'default': 1, 'min': 1, 'max': 16,
                'tooltip': 'Unknown boundary width in source pixels. Keep small for fingers/hair.'}),
            'matte_strength': ('FLOAT', {'default': .25, 'min': 0., 'max': 1., 'step': .05}),
            'cleanup_strength': ('FLOAT', {'default': .35, 'min': 0., 'max': 1., 'step': .05}),
            'layer_index': ('INT', {'default': -1, 'min': -1, 'max': MAX_LAYERS-1,
                'tooltip': '-1 refines all layers; otherwise use a zero-based layer index.'})}}
    RETURN_TYPES = ('LAYERS_PROJECT', 'MASK')
    RETURN_NAMES = ('project', 'refined_alpha')
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, project, edge_radius, matte_strength, cleanup_strength, layer_index):
        from .matting import refine_layer
        if layer_index < -1 or layer_index >= len(project['masks']):
            raise ValueError('Matting layer index is outside this project.')
        if matte_strength == 0 and cleanup_strength == 0:
            return project, project['masks']
        state = json.loads(json.dumps(project['state']))
        masks = project['masks'].clone()
        rgbs = project.get('rgbs', project['image'].repeat(len(masks),1,1,1)).clone()
        for i, layer in enumerate(state['layers']):
            if layer_index != -1 and i != layer_index:
                continue
            # Cooperate with ComfyUI's cancel button between layers.
            import comfy.model_management
            comfy.model_management.throw_exception_if_processing_interrupted()
            rgbs[i], masks[i] = refine_layer(rgbs[i], masks[i], edge_radius, matte_strength, cleanup_strength)
            layer['mask'] = data_url(pil(masks[i], 'L'))
        result = dict(project, state=state, masks=masks, rgbs=rgbs)
        result['source'] = hashlib.sha256((project['source']+hash_project(rgbs,masks,[])).encode()).hexdigest()
        state['source'] = result['source']
        return result, masks


class LayersComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',)}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK')
    RETURN_NAMES = ('composite_rgba', 'layers_rgba', 'layer_alpha')
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, project):
        composite, layers = render(project)
        return composite, layers, layers[...,3]


class LayersGetLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',), 'index': ('INT', {'default': 0, 'min': 0, 'max': MAX_LAYERS}),
                            'transformed': ('BOOLEAN', {'default': False})}}
    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    RETURN_NAMES = ('rgba', 'alpha', 'name')
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, project, index, transformed):
        if index >= len(project['masks']):
            raise ValueError(f'Layer {index} does not exist; indexes start at zero.')
        if transformed:
            rgba = render(project)[1][index:index+1]
        else:
            rgb = project.get('rgbs', project['image'].repeat(len(project['masks']),1,1,1))[index:index+1]
            rgba = torch.cat([rgb, project['masks'][index:index+1,...,None]], dim=-1)
        return rgba, rgba[...,3], project['state']['layers'][index]['name']


class LayersSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',), 'prefix': ('STRING', {'default': 'layers'})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('project_directory',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, project, prefix):
        import folder_paths
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', prefix)[:80] or 'layers'
        root = Path(folder_paths.get_output_directory()) / f'{safe}_{uuid.uuid4().hex[:12]}'
        root.mkdir(parents=True, exist_ok=False)
        composite, layers = render(project)
        pil(composite[0]).save(root/'composite.png')
        pil(project['image'][0], 'RGB').save(root/'source.png')
        rgbs = project.get('rgbs', project['image'].repeat(len(layers),1,1,1))
        for i, (layer, rgb, mask, placed) in enumerate(zip(project['state']['layers'],rgbs,project['masks'],layers)):
            name = re.sub(r'[^A-Za-z0-9_-]', '_', layer['name'])[:60]
            pil(placed).save(root/f'{i:02d}_{name}_placed.png')
            obj = pil(rgb,'RGBA'); obj.putalpha(pil(mask,'L'))
            obj.save(root/f'layer_{i:02d}.png')
            # Preserve RGB beyond alpha for non-destructive brush restoration.
            pil(rgb,'RGB').save(root/f'rgb_{i:02d}.png')
        if project.get('background') is not None:
            pil(project['background'][0],'RGB').save(root/'background.png')
        (root/'project.json').write_text(json.dumps(project['state'], indent=2))
        info = {'filename':'composite.png','subfolder':root.name,'type':'output'}
        return {'ui': {'images':[info]}, 'result': (str(root),)}


class LayersLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project_directory': ('STRING', {'default': 'layers_saved_directory'})}}
    RETURN_TYPES = ('LAYERS_PROJECT',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, project_directory):
        import folder_paths
        base = Path(folder_paths.get_output_directory()).resolve()
        root = (base/project_directory).resolve()
        if not root.is_relative_to(base):
            raise ValueError('Project must be inside ComfyUI output directory.')
        state = json.loads((root/'project.json').read_text())
        with Image.open(root/'source.png') as im:
            image = tensor(im.convert('RGB')).unsqueeze(0)
        masks = torch.stack([read_mask(x['mask'], (image.shape[2],image.shape[1])) for x in state['layers']])
        project = new_project(image, masks, [x['name'] for x in state['layers']])
        state['source'] = project['source']
        parse_state(json.dumps(state), project['source'], [x['id'] for x in project['state']['layers']])
        rgbs = []
        for i in range(len(masks)):
            with Image.open(root/f'rgb_{i:02d}.png') as im:
                if im.size != (image.shape[2],image.shape[1]):
                    raise ValueError('Saved layer size mismatch')
                rgbs.append(tensor(im.convert('RGB')))
        project.update(state=state, rgbs=torch.stack(rgbs))
        if (root/'background.png').exists():
            with Image.open(root/'background.png') as im:
                if im.size != (image.shape[2],image.shape[1]):
                    raise ValueError('Saved background size mismatch')
                project['background'] = tensor(im.convert('RGB')).unsqueeze(0)
        return (project,)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in [LayersDiscoverObjects, LayersReviewObjects, LayersSegmentObjects, LayersSAM3, LayersSAM3Auto, LayersFromMasks, LayersEditor,
    LayersReconstruct, LayersMatte, LayersComposite, LayersGetLayer, LayersSave, LayersLoad]}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LayersDiscoverObjects': 'Layers • Discover Objects (Qwen VL)',
    'LayersReviewObjects': 'Layers • Review Objects',
    'LayersSegmentObjects': 'Layers • Segment Reviewed Objects (SAM3)',
    'LayersSAM3Auto': 'Layers • SAM3 Automatic Regions',
    'LayersSAM3': 'Layers • SAM3 Named Objects', 'LayersFromMasks': 'Layers • Import Masks',
    'LayersEditor': 'Layers • Compositor & Mask Editor', 'LayersReconstruct': 'Layers • Reconstruct',
    'LayersMatte': 'Layers • Alpha Matte & Edge Cleanup',
    'LayersComposite': 'Layers • Render', 'LayersGetLayer': 'Layers • Get Layer (V4 Bridge)',
    'LayersSave': 'Layers • Save Project', 'LayersLoad': 'Layers • Load Project'}
