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


def detect(model, image, clip=None, prompt='', positive=None, negative=None, threshold=.5):
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
    result = SAM3_Detect.execute(model=model, image=image, conditioning=conditioning,
        positive_coords=json.dumps(positive) if positive else None,
        negative_coords=json.dumps(negative) if negative else None,
        threshold=threshold, refine_iterations=2, individual_masks=True)
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


def ui_payload(project, state):
    payload = dict(state=state, image=data_url(pil(project['image'][0], 'RGB')),
                   reconstructed=bool(project.get('background') is not None))
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
            raise ValueError('Enter one object description per line (maximum 64).')
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
            return {'ui': ui_payload(project, fresh), 'result': (ExecutionBlocker(None),)}
        if (state.get('width'), state.get('height')) != (project['state']['width'], project['state']['height']):
            raise ValueError('Editor dimensions changed; reset editor.')
        request = state.pop('refine', None)
        if request:
            if sam_model is None:
                raise ValueError('Connect sam_model to use positive/negative click refinement.')
            layer = next(x for x in state['layers'] if x['id'] == request)
            index = next(i for i, x in enumerate(project['state']['layers']) if x['id'] == request)
            refine_image = project['rgbs'][index:index+1] if project.get('rgbs') is not None else project['image']
            candidates = detect(sam_model, refine_image, positive=layer.get('positive'), negative=layer.get('negative'))
            old = read_mask(layer['mask'], (state['width'], state['height']))
            if len(candidates) == 0 or not candidates.max().item():
                raise ValueError('SAM3 refinement found no object. Add a positive point inside it.')
            # Prefer the candidate overlapping this layer, not an unrelated detection.
            intersection = (candidates * old).sum(dim=(1, 2))
            union = (candidates + old - candidates * old).sum(dim=(1, 2)).clamp_min(1)
            layer['mask'] = data_url(pil(candidates[(intersection/union).argmax()], 'L'))
            return {'ui': ui_payload(project, state), 'result': (ExecutionBlocker(None),)}
        return {'ui': ui_payload(project, state), 'result': (edited_project(project, state),)}


def inpaint(image, mask, model, clip, vae, prompt, negative, seed, steps, cfg):
    import nodes
    # Pad instead of cropping: preserve image/mask registration for arbitrary sizes.
    h, w = image.shape[1:3]
    multiple = int(getattr(vae, 'downscale_ratio', 8))
    ph, pw = (-h) % multiple, (-w) % multiple
    pixels = torch.nn.functional.pad(image.movedim(-1, 1), (0, pw, 0, ph), mode='replicate').movedim(1, -1)
    padded_mask = torch.nn.functional.pad(mask, (0, pw, 0, ph))
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
    # Native conditioning works for dedicated inpainting checkpoints and standard SD/SDXL.
    positive, neg, latent = nodes.InpaintModelConditioning().encode(positive=positive, negative=neg, pixels=pixels, vae=vae, mask=padded_mask, noise_mask=True)
    sampled = nodes.KSampler().sample(model, seed, steps, cfg, 'euler', 'normal', positive, neg, latent, denoise=1.0)[0]
    generated = nodes.VAEDecode().decode(vae, sampled)[0][:, :h, :w, :3].cpu()
    alpha = mask.cpu().unsqueeze(-1)
    return image.cpu() * (1-alpha) + generated * alpha


class LayersReconstruct:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'project': ('LAYERS_PROJECT',), 'model': ('MODEL',), 'clip': ('CLIP',), 'vae': ('VAE',),
            'sam_model': ('MODEL',), 'sam_clip': ('CLIP',),
            'background_prompt': ('STRING', {'default': 'empty room, continuous background, no people, no foreground objects', 'multiline': True}),
            'negative_prompt': ('STRING', {'default': 'artifacts, duplicated objects, text, watermark', 'multiline': True}),
            'complete_hidden': ('BOOLEAN', {'default': True}),
            'expand_pixels': ('INT', {'default': 32, 'min': 0, 'max': 256}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'steps': ('INT', {'default': 25, 'min': 1, 'max': 100}),
            'cfg': ('FLOAT', {'default': 6, 'min': 0, 'max': 30})}}
    RETURN_TYPES = ('LAYERS_PROJECT', 'IMAGE')
    RETURN_NAMES = ('project', 'background')
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    def run(self, project, model, clip, vae, sam_model, sam_clip, background_prompt, negative_prompt,
            complete_hidden, expand_pixels, seed, steps, cfg):
        import comfy.model_management
        image, masks = project['image'], project['masks'].clone()
        state = json.loads(json.dumps(project['state']))
        h, w = image.shape[1:3]
        union = masks.amax(dim=0)
        removal = tensor(pil(union, 'L').filter(ImageFilter.MaxFilter(7))).unsqueeze(0)
        # Native model management loads/evicts models as each stage needs them.
        background = inpaint(image, removal, model, clip, vae, background_prompt, negative_prompt, seed, steps, cfg)
        rgbs = image.repeat(len(masks), 1, 1, 1)
        # State order is back-to-front. Restrict completion to foreground occluders
        # near the visible object's bounding box; user can override with a painted region.
        if complete_hidden:
            for i, layer in enumerate(state['layers']):
                comfy.model_management.throw_exception_if_processing_interrupted()
                original = masks[i].clone()
                region = layer.get('completion')
                if region:
                    hole = read_mask(region, (w, h)) * (1-original)
                else:
                    bbox = pil(original, 'L').getbbox()
                    if bbox is None or i == len(masks)-1:
                        continue
                    x0, y0, x1, y1 = bbox
                    near = torch.zeros_like(original)
                    near[max(0,y0-expand_pixels):min(h,y1+expand_pixels), max(0,x0-expand_pixels):min(w,x1+expand_pixels)] = 1
                    hole = masks[i+1:].amax(0) * near * (1-original)
                if hole.max().item() <= 0:
                    continue
                description = re.sub(r" #\d+$", "", layer["name"])
                prompt = f"{description}, complete intact object, natural continuation of visible shape and texture"
                completed = inpaint(image, hole.unsqueeze(0), model, clip, vae, prompt, negative_prompt,
                                    (seed+i+1) % (2**64), steps, cfg)
                candidates = detect(sam_model, completed, sam_clip, re.sub(r' #\d+$', '', layer['name']))
                if not len(candidates) or not candidates.max().item():
                    raise ValueError(f"Could not segment reconstructed {layer['name']}; adjust its completion region/prompt.")
                score = (candidates*original).sum((1,2)) / (candidates+original-candidates*original).sum((1,2)).clamp_min(1)
                alpha = candidates[score.argmax()]
                masks[i] = torch.maximum(original, alpha*hole)
                rgbs[i] = completed[0]
                layer['mask'] = data_url(pil(masks[i], 'L'))
        result = dict(project, background=background, masks=masks, rgbs=rgbs, state=state)
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


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in [LayersSAM3, LayersFromMasks, LayersEditor,
    LayersReconstruct, LayersComposite, LayersGetLayer, LayersSave, LayersLoad]}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LayersSAM3': 'Layers • SAM3 Named Objects', 'LayersFromMasks': 'Layers • Import Masks',
    'LayersEditor': 'Layers • Compositor & Mask Editor', 'LayersReconstruct': 'Layers • Reconstruct',
    'LayersComposite': 'Layers • Render', 'LayersGetLayer': 'Layers • Get Layer (V4 Bridge)',
    'LayersSave': 'Layers • Save Project', 'LayersLoad': 'Layers • Load Project'}
