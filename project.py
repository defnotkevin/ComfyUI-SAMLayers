"""Portable project validation and affine transforms (no ComfyUI dependencies)."""
import json
import math

VERSION = 1
MAX_LAYERS = 64


def parse_state(raw, source, layer_ids):
    state = json.loads(raw or '{}')
    if not isinstance(state, dict):
        raise ValueError('Editor state must be an object')
    if not state or state.get('source') != source:
        return None
    if state.get('version') != VERSION:
        raise ValueError('Unsupported layer project version')
    layers = state.get('layers', [])
    if len(layers) != len(layer_ids) or {x['id'] for x in layers} != set(layer_ids):
        raise ValueError('Layer IDs do not match this source. Reset the editor.')
    groups = state.get('groups', {})
    if not isinstance(groups, dict):
        raise ValueError('Invalid groups')
    for item in [*layers, *groups.values()]:
        validate_transform(item)
    for layer in layers:
        if layer.get('group') and layer['group'] not in groups:
            raise ValueError('Missing group')
    return state


def validate_transform(item):
    for key, default, low, high in [('x', 0, -100000, 100000), ('y', 0, -100000, 100000),
                                    ('scale', 1, .01, 100), ('angle', 0, -36000, 36000)]:
        value = item.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid {key}')


def matrix(item, width, height):
    angle = math.radians(item.get('angle', 0))
    scale = item.get('scale', 1)
    a, b = scale * math.cos(angle), scale * math.sin(angle)
    cx, cy = width / 2, height / 2
    return (a, b, -b, a, cx + item.get('x', 0) - a * cx + b * cy,
            cy + item.get('y', 0) - b * cx - a * cy)


def multiply(a, b):
    return (a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
            a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5])


def inverse_pillow(m):
    a, b, c, d, e, f = m
    det = a*d-b*c
    if abs(det) < 1e-12:
        raise ValueError('Singular layer transform')
    return (d/det, -c/det, (c*f-d*e)/det, -b/det, a/det, (b*e-a*f)/det)


def layer_matrix(layer, groups, width, height):
    own = matrix(layer, width, height)
    group = groups.get(layer.get('group'))
    return multiply(matrix(group, width, height), own) if group else own


def visible(layer, groups):
    return layer.get('visible', True) and groups.get(layer.get('group'), {}).get('visible', True)
