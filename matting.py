"""Local color matting and straight-alpha foreground decontamination.

No model downloads. A trimap constrains the solve; nearby known foreground and
background colors estimate C = alpha F + (1-alpha) B in the boundary band.
This local approximation is not a neural hair/transparent-object matting model.
"""
import torch
import torch.nn.functional as F


def spread_colors(rgb, known, steps):
    colors = rgb * known
    valid = known.clone()
    for _ in range(steps):
        count = F.avg_pool2d(valid, 3, 1, 1)
        estimate = F.avg_pool2d(colors * valid, 3, 1, 1) / count.clamp_min(1e-8)
        fill = (valid == 0) & (count > 0)
        colors = torch.where(fill, estimate, colors)
        valid = torch.where(fill, torch.ones_like(valid), valid)
    return colors, valid


def refine_layer(rgb, alpha, radius=3, matte_strength=1., cleanup_strength=.75):
    if not 1 <= radius <= 16 or not 0 <= matte_strength <= 1 or not 0 <= cleanup_strength <= 1:
        raise ValueError('Invalid matting radius or strength')
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or alpha.shape != rgb.shape[:2]:
        raise ValueError('Matting expects HWC RGB and matching HW alpha')
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise ValueError('Matting inputs must be finite')
    rgb = rgb.detach().cpu().float().clamp(0, 1)
    alpha = alpha.detach().cpu().float().clamp(0, 1)
    if matte_strength == 0 and cleanup_strength == 0:
        return rgb.clone(), alpha.clone()
    a = alpha[None,None]
    image = rgb.permute(2,0,1)[None]
    kernel = radius*2+1
    # Replicate padding avoids falsely treating image borders as background.
    pad = F.pad(a, (radius,)*4, mode='replicate')
    inner = -F.max_pool2d(-pad, kernel, 1)
    outer = F.max_pool2d(pad, kernel, 1)
    fg = (inner >= .98).float()
    bg = (outer <= .02).float()
    # Thin components without a known interior are left unchanged, not erased.
    if not fg.any() or not bg.any():
        return rgb.clone(), alpha.clone()
    fore, vf = spread_colors(image, fg, radius*3+3)
    back, vb = spread_colors(image, bg, radius*3+3)
    unknown = (fg == 0) & (bg == 0)
    delta = fore-back
    separation = (delta*delta).sum(1,keepdim=True)
    usable = unknown & (vf > 0) & (vb > 0) & (separation > .01)
    # Regularize ambiguous colors toward the original mask rather than inventing detail.
    estimate = (((image-back)*delta).sum(1,keepdim=True)+.001*a) / (separation+.001)
    estimate = estimate.clamp(0,1)
    refined = torch.where(usable, a*(1-matte_strength)+estimate*matte_strength, a)
    # Only change partially transparent boundary pixels. Preserve opaque interiors.
    edge = usable & (refined > .02) & (refined < .98)
    recovered = ((image-(1-refined)*back)/refined.clamp_min(.05)).clamp(0,1)
    # Very low-alpha pixels are ill-conditioned; use the nearby foreground estimate.
    stable = (refined/.15).clamp(0,1)
    recovered = stable*recovered+(1-stable)*fore
    clean = torch.where(edge, image*(1-cleanup_strength)+recovered*cleanup_strength, image)
    return clean[0].permute(1,2,0).contiguous(), refined[0,0].contiguous()
