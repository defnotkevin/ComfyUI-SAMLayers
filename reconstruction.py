"""Background removal masks, separate from editable cutout alpha."""
from PIL import ImageChops, ImageFilter


def background_removal_mask(alpha, margin=12, feather=4):
    """Fully replace foreground, with a soft transition outside its solid margin.

    A cutout's fractional alpha must not blend the original object back into the
    reconstructed background. Preserve exact zero outside the bounded feather.
    """
    if not 0 <= margin <= 128 or not 0 <= feather <= 32:
        raise ValueError('Removal margin must be 0–128 and feather 0–32 pixels.')
    solid = alpha.convert('L').point(lambda value: 255 if value > 0 else 0)
    if margin:
        solid = solid.filter(ImageFilter.MaxFilter(2 * int(margin) + 1))
    if not feather:
        return solid
    soft = solid.filter(ImageFilter.GaussianBlur(float(feather)))
    # Blur only adds an outside transition; it never weakens the removal core.
    return ImageChops.lighter(solid, soft)


def completion_region(visible, occluders, expansion=32):
    """Bound automatic holes by the visible silhouette's convex envelope.

    The envelope bridges interrupted visible pieces; it is only a candidate region,
    not the final object alpha. Intersect it with actual front-layer occluders.
    """
    from PIL import Image, ImageDraw
    if visible.size != occluders.size or not 0 <= expansion <= 256:
        raise ValueError('Completion masks must match and expansion must be 0–256.')
    support = visible.convert('L').point(lambda x: 255 if x > 0 else 0)
    bounds = support.getbbox()
    if bounds is None:
        return Image.new('L', visible.size)
    # Row extrema suffice to construct the convex hull without listing every pixel.
    points = []
    for y in range(bounds[1], bounds[3]):
        row = support.crop((bounds[0], y, bounds[2], y+1)).getbbox()
        if row:
            points.extend(((bounds[0]+row[0], y), (bounds[0]+row[2]-1, y)))
    points = sorted(set(points))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def half(seq):
        out=[]
        for point in seq:
            while len(out)>1 and cross(out[-2],out[-1],point)<=0:
                out.pop()
            out.append(point)
        return out
    hull=half(points)[:-1]+half(reversed(points))[:-1]
    envelope=Image.new('L', visible.size)
    if len(hull)>=3:
        ImageDraw.Draw(envelope).polygon(hull,fill=255)
    else:
        envelope=support.copy()
    if expansion:
        envelope=envelope.filter(ImageFilter.MaxFilter(2*int(expansion)+1))
    front=occluders.convert('L').point(lambda x:255 if x>0 else 0)
    return ImageChops.subtract(ImageChops.darker(envelope,front),support)


def completion_crop(visible, hole, context=64, min_side=256):
    """Crop including the target and hole, with bounded neighboring context."""
    if visible.size != hole.size or not 0 <= context <= 512 or min_side < 1:
        raise ValueError('Invalid completion crop inputs.')
    box=ImageChops.lighter(visible.convert('L'),hole.convert('L')).getbbox()
    if box is None:
        raise ValueError('Cannot crop an empty completion target.')
    w,h=visible.size
    def axis(lo,hi,limit):
        size=min(limit,max(min_side,hi-lo+2*context))
        start=max(0,min(limit-size,(lo+hi-size)//2))
        return start,start+size
    x0,x1=axis(box[0],box[2],w);y0,y1=axis(box[1],box[3],h)
    return x0,y0,x1,y1


def completion_blend_mask(visible, hole, feather=4):
    """Replace the full hole and blend only into existing target support.

    The transition changes RGB, never expands the cutout silhouette. Setting
    feather to zero preserves all originally visible pixels exactly.
    """
    if visible.size != hole.size:
        raise ValueError('Completion masks must match.')
    core = background_removal_mask(hole, 0, 0)
    transition = background_removal_mask(core, 0, feather)
    support = visible.convert('L').point(lambda x: 255 if x > 0 else 0)
    return ImageChops.lighter(core, ImageChops.darker(transition, support))


def match_boundary_colors(source, generated, known, max_shift=.25):
    """Extend low-frequency RGB differences from reliable context into the fill.

    A coarse harmonic field adjusts color/lighting offsets, not texture or palette.
    No samples from the removed object are used. Returns a new HWC float array.
    """
    import numpy as np
    from PIL import Image
    src=np.asarray(source,dtype=np.float32)
    gen=np.asarray(generated,dtype=np.float32)
    weight=np.asarray(known,dtype=np.float32)
    if src.shape != gen.shape or src.ndim != 3 or src.shape[2] != 3 or weight.shape != src.shape[:2]:
        raise ValueError('Color matching expects matching HWC RGB and HW confidence.')
    if not (np.isfinite(src).all() and np.isfinite(gen).all() and np.isfinite(weight).all()):
        raise ValueError('Color matching inputs must be finite.')
    if weight.min()<0 or weight.max()>1 or not 0<=max_shift<=1:
        raise ValueError('Invalid color matching confidence or shift limit.')
    if weight.sum()<16 or max_shift==0:
        return gen.copy()
    h,w=weight.shape
    size=(max(1,round(w*min(1,96/max(h,w)))),max(1,round(h*min(1,96/max(h,w)))))
    def resize(a, size, mode):
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size,mode),dtype=np.float32)
    confidence=resize(weight,size,Image.Resampling.BOX)
    residual=src-gen
    anchors=np.stack([resize(residual[...,c]*weight,size,Image.Resampling.BOX) for c in range(3)],-1)
    anchors=anchors/np.maximum(confidence[...,None],1e-8)
    # Mixed boundary cells have valid weighted samples, not averaged object colors.
    fixed=confidence>=.05
    if not fixed.any():
        return gen.copy()
    field=np.broadcast_to((residual*weight[...,None]).sum((0,1))/weight.sum(),anchors.shape).copy()
    field[fixed]=anchors[fixed]
    for _ in range(600):
        pad=np.pad(field,((1,1),(1,1),(0,0)),mode='edge')
        relaxed=(pad[:-2,1:-1]+pad[2:,1:-1]+pad[1:-1,:-2]+pad[1:-1,2:])*.25
        relaxed[fixed]=anchors[fixed]
        change=np.max(np.abs(relaxed-field))
        field=relaxed
        if change<1e-5:
            break
    correction=np.stack([resize(field[...,c],(w,h),Image.Resampling.BILINEAR) for c in range(3)],-1)
    return np.clip(gen+np.clip(correction,-max_shift,max_shift),0,1)


def merge_completion_alpha(visible, predicted, hole, generated_support, manual=False, seam_radius=2):
    """Keep connected predicted completion; repair only narrow joins near the hole.

    Automatic holes initiate completion, not its final silhouette. Manual holes
    remain hard limits. Original visible alpha and unrelated background stay intact.
    """
    import numpy as np
    from PIL import Image
    if len({im.size for im in (visible,predicted,hole,generated_support)})!=1:
        raise ValueError('Completion alpha sizes must match.')
    if not 0<=seam_radius<=8:
        raise ValueError('Seam radius must be 0–8.')
    v=np.asarray(visible.convert('L')); p=np.asarray(predicted.convert('L'))
    target=np.asarray(hole.convert('L'))>0
    allowed=target if manual else np.asarray(generated_support.convert('L'))>0
    candidate=(p>127)&(allowed|(v>127))
    # Retain only components connected to the visible target. SAM may return
    # a mask containing multiple instances even after overlap-based selection.
    seeds=candidate&(v>127)
    connected=seeds.copy()
    stack=list(zip(*np.nonzero(seeds)))
    h,w=candidate.shape
    while stack:
        y,x=stack.pop()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and candidate[yy,xx] and not connected[yy,xx]:
                connected[yy,xx]=True;stack.append((yy,xx))
    added=np.where(allowed&connected,p,0).astype('uint8')
    merged=np.maximum(v,added)
    if not manual and seam_radius and target.any():
        # Closing fills one/two-pixel cracks, not broad missing regions. Restrict
        # changes to generated context close to the requested completion.
        mask=Image.fromarray(merged)
        closed=mask.filter(ImageFilter.MaxFilter(2*seam_radius+1)).filter(ImageFilter.MinFilter(2*seam_radius+1))
        vicinity=np.asarray(Image.fromarray(target.astype('uint8')*255).filter(ImageFilter.MaxFilter(2*seam_radius+1)))>0
        repair=vicinity&allowed
        merged=np.where(repair,np.maximum(merged,np.asarray(closed)),merged).astype('uint8')
    return Image.fromarray(merged)
