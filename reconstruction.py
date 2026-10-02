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
