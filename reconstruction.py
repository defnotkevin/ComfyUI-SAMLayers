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
