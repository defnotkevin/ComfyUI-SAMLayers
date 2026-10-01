# ComfyUI-Layers

An initial implementation of SAM3 object layers, a browser compositor with popup mask editing, and background/hidden-object reconstruction. Built for a ComfyUI server such as a RunPod 3090 instance; the editor runs in your browser.

**Status:** named SAM3 detection and the original editor have been confirmed working by the user on RunPod. Local CPU processing and JavaScript tests pass. The new zoom/thumbnail controls, automatic discovery, and diffusion reconstruction still need live RunPod validation. This remains experimental.

## Install on RunPod

1. Copy this entire `ComfyUI-Layers` directory into your RunPod installation's `ComfyUI/custom_nodes/` directory.
2. Use a ComfyUI version with native `SAM3_Detect` and the SAM3.1 checkpoint loader support. Your supplied `Sam3WF.json` should run first.
3. Put `sam3.1_multiplex_fp16.safetensors` in `ComfyUI/models/checkpoints/`.
4. For reconstruction, select a compatible SD/SDXL checkpoint with its matching CLIP and VAE. A dedicated inpainting checkpoint is preferable. **The Qwen layered checkpoint is not a drop-in inpainting model for this node.** No additional models are downloaded automatically.
5. Restart ComfyUI and refresh the browser. Search the node menu for `Layers`.

No additional Python dependencies are needed beyond ComfyUI's PyTorch, NumPy and Pillow. Do not replace the RunPod PyTorch install.

## Automatic discovery (no object names)

Open `examples/sam3_layers_auto_edit.json` to discover candidate regions, or
`examples/sam3_layers_auto_reconstruct.json` for the reconstruction pipeline.
These use **Layers • SAM3 Automatic Regions** with the same SAM3.1 checkpoint.
Existing named-object workflows remain available for targeted selection.

Automatic discovery samples individual point prompts across a grid, filters tiny
and almost-full-image masks, and removes near-duplicates by intersection-over-union.
The default 8 x 8 grid runs up to 64 separate SAM3 passes, so it can be substantially
slower than named detection. `points_per_side` controls density, `min_area` and
`max_area` are fractions of image area, `duplicate_iou` controls duplicate removal,
and `max_layers` limits output (maximum 64).

This is a point-grid heuristic, not a guarantee of finding every semantic object.
It can miss small objects, return parts or overlapping regions, and leave some
pixels uncovered. Results have generic region names; rename them and review depth
order before reconstruction. Select the regions you want to manipulate in the
editor. For automatic regions, reconstruction uses a visible point to resegment
the generated object instead of treating a generic region number as a text prompt.

## Detailed editing and node preview

After detection, the editor node displays the source-image thumbnail, detected
layer names, and whether it is paused for review. The internal JSON remains saved
in the workflow but is hidden from the node. Click **Open layer editor**, then
**Apply & Run** to continue to render/save; blank downstream previews before applying
are expected.

Both canvases support **10–3200% zoom**, a slider, an exact percentage field, plus/minus,
100%, and Fit controls. **Mouse wheel** zooms around the pointer (Ctrl/Cmd also works); **Shift + wheel** scrolls;
**middle-button drag** pans without moving a layer or painting. The scrollbars also pan. Brushes still operate in source-image pixels, so a 1-pixel brush is
usable at high zoom.

Select one visible object to show a floating **Edit mask** button over its bounding
area. The button follows movement, transforms, scrolling and zoom. The right-hand
Mask buttons and viewport double-click remain available. The former top-toolbar
Edit mask button has been removed.

## Clean up fine edges and finger gaps

SAM3's masks can include background between fingers and have binary, jagged edges.
Zooming enlarges the existing pixels; it does not recover detail missing from the source.

1. Open the object's mask popup and zoom onto the hand or another troublesome area.
2. Place positive clicks on the object **inside the visible crop**, and negative clicks in unwanted background gaps.
3. Click **Refine visible area**. This queues SAM3 on just that crop, using the original source pixels and local point coordinates. Only the cropped mask is changed; its border blends back into the previous mask. Reopen the editor to review.
4. Erase any remaining background using a small brush. Use **Shrink mask 1 px** sparingly for a thin background fringe; it affects the entire active mask and can remove thin details.
5. Use **Soften edge 0.7 px** for modest antialiasing after fixing the shape. This affects the entire active mask. Both cleanup actions support Undo and Cancel.

Point markers are a fixed 12 CSS pixels and do not grow with image zoom. Brush size
remains measured in source pixels. Point refinement uses one prompted pass so a
subsequent unprompted decoder pass does not change the click-guided result.

Feathering cannot remove large incorrectly selected gaps, restore hair detail, or
remove background colors mixed into edge pixels. The Alpha Matte & Edge Cleanup node below handles a narrow boundary band, but
ambiguous hair, glass and large segmentation errors still need a stronger matting
model or manual correction.

### Empty prompt versus automatic detection

`sam3_layers_edit.json` still uses **SAM3 Named Objects**. Leaving its objects field
blank is an error. Load `sam3_layers_auto_edit.json` for automatic discovery, or
replace Named Objects with **Layers • SAM3 Automatic Regions**, connect image and
SAM model, and connect its project output to the editor. The automatic node has
no object-name input. It uses the same checkpoint and needs no extra models.

## Alpha matting and edge-color cleanup

**Layers • Alpha Matte & Edge Cleanup** refines transparency and reduces background
color contamination at cutout boundaries. No extra checkpoint or Python package is
required. It runs on CPU using PyTorch, one layer at a time.

The regenerated examples include it automatically:

- Editing workflows: **mask editor → matting → render/save**. Inspect the final output preview to see the refined edges; the upstream editor still shows your original editable mask.
- Reconstruction/V4 workflows: **reconstruction → matting → arrangement editor → render/save or V4 bridge**. The arrangement editor receives the refined alpha and cleaned RGB.

For an existing workflow, insert the node on the `LAYERS_PROJECT` connection before
rendering, saving, or the final arrangement editor. Keep it after mask editing; changes
made upstream recompute from the upstream image rather than repeatedly cleaning the
last result. To inspect masks, connect `refined_alpha` to a standard mask preview.

| Control | Starting value | Effect |
|---|---|---|
| edge_radius | 3 | Width of the uncertain boundary in source pixels (1–16). Start at 1–2 for fine fingers or hair. |
| matte_strength | 1.0 | Blend between the original mask and estimated partial transparency. Zero keeps the original alpha. |
| cleanup_strength | 0.75 | Strength of foreground color recovery on partially transparent edges. Zero keeps original RGB. |
| layer_index | -1 | Process every layer. Otherwise choose a zero-based layer index. |

Set both strengths to zero for an exact bypass. Transforms, groups, layer names,
source pixels, and the reconstructed background are preserved. The output project
contains separate refined masks and RGB layers; existing save/load and RGBA exports
include them. No UI controls or per-mask popup settings are required for this node.

The algorithm builds a trimap from the mask: known foreground, known background,
and an uncertain boundary. Local foreground/background colors are propagated into
that band. It estimates alpha using the compositing equation, then estimates clean
foreground RGB and writes **straight (not premultiplied) alpha**. Opaque interiors
are protected. Low-contrast or unsupported regions retain their existing alpha.
Layers with no usable foreground/background seeds are left unchanged.

This is a lightweight local-color estimator, not closed-form or neural matting.
Textured backgrounds, similarly colored foreground/background, disconnected thin
parts and transparent materials can confuse it. It does not generate missing detail
or guarantee recovery of hair. Reduce radius/strength if an edge changes incorrectly.
Correct finger gaps using localized SAM refinement or erasing first; a gap outside
the narrow uncertain band is deliberately left alone. Quality on your RunPod images
still needs visual validation.

Background on the compositing model (reference only; PyMatting is not a dependency):
https://pymatting.github.io/

## Start with the editing workflow

Load [`examples/sam3_layers_edit.json`](examples/sam3_layers_edit.json).

- Choose an image, and enter one object description per line, e.g. `person` and `chair`.
- Each detection becomes a separate layer; multiple chairs remain separate objects. Overlapping/redundant prompts can produce duplicate layers; automatic deduplication is not included.
- Run once. The editor receives images and pauses downstream execution until you apply the state.
- Click **Open layer editor**. Select objects in the right-hand panel, or double-click their visible pixels in the viewport to open the mask popup.
- **Draw / Erase** edits the selected mask in original image coordinates. Use zoom and scrolling for details. Undo/redo and Apply/Cancel are available.
- **Positive / Negative click** places prompts. **Refine clicks with SAM3** queues the workflow, returns a revised mask, and pauses for review. Reopen the editor afterwards. SAM3 refinement replaces the mask; make final brush corrections after refinement.
- The right-hand list displays **front to back**. Set depth order before reconstruction.
- Shift-click or check multiple layers, then **Group selected**. Group transforms affect all members; each member keeps its own mask. Ungroup preserves appearance. Nested groups are not included.
- Drag objects to move them; use the transform fields for scale and rotation. A selected group moves together. Group pivots are the canvas center in this version.
- **Apply & Run** commits changes and queues downstream nodes.

The editing-only workflow saves transparent cutouts. Its faint original-image backdrop is an editing reference, not a reconstructed background and not part of exported composites.

## Reconstruct and arrange

Load [`examples/sam3_layers_reconstruct.json`](examples/sam3_layers_reconstruct.json).

1. Choose an image, SAM3 checkpoint, object descriptions, and **replace the reconstruction checkpoint placeholder** with your actual SD/SDXL model.
2. Describe the background without the selected objects in `background_prompt`.
3. Run and use the **first editor** to refine masks and set the back-to-front order.
4. For hidden parts, the automatic estimate intersects foreground masks with an expanded bounding box around the visible object. If it misses a hidden region, open that object's mask popup, choose **Hidden area to reconstruct**, and paint the desired region. **Automatic hidden area** clears this manual override.
5. Apply & Run. Reconstruction removes all selected objects to make a clean background. For each partially hidden layer, it inpaints its candidate region, then runs SAM3 on the generated image to recover an object-shaped alpha mask. Pixels outside the fill region stay unchanged.
6. Open the **second editor** to review and arrange reconstructed layers. Apply & Run to render/save.

Move/scale/rotate in the **second editor** to reuse ComfyUI's cached reconstruction. Change source masks or depth order in the **first editor** when regeneration is needed. Regenerated content from the same source scene keeps downstream transforms and groups while replacing stale masks. Editing a mask in the second editor changes its cutout only; it does not send edits backwards to reconstruction.

Hidden-object completion is an approximation: there is no depth model or guaranteed recovery of a fully occluded object. The automatic region can miss distant hidden parts or include unrelated foreground objects; manual regions and descriptive layer names help. Generated results require visual review. Reconstruction does not regenerate shadows or lighting when a layer is moved.

The reconstruction node uses native `InpaintModelConditioning`, `KSampler`, and VAE decoding. Sampling and SAM3 refinement run sequentially; ComfyUI handles model residency. Start with 512–768 pixel images for the first 3090 test. Peak VRAM use and model-specific quality have not been benchmarked here.

## Existing Compositor V4

[`examples/sam3_layers_v4_bridge.json`](examples/sam3_layers_v4_bridge.json) adds the existing `CompositorConfig4` and `Compositor4` nodes from **ComfyUI-enricos-nodes**. Install that pack separately.

- `Layers • Get Layer (V4 Bridge)` emits an RGBA image and an alpha mask. Connect **rgba** to a V4 image socket; leave its mask socket disconnected to avoid applying alpha twice or mixing mask conventions.
- Background occupies the first image slot; the example exposes the first two objects in the next slots. Add Get Layer nodes for more objects. V4 has eight slots, so a background leaves seven object slots.
- Set V4 Config width/height to match the source image.
- With `transformed=false`, V4 receives original-coordinate layers for its own arrangement. Set `true` to bake our editor's arrangement into full-canvas layers first.
- V4 maintains its own transforms. They are not synchronized back into this package's project.

**The popup mask editor is implemented inside the new Layers compositor. It is not injected into V4's native viewport or layer menu yet.** The bridge reuses V4 for arrangement and the normal SaveImage workflow, while avoiding dependencies on unverified frontend internals. Direct popup integration into V4 remains a follow-up.

## Save and reopen

`Layers • Save Project` writes a unique folder under ComfyUI `output/`:

- `composite.png`: final RGBA composition, honoring visibility.
- `layer_XX.png`: each untransformed RGBA layer, including hidden layers.
- `XX_name_placed.png`: each layer with its transform baked into canvas coordinates.
- `rgb_XX.png`: full RGB data retained for reversible mask editing.
- `source.png`, optional `background.png`, and `project.json`.

`Layers • Load Project` accepts that folder's name (or its absolute path inside ComfyUI output). Connect it to an editor, renderer, or V4 Get Layer nodes. PNG export is 8-bit, so save/reload can quantize float pixels by up to 1/255. Keep all project files together.

For your original **Save Layers** section, connect the `layers_rgba` batch from `Layers • Render` to a normal `SaveImage` node. Connect `composite_rgba` to a separate `SaveImage` node. These outputs do not require Qwen.

The workflow stores edited masks and transforms in its editor-state widget. Source previews are refreshed by running the workflow; they are not embedded there. After restoring a workflow, run once before opening the editor. **Reset editing state** discards that editor's saved changes on the next run.

## Nodes

| Node | Purpose |
|---|---|
| SAM3 Automatic Regions | Point-grid discovery without object names; filtered candidate masks |
| SAM3 Named Objects | One prompt per line; separate mask for each detection |
| Import Masks | Adapter for the supplied SAM3 workflow or any IMAGE + MASK batch |
| Compositor & Mask Editor | Layer arrangement, groups, brush edits and queued SAM3 clicks |
| Reconstruct | Clean background and approximate hidden-object completion |
| Alpha Matte & Edge Cleanup | Local alpha estimation and foreground edge-color recovery |
| Render | Composite, transformed RGBA layer batch, alpha batch |
| Get Layer (V4 Bridge) | Select a zero-based layer as RGBA / alpha |
| Save Project / Load Project | Portable PNG + JSON project persistence |

## Validation

```sh
python -m unittest discover -s tests -v
node --test tests/*.test.js
node --check web/layers.js
```

Tests cover pixel/mask alignment, mask replacement and ordering, affine inverses, grouping/ungrouping, visibility, project persistence, source invalidation, and native inpainting padding with stubbed inference. They do **not** establish SAM3 detection quality or diffusion reconstruction quality.

A browser-only fixture is available with `python tests/ui_harness.py` at `http://127.0.0.1:8765`. It does not execute ComfyUI or models. Browser automation was unavailable in the development environment, so popup interaction still needs live verification.

RunPod smoke checklist:

1. Editing example: identify two objects, paint/erase, refine with clicks, and inspect alpha edges.
2. Group, move, rotate, ungroup; verify the composition stays put when ungrouping.
3. Full example: inspect the clean background with layers hidden and verify reconstructed object edges.
4. Move an object in the second editor; confirm the reconstruction stage is cached.
5. Save and load the project; compare composition and masks.
6. V4 bridge: check canvas dimensions and transparency before saving.

## Upstream interfaces reviewed

- [Native ComfyUI SAM3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_sam3.py)
- [ComfyUI conditioning and sampling nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py)
- [Compositor V4](https://github.com/erosDiffusion/ComfyUI-enricos-nodes/blob/master/Compositor4.py)

No upstream compositor code is vendored or modified.
