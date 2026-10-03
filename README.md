# ComfyUI-Layers

An initial implementation of SAM3 object layers, a browser compositor with popup mask editing, and background/hidden-object reconstruction. Built for a ComfyUI server such as a RunPod 3090 instance; the editor runs in your browser.

**Status:** named SAM3 detection and the original editor have been confirmed working by the user on RunPod. Local CPU processing and JavaScript tests pass. The new zoom/thumbnail controls, automatic discovery, and diffusion reconstruction still need live RunPod validation. This remains experimental.

## Install on RunPod

1. Copy this entire `ComfyUI-Layers` directory into your RunPod installation's `ComfyUI/custom_nodes/` directory.
2. Use a ComfyUI version with native `SAM3_Detect` and the SAM3.1 checkpoint loader support. Your supplied `Sam3WF.json` should run first.
3. Put `sam3.1_multiplex_fp16.safetensors` in `ComfyUI/models/checkpoints/`.
4. For reconstruction, use FLUX.1 Fill dev with separate CLIP-L + T5XXL text encoders and the FLUX VAE (see migration below). SD/SDXL reconstruction is no longer supported. **The Qwen layered checkpoint is not a drop-in inpainting model for this node.** No additional models are downloaded automatically.
5. Restart ComfyUI and refresh the browser. Search the node menu for `Layers`.

The original named/point-grid workflows need no additional Python dependencies beyond ComfyUI's PyTorch, NumPy and Pillow. The semantic vision workflows additionally need `requirements-discovery.txt`. Do not replace the RunPod PyTorch install.

## Semantic automatic discovery (recommended automatic mode)

Start with **`examples/sam3_layers_vision_edit.json`**. For background and hidden-object
completion, use **`examples/sam3_layers_vision_reconstruct.json`** instead.

Pipeline: **Qwen VL → Review Objects → SAM3 → mask editor → matting → render/save**.
The reconstruction variant inserts reconstruction, matting and a final arrangement
editor after source mask editing. This mode replaces point-grid sampling as the primary
automatic option, while preserving the older workflows.

### Install the optional vision dependency and model on RunPod

From your actual ComfyUI directory, using the Python environment that runs ComfyUI:

```sh
python -m pip install -r custom_nodes/ComfyUI-SAMLayers/requirements-discovery.txt
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/LLM/Qwen2.5-VL-3B-Instruct
```

The second command uses the Hugging Face CLI. If `hf` is unavailable, install its
CLI in the same environment, or download the full repository through Hugging Face.
Retain all configuration, processor, tokenizer, index and weight files.

Model and files:
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main

```text
ComfyUI/
|-- models/
|   |-- LLM/
|   |   `-- Qwen2.5-VL-3B-Instruct/
|   |       |-- config.json
|   |       |-- preprocessor_config.json
|   |       |-- tokenizer_config.json
|   |       |-- tokenizer.json
|   |       |-- model.safetensors.index.json
|   |       `-- model-*.safetensors
|   `-- checkpoints/
|       `-- sam3.1_multiplex_fp16.safetensors
`-- custom_nodes/
    `-- ComfyUI-SAMLayers/
        `-- requirements-discovery.txt
```

The filenames shown are illustrative; keep the actual repository filenames and all
its files. This is a full vision-language model, **not** the Qwen Image Layered
checkpoint or its standalone ComfyUI text encoder. The runtime only loads local
files and never downloads weights automatically. `model_folder` is relative to
`ComfyUI/models/LLM/`. The 3B model is the default to keep discovery smaller; the same
loader supports a separately installed Qwen2.5-VL-7B-Instruct folder.

The vision node currently requires CUDA. It unloads ComfyUI-managed models before
loading Qwen, then releases Qwen before downstream SAM3 inference. It does not keep
both models resident intentionally. Real 3090 peak memory and detection quality
still require validation; local tests use mocked inference. Dependency compatibility
with your installed ComfyUI should also be checked after installation.

### Use the workflow

1. Choose the source image and set **whole objects** (default) or **detailed parts**.
   Use **scene_scope: full scene** (default) to include environmental layers, or
   **foreground objects** for the previous foreground-focused discovery.
2. Run. Qwen proposes names, short segmentation descriptions and instance bounding boxes.
3. Open **Review object list**. Check desired objects, rename them, change SAM descriptions,
   edit boxes, remove duplicates or add missed objects. Click a row to see its box.
   Boxes use `[left, top, right, bottom]` coordinates from 0–1000 over the original image.
4. Full-scene discovery requests separate visible cloud instances, terrain/grass,
   sky and other scene surfaces when present, with all entries checked by default.
   Background entries become normal editable masks behind foreground subjects.
   Review can select all scene layers or select objects only. Saved unchecked
   choices remain unchecked; discovering a different scope requires fresh review.
   A visible surface mask is not a completed layer behind occluding objects.
5. **Apply & Run** sends each selected description and box to SAM3 separately.
   Nearly identical masks are discarded by IoU; unrelated candidates are never unioned.
   If an object cannot be segmented, the node asks you to correct/uncheck it rather
   than silently substituting a different object.
6. Open the mask editor, review boundaries and depth order, then **Apply & Run**.
   In the reconstruction workflow, use the final editor to arrange completed layers.

### Add missed objects with clicks

Connect the SAM3 checkpoint's **MODEL** output to **Review Objects → sam_model**.
Both `sam3_layers_vision_edit.json` and `sam3_layers_vision_reconstruct.json` include
this connection. For an existing workflow, add the connection after restarting
ComfyUI and refreshing the browser to load the updated nodes and frontend.

Choose **Add object**, enter a name, and place several **Include (+)** clicks inside
the intended object. Use **Exclude (−)** clicks on unwanted areas. Choose
**Preview SAM mask** to queue detection; downstream segmentation pauses while you
review the blue highlight. Adjust clicks and preview again as needed, then choose
**Confirm object**. Repeat for other missed objects, then **Apply & Run**.

The confirmed mask is stored in the workflow review state and reused by segmentation;
it is not replaced by another text detection. Changing clicks invalidates the preview
until SAM runs again. **Advanced box coordinates** remain available; changing a box
discards its confirmed mask and returns that object to text/box segmentation.
Preview is queued inference, not instantaneous inference on each click. RunPod UI
and GPU behavior still need live verification.

The whole-object prompt explicitly asks for a person with hands, clothing and shoes
as one object. It is guidance, not a guarantee: Qwen can miss objects, invent labels,
return inaccurate boxes or include parts. Review remains essential. Bounding boxes
are supplied as SAM prompts rather than hard mask crops, so they guide instance
selection without mechanically cutting off the mask. This does not guarantee a
complete non-overlapping decomposition or solve alpha-matting errors.

If Qwen returns invalid object JSON, discovery accepts common field variants such
as `label` and `bbox_2d`, then makes one format-repair attempt if needed using the
already loaded model. It does not invent missing names or boxes. If both attempts
fail, the error includes the path to
`ComfyUI/temp/samlayers_discovery/failed_<id>.json` (or your configured temp directory).
This file contains the raw generated responses and validation errors for diagnosis.
An error about object data does not require downloading the model again. The
Transformers `min_pixels`/`max_pixels` deprecation warning is separate from JSON
validation failures.

Discovery requests absolute pixel boxes from Qwen on an explicitly resized image
(dimensions divisible by 28, approximately 0.8 megapixels maximum). It disables
further processor resizing and converts these boxes to the editor's 0–1000 scale.
It does not guess the coordinate scale from whether numbers exceed 1000.
Diagnostics include the image dimensions used for generation. Existing saved
review lists keep their normalized coordinate format. The full-scene prompt requests visible foreground and background layers,
avoids duplicate grass/hill descriptions for the same region, and requests back-to-front order;
review the results because correct JSON does not guarantee correct detections.
Qwen's coordinate convention is described at:
https://qwenlm.github.io/blog/qwen2.5-vl/

## Point-grid discovery (advanced fallback)

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
| edge_radius | 1 | Width of the uncertain boundary in source pixels (1–16). Start at 1–2 for fine fingers or hair. |
| matte_strength | 0.25 | Blend between the original mask and estimated partial transparency. Zero keeps the original alpha. |
| cleanup_strength | 0.35 | Strength of foreground color recovery on partially transparent edges. Zero keeps original RGB. |
| layer_index | -1 | Process every layer. Otherwise choose a zero-based layer index. |

The conservative defaults reduce fragmentation on patterned objects. Even at full
matte strength, initially opaque pixels remain at least 0.9 alpha and initially clear
pixels remain at most 0.1 alpha. Existing fractional alpha can change by at most 0.2
per pass. This intentionally limits recovery from inaccurate binary masks; use local
SAM refinement or the brush to fix actual holes and gaps.

If you used an older workflow, its saved control values are not automatically changed.
Set radius to 1, matte strength to 0.25, and cleanup to 0.35, or load a regenerated
example. Rerun from the upstream masks: this safeguard does not reconstruct detail
already erased in a saved project. If fragmentation remains with both strengths zero,
inspect the incoming SAM masks. Automatic point-grid discovery may return overlapping
object parts; named-object detection is the more controlled option for a whole person.

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

1. Choose an image, SAM3 checkpoint, object descriptions, and **select FLUX.1 Fill dev, its dual text encoders, and ae.safetensors** in the three reconstruction loaders.
2. Describe the background without the selected objects in `background_prompt`.
3. Run and use the **first editor** to refine masks and set the back-to-front order.
4. For hidden parts, the automatic estimate intersects foreground masks with an expanded convex envelope of the visible silhouette. If it misses a hidden region, open that object's mask popup, choose **Hidden area to reconstruct**, and paint the desired region. **Automatic hidden area** clears this manual override.
5. Apply & Run. Reconstruction removes selected foreground objects to make a clean background;
   layers marked background are excluded from this removal union to avoid erasing
   the entire scene when sky or terrain is selected. For each partially hidden layer, it inpaints its candidate region, then runs SAM3 on the generated image to recover an object-shaped alpha mask. Pixels outside the fill region stay unchanged.
6. Open the **second editor** to review and arrange reconstructed layers. Apply & Run to render/save.

Move/scale/rotate in the **second editor** to reuse ComfyUI's cached reconstruction. Change source masks or depth order in the **first editor** when regeneration is needed. Regenerated content from the same source scene keeps downstream transforms and groups while replacing stale masks. Editing a mask in the second editor changes its cutout only; it does not send edits backwards to reconstruction.

Hidden-object completion is an approximation: there is no depth model or guaranteed recovery of a fully occluded object. The automatic region can miss distant hidden parts or include unrelated foreground objects; manual regions and descriptive layer names help. Generated results require visual review. Reconstruction does not regenerate shadows or lighting when a layer is moved.

The reconstruction node uses native `InpaintModelConditioning`, `KSampler`, and VAE decoding. Sampling and SAM3 refinement run sequentially; ComfyUI handles model residency. Start with 512–768 pixel images for the first 3090 test. Peak VRAM use and model-specific quality have not been benchmarked here.

### Background removal quality controls

The reconstruction node separates the background removal mask from the editable
cutout alpha. Every nonzero cutout pixel is fully removed before filling, so soft
edges cannot blend the original object back into the generated background.
`removal_margin` expands that solid area (default **12 source pixels**), and
`removal_feather` adds a soft transition outside it (default **4 pixels**).
Neither control changes the object's editable mask. Pixels outside the removal
mask and its feather remain unchanged. FLUX conditioning uses a separate binary
mask covering the entire blend plus a 16-pixel guard at processing resolution.
The final composite uses the original soft blend, so this extra inference coverage
does not enlarge the edited area. Increase the margin for residual outlines;
larger margins also regenerate more of the surrounding scene.

Existing workflows use these defaults when the new optional inputs are absent.
`expand_pixels` is a separate control for estimating hidden-object completion
regions; it does not control background removal.

For a background-only test, set `complete_hidden` to **false**. Setting it to true
also attempts to reconstruct object parts covered by foreground layers, such as
the chair behind a person. Those generated parts are estimates, not recovered
original pixels. Background removal runs in either mode.

The removal-mask regression tests pass locally. The integrated FLUX Fill path
still needs a fresh RunPod reconstruction after deployment; a softer transition
cannot guarantee a correct generated scene.

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
| Discover Objects (Qwen VL) | Local vision-language discovery of names and boxes |
| Review Objects | Editable checklist, descriptions and boxes; pauses before segmentation |
| Segment Reviewed Objects (SAM3) | Per-instance text + box prompts and duplicate filtering |
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

### Direct compositor controls

The layer editor supports dragging cutouts to move them, proportional resizing
with the four selection corner handles, and dragging the grip beside a layer name
to change its stacking order. The list reads front to back. Bring to front / Send
to back and the existing up/down buttons are available without dragging.

With the canvas focused, arrow keys nudge the selection by one image pixel;
Shift+arrow nudges by ten. Grouped layers move, resize, and reorder together.
Resize and move actions support Undo/Redo, and Apply & Run saves transforms and
stacking in the workflow's editor state. Resizing preserves aspect ratio; numeric
Scale and Rotation controls remain available. These controls work on existing
layer masks and do not automatically discover additional background layers.

Rotate a selected mask and its cutout together by dragging the round handle
outside the selection. Rotation uses the selection's center, including grouped
or multiple selected layers. Hold Shift while dragging to snap the rotation delta
to 15-degree increments. Numeric Rotation remains available for exact angles;
rotation supports Undo/Redo and is saved with Apply & Run.

The default `foreground removal` reconstruction mode preserves environmental
features in a flattened backdrop. Moving cloud cutouts in that mode can reveal a
stationary copy beneath them. Disabling `complete_hidden` also leaves holes where
other objects originally occluded a cutout.

For independently movable scene layers, choose `independent scene layers` in
**Reconstruct background and hidden parts**, and enable `complete_hidden`. Put the
surface that should fill the scene (for example sky) at the back of the layer list.
This mode fills all gaps in that surface, including clouds, terrain and subjects,
and uses it as a full-canvas editable base layer. It does not retain a second
flattened backdrop. The IMAGE output is a preview of the completed base; do not
composite that output underneath the editable stack or it will duplicate the base.

Layers above the base use the original occluder masks and a silhouette envelope to
estimate hidden regions. Local crops include the target and bounded context.
Completed object shapes are resegmented; named continuous surfaces retain their
bounded filled alpha region.
Paint **Hidden area to reconstruct** to correct an estimated completion region.
Transforms are applied after reconstruction. Moving the completed base itself can
expose transparency at canvas edges. This mode derives a short positive base prompt from its
layer name (overriding `background_prompt`). It does not list other scene objects
in that prompt. Use a descriptive base name and review the layer order first.
The new mode has CPU regression coverage; diffusion fill quality and completed
cloud shapes still need RunPod validation. It cannot guarantee invisible geometry.

Full-scene discovery starts with three focused prompts with one Qwen model load:
foreground subjects, sky/cloud instances, then ground/structural surfaces using
the remaining layer budget. If sky was omitted and capacity remains, a focused sky check
runs and may return no sky for scenes without it. Results are merged with backgrounds behind subjects. A collective
`clouds` entry triggers the bounded repair pass requesting individual cloud boxes.
This avoids the observed single-pass result that omitted the person and grouped
both clouds. Model accuracy still requires live review; the repair is not a
guarantee that every visible instance will be found.

Reviewed-instance segmentation retains multiple SAM3 text candidates until matching
them to each reviewed bounding box. This prevents an early single-result limit
from returning the same highest-confidence cloud for separate left/right entries.
RunPod validation of a98d639 produced five separate masks: sky, left cloud, right
cloud, grass, and character. The right-cloud and sky masks were visually checked.
That validated discovery and segmentation, not independent layer completion.


## FLUX Fill migration (inpainting only)

Every background and hidden-layer fill now uses **FLUX.1 Fill dev**. Existing
saved workflows must replace the reconstruction SD checkpoint loader with:

- `UNETLoader`: `flux1-fill-dev.safetensors`, `fp8_e4m3fn`; connect MODEL.
- `DualCLIPLoader`: `clip_l.safetensors` + `t5xxl_fp8_e4m3fn_scaled.safetensors`,
  type `flux`, device `default`; connect CLIP.
- `VAELoader`: `ae.safetensors`; connect VAE.

Keep the separate SAM3 checkpoint loader connected to `sam_model` and `sam_clip`.
Models go in `models/diffusion_models/`, `models/text_encoders/`, and `models/vae/`
respectively. Use the regenerated reconstruction examples as wiring references.
SD/SDXL and ordinary FLUX dev are rejected; there is no automatic fallback.

Set the retained `cfg` widget to **1** in older workflows. Actual sampler CFG is
always 1; `flux_guidance` controls the embedded guidance (default 30). Inpainting
uses native FLUX Fill conditioning, 16-pixel padding and exact final compositing
outside the requested mask. Conventional negative conditioning is unused at CFG 1. The retained
`negative_prompt` field is now ignored rather than appending object exclusions to
the positive prompt; those lists risk introducing the very objects being removed. DifferentialDiffusion is not required because this path supplies no
sampler noise mask; the mask is supplied through Fill conditioning.

Set `style_prompt` for **all** base and object fills. For the current example:
`Simple flat-color cartoon illustration, solid colors, crisp smooth outlines;
match the existing blue sky, white clouds and green grass; no photographic texture.`
The base name controls what surface is generated; style_prompt controls its look.
The background_prompt remains the content instruction in foreground-removal mode.

Native implementation references:
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_flux.py

Local tests validate conditioning, model rejection, alignment, unchanged pixels,
and loader wiring. They do not validate generated cloud shapes. The existing
completion-region estimation and SAM resegmentation still require visual review;
this migration does not claim to have solved those quality limitations.


### Local completion crops and region controls

Object fills run on a crop around the visible target and its completion region,
with `completion_context` pixels of context (default 64, minimum crop side 256
unless the source is smaller). `completion_resolution` sets the crop's longest
processing side (default 768); dimensions align to FLUX's 16-pixel grid. The result
is mapped back to source coordinates. Final RGB changes are restricted to the
completion region plus a narrow soft transition into existing target pixels,
controlled by `removal_feather`. The transition never expands the layer silhouette.
Pixels beyond it remain exact; set `removal_feather=0` to preserve all originally
visible pixels. The hole itself is fully replaced, including softly painted holes.
Generated pixels are blended once at source resolution, avoiding a squared feather
weight or a hard cut back to the hole boundary.
The full-scene base fill still runs at source resolution.

Automatic regions use a convex envelope of the visible silhouette, expanded by
`expand_pixels`, intersected with front-layer occluders. This avoids unrelated
bounding-box corners and bridges interrupted visible pieces, but can overestimate
concave shapes. A painted **Hidden area to reconstruct** remains authoritative.
Occluder pixels elsewhere inside the context crop are also removed for inference;
SAM may accept connected parts of the target within that generated context.
The initial automatic hole is a generation hint, not a hard clipping polygon.
Disconnected instances are excluded. Painted completion regions remain hard limits.

Background layers named sky, grass, hill(s), ground, floor, wall, ceiling or water
are treated as continuous surfaces: their bounded completion alpha is retained
without SAM punching out the old occluder silhouette. Other objects, including
clouds, are resegmented within the local crop. A completion that adds no object
pixels now raises an actionable error instead of silently succeeding. This check
is not a guarantee of complete or correct hidden geometry.

Prompts are short positive instructions: `Continuous sky`, `A complete cloud`,
or `Continuous grass`, followed by style_prompt. Left/right positional suffixes
are stripped from crop prompts, since their full-scene positions no longer apply.
For the cartoon test use: `Flat-color cartoon, uniform solid colors and smooth
outlines matching the visible image.` Avoid scene inventories and exclusion lists.

Local tests cover crop placement, resizing/compositing registration, exact pixels
outside the blend, binary conditioning coverage, single-pass seam weights,
manual regions, silhouette bounds, surface alpha and failed
object completion. Generated quality still requires a RunPod visual test after
these changes are pushed.


### Reconstruction diagnostics

For a diagnostic run, start ComfyUI with `SAMLayers_DEBUG_RECONSTRUCTION=1`.
Each reconstruction creates a unique `output/samlayers_diagnostics_*` directory.
Capture is off by default and does not change sampling settings. These files
contain the source image and generated content; they remain in ComfyUI output.

The capture includes original masks, each hole and blend mask, crop coordinates,
FLUX conditioning masks and prompts/seeds, raw VAE-decoded FLUX pixels before
compositing, resized results, SAM inputs and completion alpha, and all layers
before matting. Compare these with the regular saved project's final RGB and
alpha to locate artifacts in generation, blending, segmentation, or matting.
A failed completion leaves its preceding captures available for inspection.
Restart without the environment variable to disable capture. GPU visual
validation still requires a diagnostic run; previous runs cannot recover raw
FLUX pixels from the final saved layers.


### Reconstruction color and alpha joins

Before compositing, reconstruction estimates a smooth RGB correction from
unremoved source context and propagates it through the generated region. This
corrects broad color/lighting drift without imposing a palette or flattening
generated texture. Corrections are bounded to 0.25 per channel; fewer than 16
reference pixels leave the output uncorrected. This is not a style classifier,
texture validator, or guarantee of a seamless result. Abrupt material changes,
small reference regions, and large hallucinations still require visual review.

Automatic object completion retains SAM components connected to the visible
target inside generated context, instead of clipping them to the initial convex
hole estimate. Newly accepted pixels receive generated RGB too. A two-pixel
closing near the hole repairs narrow alpha cracks; it does not repair broad
segmentation errors. Manual painted regions are never expanded by this merge.
Continuous-surface completion keeps its estimated region with a narrow join band.
Diagnostic runs additionally capture `color-matched`, `accepted-alpha`,
`accepted-blend`, and `accepted-rgb` to distinguish each correction stage.

These changes have synthetic regression coverage for textured color offsets,
lighting gradients, mask joins, and rounded completion beyond the initial hole.
They still need RunPod validation on real photographs, paintings, illustrations,
and transparent or fine-edged objects before production use.
