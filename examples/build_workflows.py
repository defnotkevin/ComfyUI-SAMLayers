"""Regenerate example ComfyUI frontend workflows; no models/downloads required."""
import json
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).parent


def requirement_notes(reconstruct=False, v4=False):
    """Workflow-specific setup notes; URLs remain fully visible in Markdown."""
    software = """# Requirements: ComfyUI and custom nodes

## ComfyUI

Use a ComfyUI build with native **SAM3_Detect** and **SAM3.1 checkpoint** support. Verify that your original SAM3 workflow runs first. If SAM3_Detect is missing, update ComfyUI; a separate SAM3 custom node pack is not required by this workflow.

https://github.com/Comfy-Org/ComfyUI

Update instructions:

https://docs.comfy.org/installation/update_comfyui

## Required: ComfyUI-SAMLayers

Provides all nodes whose names begin with **Layers**. Copy or clone this repository into `ComfyUI/custom_nodes/ComfyUI-SAMLayers/`.

https://github.com/defnotkevin/ComfyUI-SAMLayers

This pack uses ComfyUI's existing PyTorch, NumPy and Pillow. No additional Python packages are required by SAMLayers.
"""
    if v4:
        software += """
## Required for this bridge: ComfyUI-enricos-nodes

Provides **CompositorConfig4** and **Compositor4**. Install the pack in `ComfyUI/custom_nodes/ComfyUI-enricos-nodes/` and follow its own dependency instructions.

https://github.com/erosDiffusion/ComfyUI-enricos-nodes

This pack is required to execute the V4 branch included here; it is not needed by the editing-only or reconstruction-only examples.
"""
    else:
        software += """
The new Layers compositor is included in SAMLayers. ComfyUI-enricos-nodes is not required for this example.
"""
    software += """
## Alpha matting and edge cleanup

**Layers - Alpha Matte & Edge Cleanup** is included in SAMLayers. It uses CPU/PyTorch local color estimation; no extra model, package or download is needed. Start with edge_radius 3, matte_strength 1.0 and cleanup_strength 0.75. Set both strengths to zero to bypass. It refines a narrow boundary band, not large incorrectly selected background regions.

## After installing

Restart the ComfyUI server and refresh the browser. Model files go on the **RunPod server**, not just on the computer running your browser. No model downloads happen automatically.
"""

    models = """# Requirements: model files

## SAM3.1 — required

File: `sam3.1_multiplex_fp16.safetensors`

Place it in `ComfyUI/models/checkpoints/` and select it in the **SAM3.1 checkpoint** loader. That loader supplies SAM3's MODEL and CLIP outputs; no separate text-encoder or VAE file is required for segmentation.

Model page:

https://huggingface.co/Comfy-Org/sam3.1

Direct checkpoint download:

https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors
"""
    if reconstruct:
        models += """
## Inpainting checkpoint — also required

Choose a compatible **single-file SD/SDXL checkpoint with matching CLIP and VAE**, preferably an inpainting checkpoint. Place it in `ComfyUI/models/checkpoints/`.

The **Choose your SD/SDXL inpainting checkpoint** loader contains `SELECT_YOUR_INPAINT_CHECKPOINT.safetensors`. This is a placeholder, not a downloadable filename. Replace it with your installed checkpoint before running. Connect all three outputs from that same loader: MODEL, CLIP and VAE.

### Concrete download option: SD 1.5 Inpainting

File: `sd-v1-5-inpainting.ckpt`

Model page:

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting

Direct single-file checkpoint download:

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt

Select this `.ckpt` file in the reconstruction loader if using this option. The repository's separate Diffusers component folders are not needed for this loader. This is an example option, not a GPU-tested model recommendation for this package.

Qwen Image Layered is not required and is not a drop-in inpainting checkpoint for this node. A standalone UNet file alone is insufficient for the example's checkpoint loader.
"""
    else:
        models += """
## No reconstruction model needed

This workflow extracts and edits transparent cutouts only. It does not fill the background or reconstruct hidden object parts. No SD/SDXL inpainting model or Qwen Image Layered model is required.
"""

    checkpoint_line = "|       `-- sam3.1_multiplex_fp16.safetensors"
    if reconstruct:
        checkpoint_line = "|       |-- sam3.1_multiplex_fp16.safetensors\n|       `-- sd-v1-5-inpainting.ckpt  (or your chosen checkpoint)"
    pack_lines = "|   `-- ComfyUI-SAMLayers/\n|       |-- __init__.py\n|       |-- nodes.py\n|       |-- web/\n|       `-- examples/"
    if v4:
        pack_lines = "|   |-- ComfyUI-SAMLayers/\n|   |   |-- __init__.py\n|   |   |-- nodes.py\n|   |   |-- web/\n|   |   `-- examples/\n|   `-- ComfyUI-enricos-nodes/"
    directories = f"""# File placement on the ComfyUI server

Use your actual ComfyUI installation directory as the root below. RunPod template paths vary; do not assume a fixed `/workspace` path.

```text
ComfyUI/
|-- models/
|   `-- checkpoints/
{checkpoint_line}
|-- custom_nodes/
{pack_lines}
|-- input/
|   `-- your_source_image.png
`-- output/
    `-- layers_<unique_id>/        (created when saving)
        |-- composite.png
        |-- layer_00.png
        |-- rgb_00.png
        |-- source.png
        `-- project.json
```

Upload the source image using **LoadImage**, or place it in `input/` and select it. Example workflows live in the pack's `examples/` directory; open or drag the JSON file into ComfyUI.

Keep `__init__.py` directly inside the custom node pack directory, not inside an extra nested folder from extracting a ZIP. Keep the entire pack together, including its `web/` directory.

The `output/` directory is generated output, not a model installation location. The save node writes additional per-layer files; reconstruction projects also include `background.png`.
"""
    if v4:
        directories += """
## V4 setup after loading

Set **CompositorConfig4** width and height to the source image dimensions. The background uses image1; RGBA objects use image2 and image3. Leave mask inputs disconnected because transparency is already embedded in RGBA. Add Get Layer nodes for more objects (eight V4 slots total, including the background).
"""
    return [
        ('Requirements - ComfyUI and node packs', software, [720, 1150]),
        ('Requirements - Models and full download URLs', models, [720, 1450 if reconstruct else 850]),
        ('Installation - ComfyUI directory layout', directories, [720, 1100]),
    ]


class Workflow:
    def __init__(self):self.nodes=[];self.links=[]
    def node(self,kind,title,pos,inputs,outputs,widgets=None):
        n={'id':len(self.nodes)+1,'type':kind,'title':title,'pos':pos,'size':[340,240],
           'flags':{},'order':len(self.nodes),'mode':0,'inputs':[{'name':a,'type':b,'link':None} for a,b in inputs],
           'outputs':[{'name':a,'type':b,'links':[]} for a,b in outputs],
           'properties':{'Node name for S&R':kind},'widgets_values':widgets or []}
        self.nodes.append(n);return n
    def link(self,a,slot,b,target):
        i=len(self.links)+1;self.links.append([i,a['id'],slot,b['id'],target,a['outputs'][slot]['type']]);a['outputs'][slot]['links'].append(i);b['inputs'][target]['link']=i
    def save_automatic(self,name):
        workflow=deepcopy(self)
        detect=next(n for n in workflow.nodes if n['type']=='LayersSAM3')
        removed=detect['inputs'][2]['link']
        detect.update(type='LayersSAM3Auto', title='1 • Discover regions automatically',
                      inputs=detect['inputs'][:2], widgets_values=[8,.002,.95,.8,64])
        detect['properties']['Node name for S&R']='LayersSAM3Auto'
        remaining=[link for link in workflow.links if link[0]!=removed]
        remap={link[0]:i+1 for i,link in enumerate(remaining)}
        for node in workflow.nodes:
            for inp in node['inputs']:
                if inp['link'] is not None: inp['link']=remap.get(inp['link'])
            for out in node['outputs']:
                out['links']=[remap[x] for x in out['links'] if x in remap]
            if node['type']=='Note':
                node['widgets_values']=[text.replace('enter objects and Run once', 'Run automatic discovery once') for text in node['widgets_values']]
        workflow.links=[[remap[link[0]],*link[1:]] for link in remaining]
        note=workflow.node('MarkdownNote','Automatic discovery - read first',[0,650],[],[],[
            "# Automatic region discovery\n\nNo object names or additional models are needed. "
            "SAM3 is prompted at a grid of points and near-duplicate masks are filtered. "
            "An 8 x 8 grid runs up to 64 separate segmentation passes; increasing density is slower.\n\n"
            "Regions have generic names. Rename them in the editor and review overlaps and depth order before reconstruction. "
            "This heuristic can miss small objects, split an object into parts, or include background regions. "
            "It does not guarantee every object or a complete non-overlapping decomposition.\n\n"
            "min_area / max_area are fractions of image area. duplicate_iou filters similar masks. "
            "max_layers caps the number retained. Increase points_per_side to search more densely.\n\n"
            "The node displays a source thumbnail after running. Open layer editor to select the regions you want to manipulate. "
            "Use the floating Edit mask button on a selected object. Zoom up to 3200%; mouse wheel zooms around the pointer, Shift-wheel scrolls, "
            "and middle-button drag pans."])
        note['size']=[720,500]
        workflow.save(name)

    def save(self,name):
        # Annotate a copy so saving the reconstruction example does not alter
        # the shared graph subsequently extended into the V4 bridge example.
        nodes=deepcopy(self.nodes)
        for node in nodes:
            if node['type'] == 'Note':
                node['type'] = 'MarkdownNote'
                node['properties']['Node name for S&R'] = 'MarkdownNote'
        reconstruct = any(n['type'] == 'LayersReconstruct' for n in nodes)
        v4 = any(n['type'] == 'Compositor4' for n in nodes)
        notes = requirement_notes(reconstruct, v4)
        for index, (title, text, size) in enumerate(notes):
            nodes.append({'id':len(nodes)+1, 'type':'MarkdownNote', 'title':title,
                          'pos':[-2300+index*760,0], 'size':size, 'flags':{},
                          'order':len(nodes), 'mode':0, 'inputs':[], 'outputs':[],
                          'properties':{'Node name for S&R':'MarkdownNote'},
                          'widgets_values':[text], 'color':'#243447', 'bgcolor':'#172330'})
        groups=[{'title':'START HERE - Requirements and installation',
                 'bounding':[-2330,-80,2290,max(size[1] for _,_,size in notes)+120],
                 'color':'#3f789e','font_size':24,'flags':{}}]
        data={'last_node_id':len(nodes),'last_link_id':len(self.links),'nodes':nodes,'links':self.links,'groups':groups, 'config':{},'extra':{},'version':.4}
        (ROOT/name).write_text(json.dumps(data,indent=2))

def base():
    w=Workflow()
    image=w.node('LoadImage','Source image',[0,0],[],[('IMAGE','IMAGE'),('MASK','MASK')],['example.png','image'])
    sam=w.node('CheckpointLoaderSimple','SAM3.1 checkpoint',[0,300],[],[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],['sam3.1_multiplex_fp16.safetensors'])
    detect=w.node('LayersSAM3','1 • Name objects (one per line)',[400,0],[('image','IMAGE'),('sam_model','MODEL'),('sam_clip','CLIP')],[('project','LAYERS_PROJECT')],['person\nchair',.5])
    edit=w.node('LayersEditor','2 • Edit masks and layer order',[800,0],[('project','LAYERS_PROJECT'),('sam_model','MODEL')],[('project','LAYERS_PROJECT')],[''])
    w.link(image,0,detect,0);w.link(sam,0,detect,1);w.link(sam,1,detect,2);w.link(detect,0,edit,0);w.link(sam,0,edit,1)
    return w,sam,edit

def matte(w,project,x,y=0):
    node=w.node('LayersMatte','Alpha matting and edge-color cleanup',[x,y],
                [('project','LAYERS_PROJECT')],[('project','LAYERS_PROJECT'),('refined_alpha','MASK')],
                [3,1.,.75,-1])
    w.link(project,0,node,0)
    return node

def finish(w,project,x,refine=True):
    if refine:
        project=matte(w,project,x)
        x+=400
    render=w.node('LayersComposite','Render composition',[x,0],[('project','LAYERS_PROJECT')],[('composite_rgba','IMAGE'),('layers_rgba','IMAGE'),('layer_alpha','MASK')])
    preview=w.node('PreviewImage','Composite preview',[x+400,0],[('images','IMAGE')],[])
    save=w.node('LayersSave','Save editable project + PNG layers',[x,300],[('project','LAYERS_PROJECT')],[('project_directory','STRING')],['layers'])
    w.link(project,0,render,0);w.link(render,0,preview,0);w.link(project,0,save,0)

w,sam,edit=base();finish(w,edit,1200)
w.node('Note','How to use',[800,350],[],[],['Choose an image, enter objects and Run once. Open the editor, adjust masks/order, then Apply & Run. This example exports cutouts only: no background reconstruction.'])
w.save('sam3_layers_edit.json')
w.save_automatic('sam3_layers_auto_edit.json')

w,sam,edit=base()
model=w.node('CheckpointLoaderSimple','Choose your SD/SDXL inpainting checkpoint',[800,400],[],[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],['SELECT_YOUR_INPAINT_CHECKPOINT.safetensors'])
rebuild=w.node('LayersReconstruct','3 • Reconstruct background and hidden parts',[1200,0],[(n,t)for n,t in [('project','LAYERS_PROJECT'),('model','MODEL'),('clip','CLIP'),('vae','VAE'),('sam_model','MODEL'),('sam_clip','CLIP')]],[('project','LAYERS_PROJECT'),('background','IMAGE')],['empty room, continuous background, no people, no foreground objects','artifacts, duplicated objects, text, watermark',True,32,0,'fixed',25,6])
w.link(edit,0,rebuild,0)
for i in range(3):w.link(model,i,rebuild,i+1)
w.link(sam,0,rebuild,4);w.link(sam,1,rebuild,5)
refined=matte(w,rebuild,1550,-350)
arrange=w.node('LayersEditor','4 • Arrange reconstructed layers',[1950,0],[('project','LAYERS_PROJECT'),('sam_model','MODEL')],[('project','LAYERS_PROJECT')],[''])
w.link(refined,0,arrange,0);w.link(sam,0,arrange,1);finish(w,arrange,2350,refine=False)
w.node('Note','Two editing stages',[1200,450],[],[],['Set the inpainting checkpoint and background prompt before running. First editor: masks, completion regions, depth order. Second editor: arrange cached reconstructed layers. Change masks in the FIRST editor to regenerate hidden content. Then open the second editor again. Each editor pauses on new source data.'])
w.save('sam3_layers_reconstruct.json')
w.save_automatic('sam3_layers_auto_reconstruct.json')

# Existing V4 accepts RGBA IMAGE sockets, avoiding its inverse-mask convention.
v4=w.node('CompositorConfig4','Optional • Existing Compositor V4 bridge',[2000,650],sum(([('image'+str(i),'IMAGE'),('mask'+str(i),'MASK')]for i in range(1,9)),[]),[('config','COMPOSITOR_CONFIG')],[512,512,0,False,False,False,'PNG Level 0 (fastest)','output'])
# Background is socket 1; the first two objects are sockets 2 and 3.
w.link(rebuild,1,v4,0)
for i in range(2):
    get=w.node('LayersGetLayer',f'Object {i+1} → V4',[1600,650+i*280],[('project','LAYERS_PROJECT')],[('rgba','IMAGE'),('alpha','MASK'),('name','STRING')],[i,False])
    w.link(arrange,0,get,0);w.link(get,0,v4,2*(i+1))
comp=w.node('Compositor4','Existing Compositor V4',[2450,650],[('config','COMPOSITOR_CONFIG')],[('image','IMAGE'),('fabricData_output','STRING'),('imageName_output','STRING')],['','','0',''])
w.link(v4,0,comp,0)
w.node('Note','V4 bridge settings',[2450,1000],[],[],['Requires ComfyUI-enricos-nodes. Set Config width/height to the source image dimensions. Background uses image1; objects use image2/image3. Add Get Layer nodes for more objects (V4 has eight slots total). RGBA already includes transparency: leave mask inputs disconnected. V4 has a separate arrangement from our editor.'])
w.save('sam3_layers_v4_bridge.json')
