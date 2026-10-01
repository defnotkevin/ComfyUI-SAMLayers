"""Regenerate example ComfyUI frontend workflows; no models/downloads required."""
import json
from pathlib import Path

ROOT=Path(__file__).parent

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
    def save(self,name):
        data={'last_node_id':len(self.nodes),'last_link_id':len(self.links),'nodes':self.nodes,'links':self.links,'groups':[], 'config':{},'extra':{},'version':.4}
        (ROOT/name).write_text(json.dumps(data,indent=2))

def base():
    w=Workflow()
    image=w.node('LoadImage','Source image',[0,0],[],[('IMAGE','IMAGE'),('MASK','MASK')],['example.png','image'])
    sam=w.node('CheckpointLoaderSimple','SAM3.1 checkpoint',[0,300],[],[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],['sam3.1_multiplex_fp16.safetensors'])
    detect=w.node('LayersSAM3','1 • Name objects (one per line)',[400,0],[('image','IMAGE'),('sam_model','MODEL'),('sam_clip','CLIP')],[('project','LAYERS_PROJECT')],['person\nchair',.5])
    edit=w.node('LayersEditor','2 • Edit masks and layer order',[800,0],[('project','LAYERS_PROJECT'),('sam_model','MODEL')],[('project','LAYERS_PROJECT')],[''])
    w.link(image,0,detect,0);w.link(sam,0,detect,1);w.link(sam,1,detect,2);w.link(detect,0,edit,0);w.link(sam,0,edit,1)
    return w,sam,edit

def finish(w,project,x):
    render=w.node('LayersComposite','Render composition',[x,0],[('project','LAYERS_PROJECT')],[('composite_rgba','IMAGE'),('layers_rgba','IMAGE'),('layer_alpha','MASK')])
    preview=w.node('PreviewImage','Composite preview',[x+400,0],[('images','IMAGE')],[])
    save=w.node('LayersSave','Save editable project + PNG layers',[x,300],[('project','LAYERS_PROJECT')],[('project_directory','STRING')],['layers'])
    w.link(project,0,render,0);w.link(render,0,preview,0);w.link(project,0,save,0)

w,sam,edit=base();finish(w,edit,1200)
w.node('Note','How to use',[800,350],[],[],['Choose an image, enter objects and Run once. Open the editor, adjust masks/order, then Apply & Run. This example exports cutouts only: no background reconstruction.'])
w.save('sam3_layers_edit.json')

w,sam,edit=base()
model=w.node('CheckpointLoaderSimple','Choose your SD/SDXL inpainting checkpoint',[800,400],[],[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],['SELECT_YOUR_INPAINT_CHECKPOINT.safetensors'])
rebuild=w.node('LayersReconstruct','3 • Reconstruct background and hidden parts',[1200,0],[(n,t)for n,t in [('project','LAYERS_PROJECT'),('model','MODEL'),('clip','CLIP'),('vae','VAE'),('sam_model','MODEL'),('sam_clip','CLIP')]],[('project','LAYERS_PROJECT'),('background','IMAGE')],['empty room, continuous background, no people, no foreground objects','artifacts, duplicated objects, text, watermark',True,32,0,'fixed',25,6])
w.link(edit,0,rebuild,0)
for i in range(3):w.link(model,i,rebuild,i+1)
w.link(sam,0,rebuild,4);w.link(sam,1,rebuild,5)
arrange=w.node('LayersEditor','4 • Arrange reconstructed layers',[1600,0],[('project','LAYERS_PROJECT'),('sam_model','MODEL')],[('project','LAYERS_PROJECT')],[''])
w.link(rebuild,0,arrange,0);w.link(sam,0,arrange,1);finish(w,arrange,2000)
w.node('Note','Two editing stages',[1200,450],[],[],['Set the inpainting checkpoint and background prompt before running. First editor: masks, completion regions, depth order. Second editor: arrange cached reconstructed layers. Change masks in the FIRST editor to regenerate hidden content. Then open the second editor again. Each editor pauses on new source data.'])
w.save('sam3_layers_reconstruct.json')

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
