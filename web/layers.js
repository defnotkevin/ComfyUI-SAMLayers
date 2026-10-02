import { addZoomControls } from './zoom.js';
import { app } from '../../scripts/app.js';
import { hideEditorState, editorStatus } from './editor_status.js';
import { layerMatrix, matrix, invertPoint, ungroup, transformPoint, resizeTransform, moveLayerBlock, rotateTransform } from './math.js';

const clone = x => JSON.parse(JSON.stringify(x));
const el = (tag, text, parent, cls) => {
    const e=document.createElement(tag); if(text != null) e.textContent=text;
    if(cls)e.className=cls; if(parent)parent.append(e); return e;
};
const button = (parent,label,action) => { const b=el('button',label,parent);b.type='button';b.onclick=action;return b; };
const canvas = (w,h) => {const c=document.createElement('canvas');c.width=w;c.height=h;return c;};
const load = src => new Promise((ok,fail)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=()=>fail(new Error('Cannot load layer image'));i.src=src;});
const css=document.createElement('link');css.rel='stylesheet';css.href=new URL('./layers.css',import.meta.url).href;document.head.append(css);

export class Editor {
    constructor(node, payload) {
        this.node=node;this.payload=payload;this.state=clone(payload.state);this.selected=new Set();this.history=[];this.future=[];
    }
    async open() {
        this.dialog=el('dialog',null,document.body,'layers-dialog');
        const head=el('header',null,this.dialog);el('strong','Layers',head);
        this.status=el('span','Loading…',head,'layers-status');
        button(head,'Cancel',()=>this.close());
        button(head,'Apply & Run',()=>this.commit());
        this.body=el('div',null,this.dialog,'layers-body');
        const left=el('section',null,this.body,'layers-main');
        const toolbar=el('div',null,left,'layers-toolbar');
        button(toolbar,'Undo',()=>this.undo());button(toolbar,'Redo',()=>this.redo());
        button(toolbar,'Group selected',()=>this.group());button(toolbar,'Ungroup',()=>this.ungroup());
        this.scroll=el('div',null,left,'layers-scroll');
        this.stage=el('div',null,this.scroll,'layers-stage');
        this.view=canvas(this.state.width,this.state.height);this.stage.append(this.view);
        this.floatingEdit=button(this.stage,'Edit mask',()=>this.editSelected());
        this.floatingEdit.className='layers-floating-edit';this.floatingEdit.hidden=true;
        this.zoomControl=addZoomControls(toolbar,this.scroll,this.view,()=>this.draw());
        this.side=el('aside',null,this.body);
        this.dialog.addEventListener('cancel',()=>this.close());this.dialog.showModal();
        this.original=await load(this.payload.image);
        this.background=this.payload.background ? await load(this.payload.background):null;
        this.rgb=new Map();
        for(let i=0;i<this.state.layers.length;i++) this.rgb.set(this.state.layers[i].id,this.payload.rgbs ? await load(this.payload.rgbs[i]):this.original);
        await this.rebuild();this.zoomControl.fit();this.bindViewport();this.list();this.draw();
        this.status.textContent=this.payload.reconstructed?'Reconstructed layers · drag to arrange':'Edit masks and layer order, then Apply & Run to continue';
    }
    close() {this.maskDialog?.close();this.maskDialog?.remove();this.dialog?.close();this.dialog?.remove();this.node._layersEditor=null;}
    snapshot() {this.history.push(JSON.stringify(this.state));if(this.history.length>20)this.history.shift();this.future=[];}
    async undo(){if(!this.history.length)return;this.future.push(JSON.stringify(this.state));this.state=JSON.parse(this.history.pop());await this.refresh();}
    async redo(){if(!this.future.length)return;this.history.push(JSON.stringify(this.state));this.state=JSON.parse(this.future.pop());await this.refresh();}
    async refresh(){await this.rebuild();this.list();this.draw();}
    async rebuild(){
        this.assets=new Map();
        for(const layer of this.state.layers){
            const mask=await load(layer.mask),c=canvas(this.state.width,this.state.height),ctx=c.getContext('2d');
            ctx.drawImage(mask,0,0);const pixels=ctx.getImageData(0,0,c.width,c.height),alpha=new Uint8Array(c.width*c.height);
            let bounds=[c.width,c.height,-1,-1];
            for(let p=0;p<alpha.length;p++){alpha[p]=pixels.data[p*4];if(alpha[p]>20){const x=p%c.width,y=Math.floor(p/c.width);if(x<bounds[0])bounds[0]=x;if(y<bounds[1])bounds[1]=y;if(x+1>bounds[2])bounds[2]=x+1;if(y+1>bounds[3])bounds[3]=y+1;}pixels.data[p*4+3]=alpha[p];pixels.data[p*4]=pixels.data[p*4+1]=pixels.data[p*4+2]=255;}
            ctx.putImageData(pixels,0,0);ctx.globalCompositeOperation='source-in';ctx.drawImage(this.rgb.get(layer.id)||this.original,0,0);ctx.globalCompositeOperation='source-over';
            this.assets.set(layer.id,{canvas:c,alpha,bounds});
        }
    }
    positionEditButton(){
        if(!this.floatingEdit)return;
        const layer=this.state.layers.find(l=>this.selected.has(l.id));
        const bounds=layer&&this.assets?.get(layer.id)?.bounds;
        this.floatingEdit.hidden=true;
        if(this.selected.size!==1||!bounds||bounds[2]<0||layer.visible===false||this.state.groups[layer.group]?.visible===false)return;
        const m=layerMatrix(layer,this.state),r=this.view.getBoundingClientRect(),v=this.scroll.getBoundingClientRect();
        const corners=[[bounds[0],bounds[1]],[bounds[2],bounds[1]],[bounds[0],bounds[3]],[bounds[2],bounds[3]]].map(([x,y])=>[r.left+(m[0]*x+m[2]*y+m[4])*r.width/this.view.width,r.top+(m[1]*x+m[3]*y+m[5])*r.height/this.view.height]);
        const left=Math.max(v.left+8,Math.min(...corners.map(p=>p[0]))),right=Math.min(v.right-8,Math.max(...corners.map(p=>p[0])));
        const top=Math.max(v.top+8,Math.min(...corners.map(p=>p[1]))),bottom=Math.min(v.bottom-8,Math.max(...corners.map(p=>p[1])));
        if(right<left||bottom<top)return;
        this.floatingEdit.hidden=false;this.floatingEdit.title=`Edit mask: ${layer.name}`;
        const x=Math.max(v.left+8,Math.min((left+right)/2-this.floatingEdit.offsetWidth/2,v.right-this.floatingEdit.offsetWidth-8));
        this.floatingEdit.style.left=`${x-r.left}px`;this.floatingEdit.style.top=`${top-r.top}px`;
    }
    draw(){
        const ctx=this.view.getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,this.view.width,this.view.height);
        if(this.background)ctx.drawImage(this.background,0,0);
        else {ctx.globalAlpha=.35;ctx.drawImage(this.original,0,0);ctx.globalAlpha=1;}
        for(const layer of this.state.layers){
            if(layer.visible===false||this.state.groups[layer.group]?.visible===false)continue;
            ctx.save();ctx.setTransform(...layerMatrix(layer,this.state));ctx.drawImage(this.assets.get(layer.id).canvas,0,0);
            if(this.selected.has(layer.id)){const b=this.assets.get(layer.id).bounds;if(b[2]>=0){ctx.strokeStyle='#38bdf8';ctx.lineWidth=2/Math.max(.01,Math.hypot(...layerMatrix(layer,this.state).slice(0,2)));ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}}
            ctx.restore();
        }
        this.drawHandles(ctx);this.positionEditButton();
    }
    point(event,target=this.view){const r=target.getBoundingClientRect();return[(event.clientX-r.left)*target.width/r.width,(event.clientY-r.top)*target.height/r.height];}
    hit(x,y){
        for(const l of [...this.state.layers].reverse()){
            if(l.visible===false||this.state.groups[l.group]?.visible===false)continue;
            const [u,v]=invertPoint(layerMatrix(l,this.state),x,y),w=this.state.width,h=this.state.height;
            if(u>=0&&v>=0&&u<w&&v<h&&this.assets.get(l.id).alpha[Math.floor(v)*w+Math.floor(u)]>20)return l;
        }return null;
    }
    targets(){
        const targets=new Map();
        for(const layer of this.state.layers.filter(l=>this.selected.has(l.id)))
            targets.set(layer.group||layer.id,layer.group?this.state.groups[layer.group]:layer);
        return [...targets.values()];
    }
    selectionBounds(){
        const groups=new Set(this.state.layers.filter(l=>this.selected.has(l.id)).map(l=>l.group).filter(Boolean)), points=[];
        for(const layer of this.state.layers){
            if(!this.selected.has(layer.id)&&!groups.has(layer.group))continue;
            if(layer.visible===false||this.state.groups[layer.group]?.visible===false)continue;
            const b=this.assets.get(layer.id)?.bounds;if(!b||b[2]<0)continue;
            const m=layerMatrix(layer,this.state);
            for(const [x,y] of [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])points.push(transformPoint(m,x,y));
        }
        if(!points.length)return null;
        return [Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))];
    }
    handles(){
        const b=this.selectionBounds();return b?[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]:[];
    }
    handleSize(){return 10*this.view.width/(this.view.getBoundingClientRect().width||this.view.width);}
    handleAt(p){const radius=this.handleSize();return this.handles().findIndex(h=>Math.hypot(h[0]-p[0],h[1]-p[1])<=radius);}
    rotationHandle(){
        const b=this.selectionBounds();if(!b)return null;
        const size=this.handleSize(),x=(b[0]+b[2])/2;
        return [x,b[1]>=size*4?b[1]-size*3:Math.min(this.state.height-size,b[3]+size*3)];
    }
    rotationAt(p){const h=this.rotationHandle();return h&&Math.hypot(h[0]-p[0],h[1]-p[1])<=this.handleSize();}
    drawHandles(ctx){
        const handles=this.handles();if(!handles.length)return;
        const size=this.handleSize();ctx.save();ctx.setTransform(1,0,0,1,0,0);
        ctx.strokeStyle='#38bdf8';ctx.fillStyle='#f8fafc';ctx.lineWidth=size/5;
        ctx.strokeRect(handles[0][0],handles[0][1],handles[2][0]-handles[0][0],handles[2][1]-handles[0][1]);
        for(const [x,y]of handles){ctx.fillRect(x-size/2,y-size/2,size,size);ctx.strokeRect(x-size/2,y-size/2,size,size);}
        const rotation=this.rotationHandle();
        if(rotation){const mid=(handles[0][0]+handles[1][0])/2;
            ctx.beginPath();ctx.moveTo(mid,rotation[1]<handles[0][1]?handles[0][1]:handles[2][1]);ctx.lineTo(...rotation);ctx.stroke();
            ctx.beginPath();ctx.arc(...rotation,size*.65,0,Math.PI*2);ctx.fill();ctx.stroke();
        }
        ctx.restore();
    }
    bindViewport(){
        let drag=null;
        this.view.tabIndex=0;this.view.setAttribute('aria-label','Layer canvas. Drag to move; drag a corner to resize; drag the round handle to rotate (Shift snaps to 15 degrees); arrow keys to nudge.');
        this.view.onpointerdown=e=>{
            if(e.button!==0)return;this.view.focus({preventScroll:true});const p=this.point(e),handle=this.handleAt(p);
            if(this.rotationAt(p)){
                const b=this.selectionBounds(),pivot=[(b[0]+b[2])/2,(b[1]+b[3])/2];
                this.snapshot();drag={mode:'rotate',pivot,lastAngle:Math.atan2(p[1]-pivot[1],p[0]-pivot[0]),degrees:0,targets:this.targets().map(t=>({t,start:{...t}}))};
            }else if(handle>=0&&!e.shiftKey){
                const handles=this.handles(),anchor=handles[(handle+2)%4];
                this.snapshot();drag={mode:'resize',p,anchor,corner:handles[handle],targets:this.targets().map(t=>({t,start:{...t}}))};
            }else{
                const l=this.hit(...p);if(!l){this.selected.clear();this.list();return;}
                if(e.shiftKey){if(this.selected.has(l.id))this.selected.delete(l.id);else this.selected.add(l.id);this.list();return;}
                if(!this.selected.has(l.id)){this.selected.clear();this.selected.add(l.id);}
                this.snapshot();drag={mode:'move',p,targets:this.targets().map(t=>({t,start:{...t}}))};
            }
            this.view.setPointerCapture(e.pointerId);this.list();
        };
        this.view.onpointermove=e=>{
            const p=this.point(e);
            if(!drag){const h=this.handleAt(p);this.view.style.cursor=this.rotationAt(p)?'grab':h>=0?(h%2?'nesw-resize':'nwse-resize'):this.hit(...p)?'move':'default';return;}
            if(drag.mode==='rotate'){
                const angle=Math.atan2(p[1]-drag.pivot[1],p[0]-drag.pivot[0]),delta=angle-drag.lastAngle;
                drag.degrees+=Math.atan2(Math.sin(delta),Math.cos(delta))*180/Math.PI;drag.lastAngle=angle;
                const degrees=e.shiftKey?Math.round(drag.degrees/15)*15:drag.degrees;
                for(const {t,start}of drag.targets)Object.assign(t,rotateTransform(start,drag.pivot,degrees,this.state.width,this.state.height));
            }else if(drag.mode==='resize'){
                const v=[drag.corner[0]-drag.anchor[0],drag.corner[1]-drag.anchor[1]],q=[drag.corner[0]+p[0]-drag.p[0]-drag.anchor[0],drag.corner[1]+p[1]-drag.p[1]-drag.anchor[1]];
                const lower=Math.max(...drag.targets.map(({start})=>.01/(start.scale??1))),upper=Math.min(...drag.targets.map(({start})=>100/(start.scale??1)));
                const ratio=Math.max(lower,Math.min(upper,(q[0]*v[0]+q[1]*v[1])/Math.max(1,v[0]*v[0]+v[1]*v[1])));
                for(const {t,start}of drag.targets)Object.assign(t,resizeTransform(start,drag.anchor,ratio,this.state.width,this.state.height));
            }else for(const {t,start}of drag.targets){t.x=(start.x||0)+p[0]-drag.p[0];t.y=(start.y||0)+p[1]-drag.p[1];}
            this.draw();
        };
        this.view.onpointerup=()=>{drag=null;this.list();};
        this.view.onpointercancel=()=>{if(drag){for(const {t,start}of drag.targets)Object.assign(t,start);drag=null;this.list();}};
        this.view.onkeydown=e=>{
            const delta={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
            if(!delta||!this.selected.size)return;e.preventDefault();e.stopPropagation();this.snapshot();
            for(const t of this.targets()){t.x=(t.x||0)+delta[0]*(e.shiftKey?10:1);t.y=(t.y||0)+delta[1]*(e.shiftKey?10:1);}this.list();
        };
        this.view.ondblclick=e=>{const l=this.hit(...this.point(e));if(l)this.maskPopup(l);};
    }
    list(){
        this.draw();
        this.side.replaceChildren();el('h3','Layers · front to back',this.side);
        el('p','Drag an object to move it; drag its corners to resize proportionally. Drag the round handle to rotate (Shift: 15° steps). Arrow keys nudge 1 px (Shift: 10 px). Drag a layer’s grip to change stacking. Shift-click selects several; double-click edits the mask.',this.side,'layers-help');
        const groupSeen=new Set();
        for(const layer of [...this.state.layers].reverse()){
            if(layer.group&&!groupSeen.has(layer.group)){
                groupSeen.add(layer.group);const g=this.state.groups[layer.group],row=el('div',null,this.side,'layers-group');
                button(row,`${g.visible===false?'○':'●'} ${g.name}`,()=>{this.snapshot();g.visible=g.visible===false;this.draw();this.list();});
                button(row,'Select group',()=>{this.selected=new Set(this.state.layers.filter(l=>l.group===layer.group).map(l=>l.id));this.list();});
            }
            const row=el('div',null,this.side,`layers-row ${this.selected.has(layer.id)?'selected':''}`);
            const grip=button(row,'⠿',()=>{this.selected=new Set([layer.id]);this.list();});
            grip.draggable=true;grip.title=`Drag to reorder ${layer.name}`;grip.setAttribute('aria-label',grip.title);
            grip.ondragstart=e=>{this.dragLayerId=layer.id;e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('text/plain',layer.id);};
            grip.ondragend=()=>{this.dragLayerId=null;this.side.querySelectorAll('.drop-front,.drop-back').forEach(r=>r.classList.remove('drop-front','drop-back'));};
            row.ondragover=e=>{if(!this.dragLayerId)return;e.preventDefault();const front=e.clientY<row.getBoundingClientRect().top+row.offsetHeight/2;row.classList.toggle('drop-front',front);row.classList.toggle('drop-back',!front);e.dataTransfer.dropEffect='move';};
            row.ondragleave=()=>row.classList.remove('drop-front','drop-back');
            row.ondrop=e=>{e.preventDefault();const front=e.clientY<row.getBoundingClientRect().top+row.offsetHeight/2,next=moveLayerBlock(this.state.layers,this.dragLayerId,layer.id,front);this.dragLayerId=null;if(next!==this.state.layers){this.snapshot();this.state.layers=next;}this.list();};
            const select=el('input',null,row);select.type='checkbox';select.checked=this.selected.has(layer.id);
            select.onchange=()=>{if(select.checked)this.selected.add(layer.id);else this.selected.delete(layer.id);this.list();};
            button(row,layer.visible===false?'○':'●',()=>{this.snapshot();layer.visible=layer.visible===false;this.list();this.draw();});
            const name=el('input',null,row);name.value=layer.name;name.title='Object description / layer name';name.onchange=()=>{this.snapshot();layer.name=name.value;};
            button(row,'Mask',()=>this.maskPopup(layer));
            button(row,'↑',()=>this.reorder(layer,1)).title='Bring forward';button(row,'↓',()=>this.reorder(layer,-1)).title='Send backward';
        }
        const selected=this.state.layers.find(l=>this.selected.has(l.id));
        if(selected){
            const target=selected.group?this.state.groups[selected.group]:selected;
            el('h3',selected.group?'Group transform':'Layer transform',this.side);
            const actions=el('div',null,this.side,'layers-toolbar');
            button(actions,'Bring to front',()=>this.restack(selected,true));button(actions,'Send to back',()=>this.restack(selected,false));
            for(const [key,title,step] of [['x','X',1],['y','Y',1],['scale','Scale',.05],['angle','Rotation',1]]){
                const label=el('label',title,this.side,'layers-field'),input=el('input',null,label);input.type='number';input.step=step;input.value=target[key]??(key==='scale'?1:0);
                input.onchange=()=>{const n=Number(input.value);if(!Number.isFinite(n)||(key==='scale'&&(n<.01||n>100)))return;this.snapshot();target[key]=n;this.draw();};
            }
        }
    }
    restack(layer,front){
        const target=front?this.state.layers.at(-1):this.state.layers[0],next=moveLayerBlock(this.state.layers,layer.id,target.id,front);
        if(next===this.state.layers)return;this.snapshot();this.state.layers=next;this.list();
    }
    reorder(layer,delta){
        // Group members stay contiguous: reorder the whole group as one block.
        const blocks=[];for(const l of this.state.layers){const prev=blocks.at(-1);if(l.group&&prev?.[0].group===l.group)prev.push(l);else blocks.push([l]);}
        const index=blocks.findIndex(b=>b.includes(layer)),next=index+delta;if(next<0||next>=blocks.length)return;
        this.snapshot();[blocks[index],blocks[next]]=[blocks[next],blocks[index]];this.state.layers=blocks.flat();this.list();this.draw();
    }
    group(){
        const selected=this.state.layers.filter(l=>this.selected.has(l.id));if(selected.length<2)return;
        this.snapshot();for(const l of selected)if(l.group)Object.assign(l,ungroup(l,this.state.groups[l.group],this.state.width,this.state.height));
        const id=crypto.randomUUID();this.state.groups[id]={name:`Group ${Object.keys(this.state.groups).length+1}`,x:0,y:0,scale:1,angle:0,visible:true};
        const index=this.state.layers.indexOf(selected.at(-1));let insertion=this.state.layers.slice(0,index+1).filter(l=>!this.selected.has(l.id)).length;
        this.state.layers=this.state.layers.filter(l=>!this.selected.has(l.id));selected.forEach(l=>l.group=id);this.state.layers.splice(insertion,0,...selected);
        this.cleanGroups();this.list();this.draw();
    }
    ungroup(){this.snapshot();const ids=new Set(this.state.layers.filter(l=>this.selected.has(l.id)).map(l=>l.group).filter(Boolean));for(const l of this.state.layers)if(ids.has(l.group))Object.assign(l,ungroup(l,this.state.groups[l.group],this.state.width,this.state.height));this.cleanGroups();this.list();this.draw();}
    cleanGroups(){for(const id of Object.keys(this.state.groups))if(!this.state.layers.some(l=>l.group===id))delete this.state.groups[id];}
    editSelected(){const layer=this.state.layers.find(l=>this.selected.has(l.id));if(layer)this.maskPopup(layer);else this.status.textContent='Select a layer first.';}
    async commit(refine=null,region=null){
        const state=clone(this.state);if(refine)state.refine=refine;if(region)state.refine_region=region;
        const widget=this.node.widgets.find(w=>w.name==='editor_state');widget.value=JSON.stringify(state);
        this.node.graph?.setDirtyCanvas(true,true);this.node._layersPayload={...this.payload,state};
        this.close();try{await app.queuePrompt(0,1);}catch(error){console.error(error);app.extensionManager?.toast?.add({severity:'error',summary:'Layers',detail:error.message});}
    }
    async maskPopup(layer){
        if(this.maskDialog)return;
        const popup=el('dialog',null,document.body,'layers-dialog layers-mask');this.maskDialog=popup;
        const close=()=>{popup.close();popup.remove();this.maskDialog=null;};popup.oncancel=close;
        const head=el('header',null,popup);el('strong',`Edit mask · ${layer.name}`,head);button(head,'Cancel',close);
        const main=el('div',null,popup,'layers-mask-main'),toolbar=el('div',null,main,'layers-toolbar');
        const tool=el('select',null,toolbar);for(const [v,t]of[['draw','Draw'],['erase','Erase'],['positive','Positive click'],['negative','Negative click']]){const o=el('option',t,tool);o.value=v;}
        const mode=el('select',null,toolbar);for(const[v,t]of[['mask','Object mask'],['completion','Hidden area to reconstruct']]){const o=el('option',t,mode);o.value=v;}
        const sizeLabel=el('label','Brush ',toolbar),size=el('input',null,sizeLabel);size.type='range';size.min=1;size.max=200;size.value=30;
        const softLabel=el('label','Softness ',toolbar),soft=el('input',null,softLabel);soft.type='range';soft.min=0;soft.max=1;soft.step=.05;soft.value=.2;
        const help=el('p','Draw adds pixels. Erase removes them. Point refinement runs SAM3 when requested; apply brush corrections afterwards.',main,'layers-help');
        const scroll=el('div',null,main,'layers-scroll'),stage=el('div',null,scroll,'layers-stage'),view=canvas(this.state.width,this.state.height);stage.append(view);
        const markers=el('div',null,stage,'layers-point-overlay');
        const zoomControl=addZoomControls(toolbar,scroll,view);
        const planes={mask:canvas(view.width,view.height),completion:canvas(view.width,view.height)};
        for(const key of Object.keys(planes)){const ctx=planes[key].getContext('2d');ctx.fillStyle='black';ctx.fillRect(0,0,view.width,view.height);if(layer[key])ctx.drawImage(await load(layer[key]),0,0);}
        let points={positive:clone(layer.positive||[]),negative:clone(layer.negative||[])},history=[],future=[],drawing=false,last=null;
        const snapshot=()=>JSON.stringify({mask:planes.mask.toDataURL(),completion:planes.completion.toDataURL(),points});
        const restore=async raw=>{const s=JSON.parse(raw);points=s.points;for(const key of Object.keys(planes))planes[key].getContext('2d').drawImage(await load(s[key]),0,0);draw();};
        const remember=()=>{history.push(snapshot());if(history.length>20)history.shift();future=[];};
        const draw=()=>{
            const ctx=view.getContext('2d');ctx.clearRect(0,0,view.width,view.height);ctx.drawImage(this.original,0,0);
            const overlay=canvas(view.width,view.height),o=overlay.getContext('2d');o.drawImage(planes[mode.value],0,0);
            const pixels=o.getImageData(0,0,view.width,view.height);for(let p=0;p<pixels.data.length;p+=4){const a=pixels.data[p];pixels.data[p]=mode.value==='mask'?40:255;pixels.data[p+1]=mode.value==='mask'?180:140;pixels.data[p+2]=mode.value==='mask'?255:30;pixels.data[p+3]=a*.5;}o.putImageData(pixels,0,0);ctx.drawImage(overlay,0,0);
            markers.replaceChildren();
            for(const key of ['positive','negative'])for(const p of points[key]){
                const marker=el('span',null,markers,`layers-point layers-point-${key}`);
                marker.style.left=`${p.x/view.width*100}%`;marker.style.top=`${p.y/view.height*100}%`;
            }
        };
        const stamp=(x,y)=>{const ctx=planes[mode.value].getContext('2d'),r=Number(size.value)/2,color=tool.value==='erase'?'0,0,0':'255,255,255',s=Number(soft.value);ctx.fillStyle=`rgb(${color})`;if(s>0){const g=ctx.createRadialGradient(x,y,r*(1-s),x,y,r);g.addColorStop(0,`rgba(${color},1)`);g.addColorStop(1,`rgba(${color},0)`);ctx.fillStyle=g;}ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();};
        view.onpointerdown=e=>{
            if(e.button!==0)return;remember();const [x,y]=this.point(e,view);if(tool.value==='positive'||tool.value==='negative'){points[tool.value].push({x:Math.round(x),y:Math.round(y)});draw();return;}
            drawing=true;last=[x,y];view.setPointerCapture(e.pointerId);stamp(x,y);draw();
        };
        view.onpointermove=e=>{if(!drawing)return;const p=this.point(e,view),dx=p[0]-last[0],dy=p[1]-last[1],n=Math.max(1,Math.ceil(Math.hypot(dx,dy)/Math.max(1,Number(size.value)/8)));for(let i=1;i<=n;i++)stamp(last[0]+dx*i/n,last[1]+dy*i/n);last=p;draw();};
        view.onpointerup=()=>drawing=false;view.onpointercancel=()=>drawing=false;
        mode.onchange=()=>{help.textContent=mode.value==='completion'?'Paint where this object should continue behind other objects. Black leaves the area unchanged.':'Blue shows included object pixels. Clicks refine SAM3; brushes edit the mask directly.';draw();};
        button(toolbar,'Undo',async()=>{if(history.length){future.push(snapshot());await restore(history.pop());}});
        button(toolbar,'Redo',async()=>{if(future.length){history.push(snapshot());await restore(future.pop());}});
        button(toolbar,'Clear clicks',()=>{remember();points={positive:[],negative:[]};draw();});
        button(toolbar,'Automatic hidden area',()=>{remember();planes.completion.getContext('2d').fillStyle='black';planes.completion.getContext('2d').fillRect(0,0,view.width,view.height);draw();});
        const apply=()=>{this.snapshot();layer.mask=planes.mask.toDataURL();layer.positive=points.positive;layer.negative=points.negative;
            const raw=planes.completion.getContext('2d').getImageData(0,0,view.width,view.height).data;let nonzero=false;for(let i=0;i<raw.length;i+=4)if(raw[i]){nonzero=true;break;}
            if(nonzero)layer.completion=planes.completion.toDataURL();else delete layer.completion;};
        button(head,'Apply',async()=>{apply();close();await this.refresh();});
        button(toolbar,'Shrink mask 1 px',()=>{
            remember();const ctx=planes[mode.value].getContext('2d'),image=ctx.getImageData(0,0,view.width,view.height),src=image.data.slice();
            for(let y=0;y<view.height;y++)for(let x=0;x<view.width;x++){
                let value=255;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
                    const xx=Math.max(0,Math.min(view.width-1,x+dx)),yy=Math.max(0,Math.min(view.height-1,y+dy));value=Math.min(value,src[(yy*view.width+xx)*4]);
                }const i=(y*view.width+x)*4;image.data[i]=image.data[i+1]=image.data[i+2]=value;
            }ctx.putImageData(image,0,0);draw();
        });
        button(toolbar,'Soften edge 0.7 px',()=>{
            remember();const target=planes[mode.value],copy=canvas(view.width,view.height);copy.getContext('2d').drawImage(target,0,0);
            const ctx=target.getContext('2d');ctx.fillStyle='black';ctx.fillRect(0,0,view.width,view.height);ctx.filter='blur(0.7px)';ctx.drawImage(copy,0,0);ctx.filter='none';draw();
        });
        button(head,'Refine visible area',async()=>{
            const r=view.getBoundingClientRect(),s=scroll.getBoundingClientRect();
            const region={x:Math.max(0,Math.floor((s.left-r.left)*view.width/r.width)),y:Math.max(0,Math.floor((s.top-r.top)*view.height/r.height)),
                right:Math.min(view.width,Math.ceil((s.right-r.left)*view.width/r.width)),bottom:Math.min(view.height,Math.ceil((s.bottom-r.top)*view.height/r.height))};
            if(!points.positive.some(p=>p.x>=region.x&&p.x<region.right&&p.y>=region.y&&p.y<region.bottom)){
                help.textContent='Add a positive click on the object inside this view and negative clicks in the unwanted gaps. Then refine the visible area.';return;
            }apply();close();await this.commit(layer.id,region);
        });
        button(head,'Refine clicks with SAM3',async()=>{if(!points.positive.length){help.textContent='Add at least one positive point inside the object.';return;}apply();close();await this.commit(layer.id);});
        popup.showModal();zoomControl.fit();draw();
    }
}

app.registerExtension({
    name:'ComfyUI.Layers.Editor',
    async beforeRegisterNodeDef(nodeType,nodeData){
        if(nodeData.name!=='LayersEditor')return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(){
            const result=created?.apply(this,arguments);
            hideEditorState(this);
            const status = document.createElement('div');
            status.className = 'layers-node-status';
            status.textContent = editorStatus(null);
            this._layersStatus = status;
            const panel=document.createElement('div');panel.className='layers-node-panel';
            const thumbnail=document.createElement('img');thumbnail.className='layers-node-thumbnail';thumbnail.alt='Source image for layer editing';thumbnail.hidden=true;
            this._layersThumbnail=thumbnail;panel.append(thumbnail,status);
            this.addDOMWidget('layers_status', 'layers_status', panel, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 260, getMaxHeight: () => 340,
            });
            this.addWidget('button','Open layer editor',null,async()=>{
                if(this._layersEditor)return;
                if(!this._layersPayload){app.extensionManager?.toast?.add({severity:'info',summary:'Layers',detail:'Run the workflow once to load images and masks.'});return;}
                const editor=new Editor(this,this._layersPayload);this._layersEditor=editor;
                try{await editor.open();}catch(error){editor.close();console.error(error);app.extensionManager?.toast?.add({severity:'error',summary:'Layers',detail:error.message});}
            });
            this.addWidget('button','Reset editing state',null,()=>{const w=this.widgets.find(w=>w.name==='editor_state');w.value='';this._layersPayload=null;this._layersStatus.textContent=editorStatus(null);this._layersThumbnail.hidden=true;this._layersThumbnail.removeAttribute('src');this.graph?.setDirtyCanvas(true,true);});
            return result;
        };
        const configured=nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure=function(){
            const result=configured?.apply(this,arguments);
            hideEditorState(this);
            return result;
        };
        const executed=nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted=function(message){executed?.apply(this,arguments);const payload=message?.layers_project?.[0];if(!payload)return;this._layersPayload=payload;
            hideEditorState(this);
            this._layersStatus.textContent=editorStatus(payload);
            this._layersThumbnail.src=payload.image;this._layersThumbnail.hidden=false;
            // Save refined masks in the workflow, but do not auto-queue: review first.
            const w=this.widgets.find(w=>w.name==='editor_state');w.value=JSON.stringify(payload.state);
            this.graph?.setDirtyCanvas(true,true);
        };
        const removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved=function(){this._layersEditor?.close();return removed?.apply(this,arguments);};
    }
});
