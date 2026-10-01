import { app } from '../../scripts/app.js';
import { layerMatrix, matrix, invertPoint, ungroup } from './math.js';

const clone = x => JSON.parse(JSON.stringify(x));
const el = (tag, text, parent, cls) => {
    const e=document.createElement(tag); if(text != null) e.textContent=text;
    if(cls)e.className=cls; if(parent)parent.append(e); return e;
};
const button = (parent,label,action) => { const b=el('button',label,parent);b.type='button';b.onclick=action;return b; };
const canvas = (w,h) => {const c=document.createElement('canvas');c.width=w;c.height=h;return c;};
const load = src => new Promise((ok,fail)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=()=>fail(new Error('Cannot load layer image'));i.src=src;});
const css=document.createElement('link');css.rel='stylesheet';css.href=new URL('./layers.css',import.meta.url).href;document.head.append(css);

class Editor {
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
        button(toolbar,'Edit mask',()=>this.editSelected());
        this.zoom=el('input',null,toolbar);this.zoom.type='range';this.zoom.min=10;this.zoom.max=200;this.zoom.value=60;
        this.zoom.title='Zoom';this.zoom.oninput=()=>this.resize();
        this.scroll=el('div',null,left,'layers-scroll');
        this.view=canvas(this.state.width,this.state.height);this.scroll.append(this.view);
        this.side=el('aside',null,this.body);
        this.dialog.addEventListener('cancel',()=>this.close());this.dialog.showModal();
        this.original=await load(this.payload.image);
        this.background=this.payload.background ? await load(this.payload.background):null;
        this.rgb=new Map();
        for(let i=0;i<this.state.layers.length;i++) this.rgb.set(this.state.layers[i].id,this.payload.rgbs ? await load(this.payload.rgbs[i]):this.original);
        await this.rebuild();this.resize();this.bindViewport();this.list();this.draw();
        this.status.textContent=this.payload.reconstructed?'Reconstructed layers · drag to arrange':'Edit masks and back-to-front order, then Apply & Run to reconstruct';
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
            for(let p=0;p<alpha.length;p++){alpha[p]=pixels.data[p*4];pixels.data[p*4+3]=alpha[p];pixels.data[p*4]=pixels.data[p*4+1]=pixels.data[p*4+2]=255;}
            ctx.putImageData(pixels,0,0);ctx.globalCompositeOperation='source-in';ctx.drawImage(this.rgb.get(layer.id)||this.original,0,0);ctx.globalCompositeOperation='source-over';
            this.assets.set(layer.id,{canvas:c,alpha});
        }
    }
    resize(){this.view.style.width=`${this.state.width*Number(this.zoom.value)/100}px`;this.view.style.height='auto';}
    draw(){
        const ctx=this.view.getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,this.view.width,this.view.height);
        if(this.background)ctx.drawImage(this.background,0,0);
        else {ctx.globalAlpha=.35;ctx.drawImage(this.original,0,0);ctx.globalAlpha=1;}
        for(const layer of this.state.layers){
            if(layer.visible===false||this.state.groups[layer.group]?.visible===false)continue;
            ctx.save();ctx.setTransform(...layerMatrix(layer,this.state));ctx.drawImage(this.assets.get(layer.id).canvas,0,0);ctx.restore();
        }
    }
    point(event,target=this.view){const r=target.getBoundingClientRect();return[(event.clientX-r.left)*target.width/r.width,(event.clientY-r.top)*target.height/r.height];}
    hit(x,y){
        for(const l of [...this.state.layers].reverse()){
            if(l.visible===false||this.state.groups[l.group]?.visible===false)continue;
            const [u,v]=invertPoint(layerMatrix(l,this.state),x,y),w=this.state.width,h=this.state.height;
            if(u>=0&&v>=0&&u<w&&v<h&&this.assets.get(l.id).alpha[Math.floor(v)*w+Math.floor(u)]>20)return l;
        }return null;
    }
    bindViewport(){
        let drag=null;
        this.view.onpointerdown=e=>{
            const p=this.point(e),l=this.hit(...p);if(!l){this.selected.clear();this.list();return;}
            if(e.shiftKey){if(this.selected.has(l.id))this.selected.delete(l.id);else this.selected.add(l.id);this.list();return;}
            if(!this.selected.has(l.id)){this.selected.clear();this.selected.add(l.id);}
            const targets=new Map();for(const a of this.state.layers.filter(x=>this.selected.has(x.id)))targets.set(a.group||a.id,a.group?this.state.groups[a.group]:a);
            this.snapshot();drag={p,targets:[...targets.values()].map(t=>({t,x:t.x||0,y:t.y||0}))};
            this.view.setPointerCapture(e.pointerId);this.list();
        };
        this.view.onpointermove=e=>{if(!drag)return;const p=this.point(e);for(const {t,x,y} of drag.targets){t.x=x+p[0]-drag.p[0];t.y=y+p[1]-drag.p[1];}this.draw();};
        this.view.onpointerup=()=>{drag=null;this.list();};this.view.onpointercancel=()=>{drag=null;};
        this.view.ondblclick=e=>{const l=this.hit(...this.point(e));if(l)this.maskPopup(l);};
    }
    list(){
        this.side.replaceChildren();el('h3','Layers · front to back',this.side);
        el('p','Shift-click to select several. Double-click an object to edit its mask.',this.side,'layers-help');
        const groupSeen=new Set();
        for(const layer of [...this.state.layers].reverse()){
            if(layer.group&&!groupSeen.has(layer.group)){
                groupSeen.add(layer.group);const g=this.state.groups[layer.group],row=el('div',null,this.side,'layers-group');
                button(row,`${g.visible===false?'○':'●'} ${g.name}`,()=>{this.snapshot();g.visible=g.visible===false;this.draw();this.list();});
                button(row,'Select group',()=>{this.selected=new Set(this.state.layers.filter(l=>l.group===layer.group).map(l=>l.id));this.list();});
            }
            const row=el('div',null,this.side,`layers-row ${this.selected.has(layer.id)?'selected':''}`);
            const select=el('input',null,row);select.type='checkbox';select.checked=this.selected.has(layer.id);
            select.onchange=()=>{if(select.checked)this.selected.add(layer.id);else this.selected.delete(layer.id);this.list();};
            button(row,layer.visible===false?'○':'●',()=>{this.snapshot();layer.visible=layer.visible===false;this.list();this.draw();});
            const name=el('input',null,row);name.value=layer.name;name.title='Object description / layer name';name.onchange=()=>{this.snapshot();layer.name=name.value;};
            button(row,'Mask',()=>this.maskPopup(layer));
            button(row,'↑',()=>this.reorder(layer,1));button(row,'↓',()=>this.reorder(layer,-1));
        }
        const selected=this.state.layers.find(l=>this.selected.has(l.id));
        if(selected){
            const target=selected.group?this.state.groups[selected.group]:selected;
            el('h3',selected.group?'Group transform':'Layer transform',this.side);
            for(const [key,title,step] of [['x','X',1],['y','Y',1],['scale','Scale',.05],['angle','Rotation',1]]){
                const label=el('label',title,this.side,'layers-field'),input=el('input',null,label);input.type='number';input.step=step;input.value=target[key]??(key==='scale'?1:0);
                input.onchange=()=>{const n=Number(input.value);if(!Number.isFinite(n)||(key==='scale'&&(n<.01||n>100)))return;this.snapshot();target[key]=n;this.draw();};
            }
        }
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
    async commit(refine=null){
        const state=clone(this.state);if(refine)state.refine=refine;
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
        const zoom=el('input',null,toolbar);zoom.type='range';zoom.min=10;zoom.max=200;zoom.value=65;zoom.title='Zoom';
        const help=el('p','Draw adds pixels. Erase removes them. Point refinement runs SAM3 when requested; apply brush corrections afterwards.',main,'layers-help');
        const scroll=el('div',null,main,'layers-scroll'),view=canvas(this.state.width,this.state.height);scroll.append(view);
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
            for(const key of ['positive','negative'])for(const p of points[key]){ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fillStyle=key==='positive'?'#00ff88':'#ff4466';ctx.fill();ctx.strokeStyle='white';ctx.stroke();}
        };
        const stamp=(x,y)=>{const ctx=planes[mode.value].getContext('2d'),r=Number(size.value)/2,color=tool.value==='erase'?'0,0,0':'255,255,255',s=Number(soft.value);ctx.fillStyle=`rgb(${color})`;if(s>0){const g=ctx.createRadialGradient(x,y,r*(1-s),x,y,r);g.addColorStop(0,`rgba(${color},1)`);g.addColorStop(1,`rgba(${color},0)`);ctx.fillStyle=g;}ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();};
        view.onpointerdown=e=>{
            remember();const [x,y]=this.point(e,view);if(tool.value==='positive'||tool.value==='negative'){points[tool.value].push({x:Math.round(x),y:Math.round(y)});draw();return;}
            drawing=true;last=[x,y];view.setPointerCapture(e.pointerId);stamp(x,y);draw();
        };
        view.onpointermove=e=>{if(!drawing)return;const p=this.point(e,view),dx=p[0]-last[0],dy=p[1]-last[1],n=Math.max(1,Math.ceil(Math.hypot(dx,dy)/Math.max(1,Number(size.value)/8)));for(let i=1;i<=n;i++)stamp(last[0]+dx*i/n,last[1]+dy*i/n);last=p;draw();};
        view.onpointerup=()=>drawing=false;view.onpointercancel=()=>drawing=false;
        mode.onchange=()=>{help.textContent=mode.value==='completion'?'Paint where this object should continue behind other objects. Black leaves the area unchanged.':'Blue shows included object pixels. Clicks refine SAM3; brushes edit the mask directly.';draw();};
        zoom.oninput=()=>{view.style.width=`${view.width*Number(zoom.value)/100}px`;view.style.height='auto';};zoom.oninput();
        button(toolbar,'Undo',async()=>{if(history.length){future.push(snapshot());await restore(history.pop());}});
        button(toolbar,'Redo',async()=>{if(future.length){history.push(snapshot());await restore(future.pop());}});
        button(toolbar,'Clear clicks',()=>{remember();points={positive:[],negative:[]};draw();});
        button(toolbar,'Automatic hidden area',()=>{remember();planes.completion.getContext('2d').fillStyle='black';planes.completion.getContext('2d').fillRect(0,0,view.width,view.height);draw();});
        const apply=()=>{this.snapshot();layer.mask=planes.mask.toDataURL();layer.positive=points.positive;layer.negative=points.negative;
            const raw=planes.completion.getContext('2d').getImageData(0,0,view.width,view.height).data;let nonzero=false;for(let i=0;i<raw.length;i+=4)if(raw[i]){nonzero=true;break;}
            if(nonzero)layer.completion=planes.completion.toDataURL();else delete layer.completion;};
        button(head,'Apply',async()=>{apply();close();await this.refresh();});
        button(head,'Refine clicks with SAM3',async()=>{if(!points.positive.length){help.textContent='Add at least one positive point inside the object.';return;}apply();close();await this.commit(layer.id);});
        popup.showModal();draw();
    }
}

app.registerExtension({
    name:'ComfyUI.Layers.Editor',
    async beforeRegisterNodeDef(nodeType,nodeData){
        if(nodeData.name!=='LayersEditor')return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(){
            const result=created?.apply(this,arguments);
            this.addWidget('button','Open layer editor',null,async()=>{
                if(this._layersEditor)return;
                if(!this._layersPayload){app.extensionManager?.toast?.add({severity:'info',summary:'Layers',detail:'Run the workflow once to load images and masks.'});return;}
                const editor=new Editor(this,this._layersPayload);this._layersEditor=editor;
                try{await editor.open();}catch(error){editor.close();console.error(error);app.extensionManager?.toast?.add({severity:'error',summary:'Layers',detail:error.message});}
            });
            this.addWidget('button','Reset editing state',null,()=>{const w=this.widgets.find(w=>w.name==='editor_state');w.value='';this._layersPayload=null;this.graph?.setDirtyCanvas(true,true);});
            return result;
        };
        const executed=nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted=function(message){executed?.apply(this,arguments);const payload=message?.layers_project?.[0];if(!payload)return;this._layersPayload=payload;
            // Save refined masks in the workflow, but do not auto-queue: review first.
            const w=this.widgets.find(w=>w.name==='editor_state');w.value=JSON.stringify(payload.state);
            this.graph?.setDirtyCanvas(true,true);
        };
        const removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved=function(){this._layersEditor?.close();return removed?.apply(this,arguments);};
    }
});
