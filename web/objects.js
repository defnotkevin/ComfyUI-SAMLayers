import { app } from '../../scripts/app.js';
const el=(tag,parent,text)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;if(parent)parent.append(e);return e;};
const button=(p,t,fn)=>{const b=el('button',p,t);b.type='button';b.onclick=fn;return b;};
function hide(node){const w=node.widgets?.find(w=>w.name==='object_state');if(!w)return;w.hidden=true;w.computeSize=()=>[0,-4];w.draw=()=>{};if(w.inputEl)w.inputEl.style.display='none';if(w.element)w.element.style.display='none';}
function review(node){
    if(node._objectsDialog)return;
    if(!node._catalog){app.extensionManager?.toast?.add({severity:'info',summary:'Objects',detail:'Run discovery once first.'});return;}
    const state=JSON.parse(JSON.stringify(node._catalog.state));
    const dialog=el('dialog',document.body);dialog.className='layers-dialog objects-dialog';node._objectsDialog=dialog;
    const close=()=>{dialog.close();dialog.remove();node._objectsDialog=null;};dialog.oncancel=close;
    const head=el('header',dialog);el('strong',head,'Review discovered objects');
    const status=el('span',head,'Boxes use coordinates from 0 to 1000. Select a row to inspect its box.');status.className='layers-status';
    button(head,'Cancel',close);
    button(head,'Apply & Run',async()=>{
        for(const o of state.objects){const [x,y,r,b]=o.bbox;if(!o.name.trim()||!o.prompt.trim()||!o.bbox.every(Number.isFinite)||x<0||y<0||r>1000||b>1000||x>=r||y>=b){status.textContent='Each object needs a name, description and a valid box inside 0–1000.';return;}}
        if(!state.objects.some(o=>o.enabled)){status.textContent='Select at least one object.';return;}
        node.widgets.find(w=>w.name==='object_state').value=JSON.stringify(state);
        node._catalog.state=state;node.graph?.setDirtyCanvas(true,true);close();
        try{await app.queuePrompt(0,1);}catch(error){app.extensionManager?.toast?.add({severity:'error',summary:'Objects',detail:error.message});}
    });
    const body=el('div',dialog);body.className='objects-body';
    const preview=el('canvas',body);preview.className='objects-preview';
    const pane=el('section',body),actions=el('div',pane),rows=el('div',pane);
    let selected=0;const image=new Image();image.onload=()=>{preview.width=image.naturalWidth;preview.height=image.naturalHeight;draw();};image.src=node._catalog.image;
    function draw(){if(!image.complete||!image.naturalWidth)return;const c=preview.getContext('2d');c.clearRect(0,0,preview.width,preview.height);c.drawImage(image,0,0);
        const item=state.objects[selected];if(!item)return;const [x,y,r,b]=item.bbox;c.strokeStyle='#00cfff';c.lineWidth=3; c.strokeRect(x/1000*preview.width,y/1000*preview.height,(r-x)/1000*preview.width,(b-y)/1000*preview.height);}
    function list(){rows.replaceChildren();state.objects.forEach((o,i)=>{
        const row=el('div',rows);row.className='objects-row';row.onclick=()=>{selected=i;draw();};
        const checkbox=el('input',row);checkbox.type='checkbox';checkbox.checked=o.enabled;checkbox.title='Include this object';checkbox.onchange=()=>o.enabled=checkbox.checked;
        for(const key of ['name','prompt']){const label=el('label',row,key==='name'?'Layer name':'SAM description'),input=el('input',label);input.value=o[key];input.oninput=()=>o[key]=input.value;}
        const kind=el('select',row);for(const k of ['object','background']){const opt=el('option',kind,k);opt.value=k;}kind.value=o.kind;kind.onchange=()=>o.kind=kind.value;
        const coords=el('div',row);coords.className='objects-box';
        ['Left','Top','Right','Bottom'].forEach((text,k)=>{const label=el('label',coords,text),input=el('input',label);input.type='number';input.min=0;input.max=1000;input.value=o.bbox[k];input.oninput=()=>{o.bbox[k]=Number(input.value);selected=i;draw();};});
        button(row,'Remove',()=>{state.objects.splice(i,1);selected=Math.min(selected,state.objects.length-1);list();draw();});
    });}
    button(actions,'Add object',()=>{if(state.objects.length>=64)return;state.objects.push({name:'New object',prompt:'object',bbox:[0,0,1000,1000],kind:'object',enabled:true});selected=state.objects.length-1;list();draw();});
    button(actions,'Select objects only',()=>{state.objects.forEach(o=>o.enabled=o.kind==='object');list();});
    list();dialog.showModal();
}
app.registerExtension({name:'ComfyUI.Layers.ObjectReview',async beforeRegisterNodeDef(Type,data){
    if(data.name!=='LayersReviewObjects')return;
    const created=Type.prototype.onNodeCreated,executed=Type.prototype.onExecuted,configured=Type.prototype.onConfigure,removed=Type.prototype.onRemoved;
    Type.prototype.onNodeCreated=function(){created?.apply(this,arguments);hide(this);
        const panel=el('div');panel.className='layers-node-panel';this._objectImage=el('img',panel);this._objectImage.className='layers-node-thumbnail';this._objectImage.hidden=true;
        this._objectStatus=el('div',panel,'Run once to discover objects, then review the list.');this._objectStatus.className='layers-node-status';
        this.addDOMWidget('object_summary','object_summary',panel,{serialize:false,getMinHeight:()=>240,getMaxHeight:()=>320});
        this.addWidget('button','Review object list',null,()=>review(this));
    };
    Type.prototype.onConfigure=function(){configured?.apply(this,arguments);hide(this);};
    Type.prototype.onExecuted=function(message){executed?.apply(this,arguments);const p=message?.object_catalog?.[0];if(!p)return;this._catalog=p;
        this.widgets.find(w=>w.name==='object_state').value=JSON.stringify(p.state);this._objectImage.src=p.image;this._objectImage.hidden=false;
        this._objectStatus.textContent=`${p.state.objects.length} objects discovered. ${p.review_required?'Paused: review the list, then Apply & Run.':'List applied.'}`;hide(this);
    };
    Type.prototype.onRemoved=function(){this._objectsDialog?.close();this._objectsDialog?.remove();removed?.apply(this,arguments);};
}});
