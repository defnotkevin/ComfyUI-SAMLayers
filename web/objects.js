import { app } from '../../scripts/app.js';
const el=(tag,parent,text)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;if(parent)parent.append(e);return e;};
const button=(p,t,fn)=>{const b=el('button',p,t);b.type='button';b.onclick=fn;return b;};
function hide(node){const w=node.widgets?.find(w=>w.name==='object_state');if(!w)return;w.hidden=true;w.computeSize=()=>[0,-4];w.draw=()=>{};if(w.inputEl)w.inputEl.style.display='none';if(w.element)w.element.style.display='none';}
function review(node){
    if(node._objectsDialog)return;
    if(!node._catalog){app.extensionManager?.toast?.add({severity:'info',summary:'Objects',detail:'Run discovery once first.'});return;}
    const state=JSON.parse(JSON.stringify(node._catalog.state));
    const dialog=el('dialog',document.body);dialog.className='layers-dialog objects-dialog';node._objectsDialog=dialog;
    const close=()=>{dialog.close();dialog.remove();node._objectsDialog=null;node._receiveObjectPreview=null;};dialog.oncancel=close;
    const head=el('header',dialog);el('strong',head,'Review discovered objects');
    const status=el('span',head,'Add object: include/exclude clicks → Preview SAM mask → Confirm object.');status.className='layers-status';
    button(head,'Cancel',close);
    button(head,'Apply & Run',async()=>{
        if(draft){status.textContent='Confirm or cancel the new object first.';return;}
        for(const o of state.objects){const [x,y,r,b]=o.bbox;if(!o.name.trim()||!o.prompt.trim()||!o.bbox.every(Number.isFinite)||x<0||y<0||r>1000||b>1000||x>=r||y>=b){status.textContent='Each object needs a name, description and a valid box inside 0–1000.';return;}}
        if(!state.objects.some(o=>o.enabled)){status.textContent='Select at least one object.';return;}
        node.widgets.find(w=>w.name==='object_state').value=JSON.stringify(state);
        node._catalog.state=state;node.graph?.setDirtyCanvas(true,true);close();
        try{await app.queuePrompt(0,1);}catch(error){app.extensionManager?.toast?.add({severity:'error',summary:'Objects',detail:error.message});}
    });
    const body=el('div',dialog);body.className='objects-body';
    const preview=el('canvas',body);preview.className='objects-preview';
    const pane=el('section',body),actions=el('div',pane),rows=el('div',pane);
    let selected=0,draft=null;const editor=el('div',pane);editor.hidden=true;const image=new Image();image.onload=()=>{preview.width=image.naturalWidth;preview.height=image.naturalHeight;draw();};image.src=node._catalog.image;
    function draw(){if(!image.complete||!image.naturalWidth)return;const c=preview.getContext('2d');c.clearRect(0,0,preview.width,preview.height);c.drawImage(image,0,0);
        const scale=preview.width/(preview.getBoundingClientRect().width||preview.width);
        if(draft){
            if(draft.overlay)c.drawImage(draft.overlay,0,0);
            for(const [key,color] of [['positive','#22c55e'],['negative','#ef4444']])for(const p of draft[key]){
                c.beginPath();c.arc(p.x,p.y,5*scale,0,Math.PI*2);c.fillStyle=color;c.fill();c.strokeStyle='white';c.lineWidth=scale;c.stroke();
            }
            return;
        }
        const item=state.objects[selected];if(!item)return;const [x,y,r,b]=item.bbox;c.strokeStyle='#00cfff';c.lineWidth=3*scale; c.strokeRect(x/1000*preview.width,y/1000*preview.height,(r-x)/1000*preview.width,(b-y)/1000*preview.height);}
    function invalidate(){if(!draft)return;draft.id=null;draft.mask=null;draft.overlay=null;draw();}
    preview.onpointerdown=e=>{if(!draft||e.button!==0)return;const r=preview.getBoundingClientRect();
        draft[draft.mode].push({x:Math.max(0,Math.min(preview.width-1,(e.clientX-r.left)*preview.width/r.width)),y:Math.max(0,Math.min(preview.height-1,(e.clientY-r.top)*preview.height/r.height))});invalidate();};
    node._receiveObjectPreview=async result=>{
        if(!draft||draft.id!==result.id)return;
        const target=draft,maskImage=new Image();maskImage.src=result.mask;await maskImage.decode();
        if(draft!==target||draft.id!==result.id)return;
        const overlay=document.createElement('canvas');overlay.width=preview.width;overlay.height=preview.height;
        const ctx=overlay.getContext('2d');ctx.drawImage(maskImage,0,0);const pixels=ctx.getImageData(0,0,overlay.width,overlay.height);
        for(let i=0;i<pixels.data.length;i+=4){const alpha=pixels.data[i];pixels.data[i]=0;pixels.data[i+1]=210;pixels.data[i+2]=255;pixels.data[i+3]=Math.round(alpha*.5);}
        ctx.putImageData(pixels,0,0);draft.overlay=overlay;draft.mask=result.mask;draft.bbox=result.bbox;
        status.textContent='Inspect the blue highlight. Add more include/exclude clicks and preview again, or Confirm object.';draw();
    };
    function endDraft(){draft=null;editor.hidden=true;rows.hidden=false;draw();}
    function addObject(){
        if(state.objects.length>=64||draft)return;
        draft={positive:[],negative:[],mode:'positive',name:'New object'};editor.replaceChildren();editor.hidden=false;rows.hidden=true;
        el('p',editor,'Click inside the object to include it. Use exclude clicks for unwanted areas. Each preview runs SAM through the queue.');
        const name=el('input',editor);name.value=draft.name;name.placeholder='Object name';name.oninput=()=>draft.name=name.value;
        const mode=el('select',editor);for(const [value,label] of [['positive','Include (+)'],['negative','Exclude (−)']]){const opt=el('option',mode,label);opt.value=value;}mode.onchange=()=>draft.mode=mode.value;
        button(editor,'Undo click',()=>{draft[draft.mode].pop();invalidate();});
        button(editor,'Clear clicks',()=>{draft.positive=[];draft.negative=[];invalidate();});
        button(editor,'Preview SAM mask',async()=>{
            if(!draft.positive.length){status.textContent='Add at least one include click.';return;}
            if(!node.inputs?.find(i=>i.name==='sam_model')?.link){status.textContent='Connect the SAM3 checkpoint MODEL output to this Review Objects node’s sam_model input.';return;}
            invalidate();draft.id=globalThis.crypto.randomUUID();
            node.widgets.find(w=>w.name==='object_state').value=JSON.stringify({...state,detect_object:{id:draft.id,positive:draft.positive,negative:draft.negative}});
            status.textContent='SAM preview queued. If execution fails, adjust clicks and retry; see ComfyUI’s error details.';
            try{await app.queuePrompt(0,1);}catch(error){status.textContent=error.message;}
        });
        button(editor,'Confirm object',()=>{
            if(!draft.mask){status.textContent='Preview the current clicks before confirming.';return;}
            if(!draft.name.trim()){status.textContent='Enter an object name.';return;}
            state.objects.push({name:draft.name.trim(),prompt:draft.name.trim(),bbox:draft.bbox,kind:'object',enabled:true,confirmed_mask:draft.mask});selected=state.objects.length-1;
            endDraft();list();status.textContent='Object added with its confirmed SAM mask.';
        });
        button(editor,'Cancel new object',endDraft);draw();
    }
    function list(){rows.replaceChildren();state.objects.forEach((o,i)=>{
        const row=el('div',rows);row.className='objects-row';row.onclick=()=>{selected=i;draw();};
        const checkbox=el('input',row);checkbox.type='checkbox';checkbox.checked=o.enabled;checkbox.title='Include this object';checkbox.onchange=()=>o.enabled=checkbox.checked;
        for(const key of ['name','prompt']){const label=el('label',row,key==='name'?'Layer name':'SAM description'),input=el('input',label);input.value=o[key];input.oninput=()=>o[key]=input.value;}
        const kind=el('select',row);for(const k of ['object','background']){const opt=el('option',kind,k);opt.value=k;}kind.value=o.kind;kind.onchange=()=>o.kind=kind.value;
        if(o.confirmed_mask)el('span',row,'Confirmed click mask');
        const details=el('details',row);el('summary',details,'Advanced box coordinates');
        const coords=el('div',details);coords.className='objects-box';
        ['Left','Top','Right','Bottom'].forEach((text,k)=>{const label=el('label',coords,text),input=el('input',label);input.type='number';input.min=0;input.max=1000;input.value=o.bbox[k];input.oninput=()=>{o.bbox[k]=Number(input.value);delete o.confirmed_mask;selected=i;draw();};});
        button(row,'Remove',()=>{state.objects.splice(i,1);selected=Math.min(selected,state.objects.length-1);list();draw();});
    });}
    button(actions,'Add object',addObject);
    button(actions,'Select all scene layers',()=>{state.objects.forEach(o=>o.enabled=true);list();});
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
        if(p.object_preview)this._receiveObjectPreview?.(p.object_preview);
    };
    Type.prototype.onRemoved=function(){this._objectsDialog?.close();this._objectsDialog?.remove();removed?.apply(this,arguments);};
}});
