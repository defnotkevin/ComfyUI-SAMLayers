export const MIN_ZOOM = 10;
export const MAX_ZOOM = 3200;
export function clampZoom(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, n)) : 100;
}
// Preserve the image point under a viewport anchor when changing CSS scale.
export function zoomScroll(scroll, anchor, before, after, padding = 20) {
    return (scroll + anchor - padding) * after / before + padding - anchor;
}
export function addZoomControls(toolbar, scroll, view, onChange = () => {}) {
    let value = 60;
    const add = (tag, text) => {const e=document.createElement(tag);e.textContent=text;toolbar.append(e);return e;};
    const label=add('label','Zoom '),input=document.createElement('input');
    input.type='number';input.min=MIN_ZOOM;input.max=MAX_ZOOM;input.step=10;input.value=value;input.title='Zoom percent (10–3200%)';input.style.width='78px';label.append(input,document.createTextNode('%'));
    function set(next, anchor=[scroll.clientWidth/2,scroll.clientHeight/2]) {
        next=clampZoom(next);const old=value;value=next;input.value=Math.round(value);
        const x=zoomScroll(scroll.scrollLeft,anchor[0],old,next),y=zoomScroll(scroll.scrollTop,anchor[1],old,next);
        view.style.width=`${view.width*value/100}px`;view.style.height=`${view.height*value/100}px`;
        scroll.scrollLeft=x;scroll.scrollTop=y;onChange();
    }
    for(const [text,fn] of [['−',()=>set(value/1.4)],['+',()=>set(value*1.4)],['100%',()=>set(100)],['Fit',()=>set(100*Math.min((scroll.clientWidth-40)/view.width,(scroll.clientHeight-40)/view.height))]]) {
        const b=add('button',text);b.type='button';b.onclick=fn;
    }
    const hint=add('span','Ctrl/Cmd-wheel: zoom · middle drag: pan');hint.className='layers-help';
    input.onchange=()=>set(input.value);
    scroll.addEventListener('wheel',e=>{
        if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();
        const r=scroll.getBoundingClientRect();set(value*Math.exp(-e.deltaY*.002),[e.clientX-r.left,e.clientY-r.top]);
    },{passive:false});
    // Capture middle drags before the canvas can start a brush stroke or move a layer.
    let pan=null;
    scroll.addEventListener('pointerdown',e=>{if(e.button!==1)return;e.preventDefault();e.stopPropagation();pan={x:e.clientX,y:e.clientY,left:scroll.scrollLeft,top:scroll.scrollTop};scroll.setPointerCapture(e.pointerId);},true);
    scroll.addEventListener('pointermove',e=>{if(!pan)return;e.preventDefault();e.stopPropagation();scroll.scrollLeft=pan.left+pan.x-e.clientX;scroll.scrollTop=pan.top+pan.y-e.clientY;},true);
    for(const name of ['pointerup','pointercancel'])scroll.addEventListener(name,e=>{if(!pan)return;pan=null;e.stopPropagation();},true);
    scroll.addEventListener('auxclick',e=>{if(e.button===1)e.preventDefault();});
    scroll.addEventListener('scroll',onChange);
    return {set,fit:()=>set(100*Math.min((scroll.clientWidth-40)/view.width,(scroll.clientHeight-40)/view.height))};
}
