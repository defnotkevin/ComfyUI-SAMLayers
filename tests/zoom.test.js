import test from 'node:test';
import assert from 'node:assert/strict';
import {clampZoom,zoomScroll} from '../web/zoom.js';
test('allows pixel-level magnification with finite bounds',()=>{
    assert.equal(clampZoom(3200),3200);assert.equal(clampZoom(9999),3200);
    assert.equal(clampZoom(0),10);assert.equal(clampZoom(NaN),100);
});
test('zoom preserves the source pixel beneath the pointer',()=>{
    const scroll=300,anchor=180,before=60,after=1600;
    const next=zoomScroll(scroll,anchor,before,after);
    assert.ok(Math.abs((scroll+anchor-20)/before-(next+anchor-20)/after)<1e-9);
});

// Exercise the real event callbacks independently of the ComfyUI host.
test('plain wheel and middle drag navigate either editor without modifier keys',async()=>{
    const {addZoomControls}=await import('../web/zoom.js');
    const previous=globalThis.document;
    const element=()=>({style:{},children:[],append(...items){this.children.push(...items);}});
    globalThis.document={createElement:element,createTextNode:text=>({text})};
    try {
        const listeners={},toolbar=element(),view={width:1000,height:800,style:{}};
        const scroll={clientWidth:600,clientHeight:400,scrollLeft:100,scrollTop:50,
            addEventListener(name,fn){(listeners[name]??=[]).push(fn);},
            getBoundingClientRect:()=>({left:10,top:20}),setPointerCapture(){}};
        addZoomControls(toolbar,scroll,view);
        let prevented=0,stopped=0;
        listeners.wheel[0]({deltaY:-100,deltaMode:0,clientX:200,clientY:200,
            preventDefault(){prevented++;},stopPropagation(){stopped++;}});
        assert.ok(parseFloat(view.style.width)>600);
        assert.equal(view.style.minWidth,view.style.width);
        assert.equal(prevented,1);assert.equal(stopped,1);
        const original=[scroll.scrollLeft,scroll.scrollTop];
        const e={button:1,pointerId:1,clientX:100,clientY:100,preventDefault(){},stopPropagation(){}};
        listeners.pointerdown[0](e);listeners.pointermove[0]({...e,clientX:80,clientY:70});
        assert.equal(scroll.scrollLeft,original[0]+20);assert.equal(scroll.scrollTop,original[1]+30);
        listeners.pointerup[0](e);
        const width=view.style.width;
        listeners.wheel[0]({...e,shiftKey:true,deltaY:-100});
        assert.equal(view.style.width,width);
    } finally {globalThis.document=previous;}
});
