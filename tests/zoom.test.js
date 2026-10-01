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
