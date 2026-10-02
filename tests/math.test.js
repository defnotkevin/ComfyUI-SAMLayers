import test from 'node:test';
import assert from 'node:assert/strict';
import { matrix, layerMatrix, invertPoint, ungroup } from '../web/math.js';
const near=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-8,`${a} != ${b}`));
test('viewport hit test reverses rotation, scale and translation',()=>{
    const m=matrix({x:43,y:-18,scale:1.7,angle:39},640,480),p=[123,231];
    near(invertPoint(m,m[0]*p[0]+m[2]*p[1]+m[4],m[1]*p[0]+m[3]*p[1]+m[5]),p);
});
test('ungroup preserves appearance under combined transformations',()=>{
    const group={x:-20,y:11,scale:.7,angle:80,visible:false},layer={x:21,y:32,scale:1.3,angle:-25,group:'g'};
    const state={width:640,height:480,groups:{g:group}};
    near(layerMatrix(layer,state),matrix(ungroup(layer,group,640,480),640,480));
    assert.equal(ungroup(layer,group,640,480).visible,false);
});
test('identity transform keeps pixels in their original location',()=>near(matrix({},640,480),[1,0,0,1,0,0]));

test('corner resize keeps the opposite world anchor fixed after rotation',async()=>{
    const {resizeTransform,transformPoint}=await import('../web/math.js');
    const item={x:50,y:-30,scale:1.4,angle:37},w=800,h=600;
    const local=[110,160],anchor=transformPoint(matrix(item,w,h),...local);
    const next=resizeTransform(item,anchor,2,w,h);
    near(transformPoint(matrix(next,w,h),...local),anchor);
    assert.equal(next.scale,2.8);assert.equal(next.angle,37);
});
test('resizing a group preserves child geometry around a shared anchor',async()=>{
    const {resizeTransform,transformPoint}=await import('../web/math.js');
    const group={x:10,y:20,scale:.8,angle:15},layer={group:'g',x:70,y:30,scale:1.2,angle:50};
    const state={width:800,height:600,groups:{g:group}},anchor=[10,40];
    const before=transformPoint(layerMatrix(layer,state),150,200);
    state.groups.g=resizeTransform(group,anchor,.5,800,600);
    near(transformPoint(layerMatrix(layer,state),150,200),before.map((v,i)=>anchor[i]+(v-anchor[i])*.5));
});
test('dragging layer order preserves contiguous groups and member order',async()=>{
    const {moveLayerBlock}=await import('../web/math.js');
    const layers=[{id:'sky'},{id:'a',group:'g'},{id:'b',group:'g'},{id:'person'}];
    assert.deepEqual(moveLayerBlock(layers,'a','person',true).map(l=>l.id),['sky','person','a','b']);
    assert.deepEqual(moveLayerBlock(layers,'person','a',false).map(l=>l.id),['sky','person','a','b']);
    assert.equal(moveLayerBlock(layers,'a','b',true),layers);
    assert.deepEqual(layers.map(l=>l.id),['sky','a','b','person']);
});

test('rotation turns the mask about its selected center without changing scale',async()=>{
    const {rotateTransform,transformPoint}=await import('../web/math.js');
    const item={x:33,y:-20,scale:1.6,angle:20},w=800,h=600,local=[150,90];
    const pivot=transformPoint(matrix(item,w,h),...local);
    const result=rotateTransform(item,pivot,90,w,h);
    near(transformPoint(matrix(result,w,h),...local),pivot);
    assert.equal(result.scale,item.scale);assert.equal(result.angle,110);
    const before=transformPoint(matrix(item,w,h),200,110),after=transformPoint(matrix(result,w,h),200,110);
    near(after,[pivot[0]-(before[1]-pivot[1]),pivot[1]+before[0]-pivot[0]]);
});
test('group rotation preserves relative geometry and a full turn restores placement',async()=>{
    const {rotateTransform,transformPoint}=await import('../web/math.js');
    const group={x:20,y:-15,angle:37,scale:.8},layer={group:'g',x:12,y:23,angle:10,scale:1.4};
    const state={width:800,height:600,groups:{g:group}},pivot=[120,190];
    const before=transformPoint(layerMatrix(layer,state),80,90);
    state.groups.g=rotateTransform(group,pivot,180,800,600);
    near(transformPoint(layerMatrix(layer,state),80,90),[2*pivot[0]-before[0],2*pivot[1]-before[1]]);
    const full=rotateTransform(group,pivot,360,800,600);
    near(matrix(full,800,600),matrix(group,800,600));
});
