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
