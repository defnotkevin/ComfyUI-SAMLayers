export function matrix(item, w, h) {
    const r = (item.angle || 0) * Math.PI / 180, s = item.scale ?? 1;
    const a = s * Math.cos(r), b = s * Math.sin(r), x = w/2, y = h/2;
    return [a,b,-b,a,x+(item.x||0)-a*x+b*y,y+(item.y||0)-b*x-a*y];
}
export function multiply(a,b) {
    return [a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3],a[0]*b[4]+a[2]*b[5]+a[4],a[1]*b[4]+a[3]*b[5]+a[5]];
}
export function layerMatrix(layer, state) {
    const own = matrix(layer,state.width,state.height), group = state.groups[layer.group];
    return group ? multiply(matrix(group,state.width,state.height),own) : own;
}
export function invertPoint(m,x,y) {
    const det=m[0]*m[3]-m[1]*m[2], dx=x-m[4],dy=y-m[5];
    return [(m[3]*dx-m[2]*dy)/det,(-m[1]*dx+m[0]*dy)/det];
}
export function ungroup(layer, group, w, h) {
    const m=multiply(matrix(group,w,h),matrix(layer,w,h));
    return {...layer, group:null, scale:Math.hypot(m[0],m[1]), angle:Math.atan2(m[1],m[0])*180/Math.PI,
        x:m[0]*w/2+m[2]*h/2+m[4]-w/2,y:m[1]*w/2+m[3]*h/2+m[5]-h/2,
        visible:layer.visible!==false && group.visible!==false};
}

export function transformPoint(m, x, y) {
    return [m[0]*x+m[2]*y+m[4], m[1]*x+m[3]*y+m[5]];
}

// Scale a top-level layer/group about a world-space anchor, keeping rotation.
export function resizeTransform(item, anchor, ratio, w, h) {
    const center=[w/2+(item.x||0),h/2+(item.y||0)];
    return {...item, scale:(item.scale??1)*ratio,
        x:anchor[0]+(center[0]-anchor[0])*ratio-w/2,
        y:anchor[1]+(center[1]-anchor[1])*ratio-h/2};
}

export function layerBlocks(layers) {
    const blocks=[];
    for(const layer of layers){
        const previous=blocks.at(-1);
        if(layer.group && previous?.[0].group===layer.group)previous.push(layer);
        else blocks.push([layer]);
    }
    return blocks;
}

// Storage order is back-to-front; the visible sidebar presents the reverse.
export function moveLayerBlock(layers, sourceId, targetId, inFront) {
    const blocks=layerBlocks(layers), source=blocks.find(b=>b.some(l=>l.id===sourceId)), target=blocks.find(b=>b.some(l=>l.id===targetId));
    if(!source||!target||source===target)return layers;
    const rest=blocks.filter(b=>b!==source);
    rest.splice(rest.indexOf(target)+(inFront?1:0),0,source);
    return rest.flat();
}

// Rotate a top-level layer/group around the selection's world-space center.
export function rotateTransform(item, pivot, degrees, w, h) {
    const r=degrees*Math.PI/180,c=Math.cos(r),s=Math.sin(r);
    const dx=w/2+(item.x||0)-pivot[0],dy=h/2+(item.y||0)-pivot[1];
    return {...item,angle:(item.angle||0)+degrees,
        x:pivot[0]+c*dx-s*dy-w/2,y:pivot[1]+s*dx+c*dy-h/2};
}
