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
