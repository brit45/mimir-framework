/* Geometry-only preview; never changes annotations or claims semantic certainty. */
(function (root) {
  'use strict';
  const colors = {plausible:'#7ce1ba', ambiguous:'#ffd58a', misplaced:'#ffa591', unknown:'#a1b0bf'};
  function nearest(position, box) {
    return [Math.max(box[0],Math.min(box[2],position[0])),Math.max(box[1],Math.min(box[3],position[1]))];
  }
  function validMask(mask) {
    return mask?.type==='polygon' && Array.isArray(mask.points) && mask.points.length>=3 &&
      mask.points.every(p=>Array.isArray(p) && p.length===2 && p.every(v=>Number.isFinite(v) && v>=0 && v<=1));
  }
  function inside(position, points) {
    let result=false;
    for(let i=0,j=points.length-1;i<points.length;j=i++) {
      const [x,y]=points[i],[xx,yy]=points[j];
      if((y>position[1])!==(yy>position[1]) && position[0]<(xx-x)*(position[1]-y)/(yy-y)+x)result=!result;
    }
    return result;
  }
  function nearestMask(position, points, width, height) {
    if(inside(position,points))return position;
    let closest=null,best=Infinity;
    const px=position[0]*width,py=position[1]*height;
    for(let i=0;i<points.length;i++) {
      const a=points[i],b=points[(i+1)%points.length];
      const ax=a[0]*width,ay=a[1]*height,bx=b[0]*width,by=b[1]*height;
      const dx=bx-ax,dy=by-ay,denominator=dx*dx+dy*dy;
      const t=denominator?Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/denominator)):0;
      const x=ax+t*dx,y=ay+t*dy,d=(px-x)**2+(py-y)**2;
      if(d<best){best=d;closest=[x/width,y/height];}
    }
    return closest;
  }
  function assess(annotation, width, height, tolerance = .03) {
    const results=[], entities=annotation?.entities || [];
    const validPosition = p => Array.isArray(p) && p.length===2 && p.every(v=>Number.isFinite(v) && v>=0 && v<=1);
    const validBox = b => Array.isArray(b) && b.length===4 && b.every(v=>Number.isFinite(v) && v>=0 && v<=1) && b[0]<b[2] && b[1]<b[3];
    const diagonal=Math.hypot(width,height);
    const endpoint=(p,e)=>validMask(e.mask)?nearestMask(p,e.mask.points,width,height):validBox(e.bbox)?nearest(p,e.bbox):null;
    const distance=(p,e)=>{const q=endpoint(p,e);return q?Math.hypot((p[0]-q[0])*width,(p[1]-q[1])*height)/diagonal:Infinity;};
    for(const owner of entities) for(const point of owner.skeleton?.points || []) {
      // Root points represent the attachment of the skeleton to its owner. Other
      // points can extend outside the body, including tails and custom appendages.
      const rootPoint=!point.parent && !(owner.skeleton.edges || []).some(edge=>edge.to===point.name);
      const targets=[...new Set([...(rootPoint ? [owner.id] : []),...(point.entities || [])])];
      for(const targetId of targets) {
        const target=entities.find(e=>e.id===targetId);
        const result={owner:owner.id, point:point.name, target:targetId, implicit:!(point.entities || []).includes(targetId), status:'unknown', reason:'', position:point.position, endpoint:null, suggestion:null};
        if(!target)result.reason='Entité cible introuvable.';
        else if(!validPosition(point.position))result.reason='Placez ce point pour vérifier son attache.';
        else if(!validBox(target.bbox) && !validMask(target.mask))result.reason='Dessinez une boîte ou un masque pour l’entité cible.';
        else if(!(width>0 && height>0 && Number.isFinite(diagonal)))result.reason='Chargez l’image pour vérifier les distances.';
        else {
          const d=distance(point.position,target);result.distance=d;result.endpoint=endpoint(point.position,target);
          const geometryName=validMask(target.mask)?'masque':'boîte';
          const targetPhrase=geometryName==='masque'?'du masque':'de la boîte';
          const alternatives=entities.filter(e=>e.id!==owner.id && !targets.includes(e.id) && (validBox(e.bbox) || validMask(e.mask)))
            .map(e=>({id:e.id,distance:distance(point.position,e)})).filter(e=>e.distance<=tolerance).sort((a,b)=>a.distance-b.distance);
          if(d>tolerance) {
            result.status='misplaced';result.reason=`Point trop éloigné ${targetPhrase} cible (${(d*100).toFixed(1)} % de la diagonale ; tolérance ${(tolerance*100).toFixed(1)} %).`;
            if(alternatives.length) {result.suggestion=alternatives[0].id;result.reason+=` Cible proche possible : ${alternatives[0].id}.`;}
          } else if(alternatives.some(e=>e.distance<=d)) {
            result.status='ambiguous';result.reason=`Plusieurs boîtes sont aussi proches : ${alternatives.map(e=>e.id).join(', ')}. Vérifiez l’identité de la cible.`;
          } else {
            result.status='plausible';result.reason=d===0
              ?`Point situé ${geometryName==='masque'?'dans le':'dans la'} ${geometryName} cible.`
              :`Point proche du bord ${geometryName==='masque'?'du':'de la'} ${geometryName} cible.`;
          }
        }
        results.push(result);
      }
    }
    return results;
  }
  const api={assess,colors};root.SkeletonLinks=api;
  if(typeof module!=='undefined' && module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
