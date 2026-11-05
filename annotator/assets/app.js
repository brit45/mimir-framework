'use strict';
const $ = id => document.getElementById(id);
const csrf = document.querySelector('meta[name="csrf-token"]').content;
let dataset = null, index = -1, annotation = null, revision = null, selected = -1, dirty = false, busy = false, gesture = null, forceDraw = false, skeletonTarget = null, forceMask = null, pendingCreation = null, importedMasks = {};
let linkResults = [], focusedLink = null, linkResultsSignature = '';
const canvas = $('canvas'), ctx = canvas.getContext('2d'), img = $('image');
function message(text = '', error = false) { $('message').textContent = text; $('message').className = error ? 'error' : ''; }
async function api(name, data) {
  if (name !== 'scan') data = {...data, dataset_root:dataset?.root};
  const res = await fetch(`api/${name}.php`, {method:'POST', headers:{'Content-Type':'application/json','X-CSRF-Token':csrf}, body:JSON.stringify(data)});
  const value = await res.json(); if (!res.ok || value.error) throw new Error(value.error || `Erreur ${res.status}`); return value;
}
function status(text) { $('status').textContent = text; $('status').classList.toggle('dirty', dirty); }
function changed() { dirty = true; status('Modifiée · non sauvegardée'); }
function discard() { return !dirty || confirm('Des modifications ne sont pas sauvegardées. Les abandonner ?'); }
function progress() {
  const total = dataset.images.length, done = dataset.annotated.length, percent = total ? done / total * 100 : 0;
  $('counter').textContent = `Image ${index + 1} / ${total}`;
  $('counts').textContent = `Annotées : ${done} · Restantes : ${total - done} · Progression : ${percent.toFixed(1)} %`;
  $('progress').value = percent; $('prev').disabled = index <= 0; $('next').disabled = index >= total - 1;
  $('last').disabled = !dataset.state.last_annotation;
}
async function prepareMaskBoxes(next) {
  const found={};
  for(const [image, masks] of Object.entries(next.mask_links || {})) {
    found[image]=[];
    for(const source of masks) {
      const url=`api/image.php?image=${encodeURIComponent(source)}&dataset_root=${encodeURIComponent(next.root)}`;
      const bitmap=await createImageBitmap(await (await fetch(url)).blob());
      const width=bitmap.width,height=bitmap.height;
      const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
      const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(bitmap,0,0);
      const pixels=context.getImageData(0,0,width,height).data;
      bitmap.close();
      const corners=[0,(width-1)*4,((height-1)*width)*4,(height*width-1)*4];
      const colorAt=offset=>[pixels[offset],pixels[offset+1],pixels[offset+2],pixels[offset+3]].join(',');
      const cornerColors=corners.map(colorAt);
      const background=cornerColors.find(color=>cornerColors.filter(value=>value===color).length>=2);
      let minX=width,minY=height,maxX=-1,maxY=-1;
      for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
        const offset=(y*width+x)*4;
        const color=colorAt(offset);
        const foreground=background ? color!==background : pixels[offset+3]>0 &&
          (pixels[offset]!==0 || pixels[offset+1]!==0 || pixels[offset+2]!==0);
        if(foreground && pixels[offset+3]>0) {minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);}
      }
      if(maxX>=minX && maxY>=minY)found[image].push({source,bbox:[minX/width,minY/height,(maxX+1)/width,(maxY+1)/height]});
    }
  }
  importedMasks=found;
}
async function run(fn) { if (busy) return; busy = true; document.body.setAttribute('aria-busy','true'); document.querySelector('.editor').inert = true; document.querySelector('nav').inert = true; $('open').inert = true; $('create-form').inert = true; try { await fn(); } catch (e) { message(e.message, true); } finally { busy = false; document.body.removeAttribute('aria-busy'); document.querySelector('.editor').inert = false; document.querySelector('nav').inert = false; $('open').inert = false; $('create-form').inert = false; } }
function input(value, onInput, placeholder = '') { const el = document.createElement('input'); el.value = value; el.placeholder = placeholder; el.setAttribute('aria-label', placeholder); el.oninput = () => { onInput(el.value); changed(); }; return el; }
function button(text, action) { const el = document.createElement('button'); el.type = 'button'; el.textContent = text; el.onclick = event => { event.stopPropagation(); action(); }; return el; }
function select(value, values, onChange, label) {
  const el = document.createElement('select'); el.setAttribute('aria-label', label);
  for (const v of [...new Set([value, ...values])]) { const option = document.createElement('option'); option.value = v; option.textContent = v || (label === 'Point parent' ? 'Aucune liaison' : 'Choisir une entité'); el.append(option); }
  el.value = value; el.onchange = () => { onChange(el.value); changed(); }; return el;
}
function label(text, el) { const wrapper = document.createElement('label'); wrapper.append(document.createTextNode(text), el); return wrapper; }
function newEntity(box) { let n = 1; while (annotation.entities.some(e => e.id === `entity_${n}`)) n++; const entity = {id:`entity_${n}`,phrase:'',role:'object'}; if (box) entity.bbox = box; annotation.entities.push(entity); selected = annotation.entities.length - 1; changed(); renderEntities(); renderRecords(); draw(); return entity; }
function maskBounds(points) {
  return [Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))];
}
function polygonArea(points) {
  return Math.abs(points.reduce((sum,p,i)=>sum+p[0]*points[(i+1)%points.length][1]-points[(i+1)%points.length][0]*p[1],0))/2;
}
function pointInPolygon(p, points) {
  let inside=false;
  for(let i=0,j=points.length-1;i<points.length;j=i++) {
    const [x,y]=points[i],[xx,yy]=points[j];
    if((y>p[1])!==(yy>p[1]) && p[0]<(xx-x)*(p[1]-y)/(yy-y)+x)inside=!inside;
  }
  return inside;
}
function setMask(points, target, kind = 'decor') {
  if(points.length<3 || polygonArea(points)<=1e-8) { message('Masque ignoré : tracez une forme fermée avec une aire visible.',true);return; }
  let entity;
  if(target>=0) entity=annotation.entities[target];
  else {
    const prefix=kind==='ocr'?'ocr':'decor';let n=1;while(annotation.entities.some(e=>e.id===`${prefix}_${n}`))n++;
    entity=kind==='ocr'?{id:`${prefix}_${n}`,phrase:'texte dans l’image',role:'text'}:{id:`${prefix}_${n}`,phrase:'décor',role:'background'};
    annotation.entities.push(entity);target=annotation.entities.length-1;
  }
  entity.mask={type:'polygon',points};
  entity.bbox=maskBounds(points);
  if(kind==='ocr')entity.ocr ||= {text:'',translation:'',source_language:'',target_language:'',tags:[]};
  selected=target;changed();renderEntities();renderRecords();draw();
  message(kind==='ocr'?'Zone OCR créée. Saisissez le texte, sa traduction, les langues et les tags dans sa fiche.':kind==='track'?'Masque de l’occurrence suivie enregistré.':'Masque enregistré comme entité de décor. Décrivez-le et liez-y les points du personnage.');
}
function renameEntity(entity, next) {
  const prev = entity.id; entity.id = next;
  for (const a of annotation.attributes) if (a.entity === prev) a.entity = next;
  for (const r of annotation.relations) { if (r.subject === prev) r.subject = next; if (r.object === prev) r.object = next; }
  for (const e of annotation.entities) for (const p of e.skeleton?.points || []) if(p.entities) p.entities=p.entities.map(id=>id===prev?next:id);
  renderRecords(); draw();
}
function renderEntities() {
  $('entities').replaceChildren();
  annotation.entities.forEach((entity, i) => {
    const card = document.createElement('div'); card.className = `entity${selected === i ? ' selected' : ''}`;
    card.onclick = () => { selected = i; for (const [j, el] of [...$('entities').children].entries()) el.classList.toggle('selected', j === i); draw(); };
    const row = document.createElement('div'); row.className = 'mini-row';
    row.append(label('Identifiant', input(entity.id, v => renameEntity(entity, v), 'Identifiant')), label('Rôle', select(entity.role, ['primary_subject','secondary_subject','object','background'], v => { entity.role = v; }, 'Rôle')));
    card.append(row, label('Description de l’entité', input(entity.phrase, v => { entity.phrase = v; }, 'Une femme aux cheveux noirs')));
    const trackRow=document.createElement('div');trackRow.className='mini-row';
    const trackId=input(entity.track?.id || '',()=>syncTrack(), 'ex. personnage-asha');trackId.setAttribute('list','track-identities');
    const trackName=input(entity.track?.name || '',()=>syncTrack(), 'Nom explicite du personnage');
    function syncTrack() {
      if(trackId.value.trim() || trackName.value.trim()) entity.track={id:trackId.value.trim(),name:trackName.value.trim()};
      else delete entity.track;
    }
    trackId.addEventListener('change',()=>{
      const known=dataset?.identities?.find(item=>item.id===trackId.value);
      if(known) trackName.value=known.name;
      syncTrack();renderIdentityChoices();changed();
    });
    trackRow.append(label('ID de suivi (partagé entre images)',trackId),label('Nom de suivi',trackName));
    const trackHint=document.createElement('small');trackHint.className='wide';trackHint.textContent=entity.track
      ? `Suivi : ${entity.track.name || 'nom à compléter'} · identifiant ${entity.track.id || 'à compléter'}`
      : 'Pour suivre cette entité dans d’autres images, tracez son masque puis renseignez le même ID et le même nom.';
    card.append(trackRow,trackHint);
    if(entity.ocr) {
      const ocr=document.createElement('div');ocr.className='ocr-fields';
      const ocrText=document.createElement('textarea');ocrText.rows=2;ocrText.value=entity.ocr.text || '';ocrText.placeholder='Texte exact reconnu dans la zone';ocrText.setAttribute('aria-label','Texte OCR source');
      ocrText.oninput=()=>{entity.ocr.text=ocrText.value;changed();};
      const translation=document.createElement('textarea');translation.rows=2;translation.value=entity.ocr.translation || '';translation.placeholder='Traduction facultative';translation.setAttribute('aria-label','Traduction OCR');
      translation.oninput=()=>{entity.ocr.translation=translation.value;changed();};
      const sourceLanguage=input(entity.ocr.source_language || '',v=>{entity.ocr.source_language=v;},'ja');
      const targetLanguage=input(entity.ocr.target_language || '',v=>{entity.ocr.target_language=v;},'fr');
      const tags=input((entity.ocr.tags || []).join(', '),v=>{entity.ocr.tags=v.split(',').map(tag=>tag.trim()).filter(Boolean);},'dialogue, enseigne, SFX');
      ocr.append(label('Texte source',ocrText),label('Traduction',translation));
      const languages=document.createElement('div');languages.className='mini-row';languages.append(label('Langue source',sourceLanguage),label('Langue cible',targetLanguage));ocr.append(languages,label('Tags OCR (séparés par des virgules)',tags));
      card.append(ocr);
    }
    const coordinates = document.createElement('small'); coordinates.textContent = entity.mask
      ? `Masque polygonal · ${entity.mask.points.length} sommets · boîte englobante : ${(entity.bbox || maskBounds(entity.mask.points)).map(v => v.toFixed(3)).join(', ')}`
      : entity.bbox ? `BBox : ${entity.bbox.map(v => v.toFixed(3)).join(', ')}` : 'Sans boîte · sélectionnez cette entité et dessinez sur une zone libre.';
    renderSkeleton(entity, card);
    const drawBox=button('Sélectionner / dessiner', () => { selected = i; forceDraw = true; skeletonTarget = null; renderEntities(); draw(); message('Glissez sur l’image pour dessiner la boîte de cette entité.'); });
    drawBox.disabled=Boolean(entity.mask);
    card.append(coordinates, document.createElement('br'), button(entity.mask?'Redessiner le masque':'Tracer un masque', () => { selected=i;forceMask={target:i,kind:entity.ocr?'ocr':'decor'};forceDraw=false;skeletonTarget=null;renderEntities();draw();message('Glissez sur l’image pour tracer le masque de cette entité.'); }), drawBox, button('Supprimer l’entité', () => {
      if (!confirm(`Supprimer ${entity.id}, ses attributs et ses relations ?`)) return;
      annotation.entities.splice(i,1); for(const e of annotation.entities) for(const p of e.skeleton?.points || []) if(p.entities) p.entities=p.entities.filter(id=>id!==entity.id); skeletonTarget=null; annotation.attributes = annotation.attributes.filter(a => a.entity !== entity.id); annotation.relations = annotation.relations.filter(r => r.subject !== entity.id && r.object !== entity.id); selected = -1; changed(); renderEntities(); renderRecords(); draw();
    })); $('entities').append(card);
  });
  renderIdentityChoices();
}
function renderIdentityChoices() {
  const choices=$('track-identity');if(!choices)return;
  const current=choices.value;choices.replaceChildren(new Option('Choisir une identité suivie',''));
  for(const identity of dataset?.identities || []) {
    const option=new Option(`${identity.name} · ${identity.id} · ${identity.images.length} image(s)`,identity.id);
    if(identity.conflicting_names?.length)option.textContent+=` · noms divergents : ${identity.conflicting_names.join(', ')}`;
    choices.append(option);
  }
  choices.value=(dataset?.identities || []).some(item=>item.id===current)?current:'';
  const datalist=$('track-identities');datalist.replaceChildren();
  for(const identity of dataset?.identities || []) {
    const option=document.createElement('option');option.value=identity.id;option.label=`${identity.name} (${identity.images.length} image(s))`;datalist.append(option);
  }
}
async function uploadImage(file, relativePath, root) {
  const body=new FormData();body.append('file',file,file.name);body.append('relative_path',relativePath);body.append('dataset_root',root);
  const response=await fetch('api/upload.php',{method:'POST',headers:{'X-CSRF-Token':csrf},body});
  const result=await response.json();
  if(!response.ok || result.error)throw new Error(result.error || `Erreur ${response.status}`);
  return result;
}
function selectedImportFiles() {
  const files=[...$('import-files').files,...$('import-folder').files];
  return files.map(file=>({file,path:file.webkitRelativePath || file.name}));
}
function importPathForFile(path) {
  return path.split('/').filter(Boolean).map(part=>part.normalize('NFC')).join('/');
}
let directoryBrowserPath='', directoryPickerMode='parent';
async function showDirectory(path) {
  const result=await api('directories',{path});directoryBrowserPath=result.path;
  $('directory-current').textContent=result.path;
  $('directory-up').disabled=!result.parent;
  $('directory-up').onclick=()=>{if(result.parent)showDirectory(result.parent).catch(error=>message(error.message,true));};
  const list=$('directory-list');list.replaceChildren();
  if(!result.directories.length) {
    const empty=document.createElement('p');empty.textContent='Aucun sous-dossier accessible.';list.append(empty);
  }
  for(const entry of result.directories) {
    const item=document.createElement('button');item.type='button';item.textContent=`Dossier : ${entry.name}`;
    item.onclick=()=>showDirectory(entry.path).catch(error=>message(error.message,true));list.append(item);
  }
}
async function openDirectoryPicker(path, mode) {
  directoryPickerMode=mode;
  $('directory-title').textContent=mode==='parent'
    ? 'Choisir le dossier parent du nouveau dataset'
    : 'Choisir le dataset à ouvrir ou modifier';
  $('directory-select').textContent=mode==='parent'?'Choisir ce dossier parent':'Ouvrir / modifier ce dataset';
  await showDirectory(path || '/');
  $('directory-dialog').showModal();
}
async function createDataset(event) {
  event.preventDefault();
  run(async()=>{
    if(!discard())return;
    const parent=$('dataset-parent').value.trim(),name=$('dataset-name').value.trim();
    const files=selectedImportFiles();
    if(!parent || !name)throw new Error('Choisissez un dossier parent et un nom de dataset.');
    if(!files.length)throw new Error('Sélectionnez au moins une image ou un dossier d’images à importer.');
    const accepted=files.filter(({file,path})=>/\.(jpe?g|png|webp|bmp|json|mimir-meta)$/i.test(path || file.name));
    if(!accepted.some(({file,path})=>/\.(jpe?g|png|webp|bmp)$/i.test(path || file.name)))throw new Error('Aucune image JPG, PNG, WebP ou BMP dans la sélection.');
    let root;
    if(pendingCreation?.parent===parent && pendingCreation.name===name) root=pendingCreation.root;
    else {
      const created=await api('create',{parent,name});
      root=created.root;pendingCreation={parent,name,root};
    }
    $('root').value=root;
    const failures=[];let imported=0,skipped=0,metadataImported=0;
    $('import-progress').hidden=false;$('import-progress').max=accepted.length;$('import-progress').value=0;
    $('create-submit').disabled=true;
    try {
      for(let i=0;i<accepted.length;i++) {
        const {file,path}=accepted[i];$('import-status').textContent=`Import ${i+1}/${accepted.length} · ${path}`;
        try {
          const result=await uploadImage(file,importPathForFile(path),root);
          if(/\.(jpe?g|png|webp|bmp)$/i.test(path)) {
            if(result.skipped)skipped++;else imported++;
          } else if(!result.skipped)metadataImported++;
        } catch(error) { failures.push(`${path} : ${error.message}`); }
        $('import-progress').value=i+1;
      }
    } finally { $('create-submit').disabled=false; }
    const next=await api('scan',{root});dataset=next;index=-1;annotation=null;dirty=false;
    await prepareMaskBoxes(next);
    $('workspace').hidden=true;img.removeAttribute('src');
    $('warnings').hidden=!next.warnings.length;$('warnings').querySelector('pre').textContent=next.warnings.join('\n');
    if(next.images.length)await load(next.resume_index);
    if(!failures.length) {
      pendingCreation=null;$('import-files').value='';$('import-folder').value='';
    }
    const summary=`Dataset créé · ${imported} image(s) importée(s)${metadataImported?` · ${metadataImported} fichier(s) de métadonnées importé(s)`:''}${skipped?` · ${skipped} image(s) déjà identique(s) ignorée(s)`:''} · ${next.images.length} image(s) prête(s).`;
    $('import-status').textContent=failures.length?`${failures.length} échec(s)`:summary;
    message(failures.length?`${summary}\n${failures.join('\n')}`:summary,failures.length>0);
    status(next.images.length?'Dataset prêt':'Aucune image importée');
    if(next.images.length)progress();
  });
}
function renderRecords() {
  const ids = annotation.entities.map(e => e.id);
  for (const kind of ['attributes','relations']) {
    $(kind).replaceChildren();
    annotation[kind].forEach((record, i) => {
      const row = document.createElement('div'); row.className = 'record';
      if (kind === 'attributes') row.append(select(record.entity, ids, v => { record.entity = v; }, 'Entité'), input(record.property, v => { record.property = v; }, 'Propriété'), input(record.value, v => { record.value = v; }, 'Valeur'));
      else row.append(select(record.subject, ids, v => { record.subject = v; }, 'Sujet'), input(record.predicate, v => { record.predicate = v; }, 'Relation (near, on…)'), select(record.object, ids, v => { record.object = v; }, 'Objet'));
      row.append(button('×', () => { annotation[kind].splice(i,1); changed(); renderRecords(); })); $(kind).append(row);
    });
  }
}
const fields = {
  caption: ['captions','primary'], 'light-description':['lighting','description'], 'light-source':['lighting','source'], 'light-target':['lighting','target'], composition:['composition','description'], 'caption-verified':['quality','caption_verified'], 'relations-verified':['quality','relations_verified']
};
for (const [id, [group, key]] of Object.entries(fields)) $(id).addEventListener('input', () => { if (!annotation || busy) return; annotation[group][key] = $(id).type === 'checkbox' ? $(id).checked : $(id).value; changed(); });
$('alternatives').oninput = () => { if (!annotation || busy) return; annotation.captions.alternatives = $('alternatives').value.split('\n').filter(v => v.trim()); changed(); };
function renderForm() {
  for (const [id, [group, key]] of Object.entries(fields)) { if ($(id).type === 'checkbox') $(id).checked = annotation[group][key]; else $(id).value = annotation[group][key]; }
  $('alternatives').value = annotation.captions.alternatives.join('\n'); renderEntities(); renderRecords();
}
async function load(i) {
  if (i < 0 || i >= dataset.images.length) return;
  const relative = dataset.images[i], result = await api('load', {image:relative});
  index = i; annotation = result.annotation; revision = result.revision; dirty = false; selected = -1; forceDraw = false; forceMask = null; skeletonTarget = null; focusedLink = null; gesture = null;
  for(const mask of importedMasks[relative] || []) {
    if(annotation.entities.some(entity=>entity.mask_source===mask.source))continue;
    let n=1;while(annotation.entities.some(entity=>entity.id===`mask_${n}`))n++;
    const label=dataset.class_names?.[0] || 'Masque';
    annotation.entities.push({id:`mask_${n}`,phrase:label,role:'object',bbox:mask.bbox,class_id:0,mask_source:mask.source});
    if(!annotation.captions.primary.trim())annotation.captions.primary=`Zone annotée : ${label}.`;
    dirty=true;
  }
  for(const entity of annotation.entities)if(entity.mask?.type==='polygon' && Array.isArray(entity.mask.points) && entity.mask.points.length>=3)entity.bbox=maskBounds(entity.mask.points);
  $('workspace').hidden = false; $('filename').textContent = relative; message(); renderForm(); progress(); status(dirty ? 'Masques importés · non sauvegardés' : result.exists ? 'Annotation existante' : 'Nouvelle annotation');
  img.src = `api/image.php?image=${encodeURIComponent(relative)}&dataset_root=${encodeURIComponent(dataset.root)}`;
  try { dataset.state = (await api('state', {image:relative})).state; progress(); } catch (e) { message(`Image chargée ; historique non enregistré : ${e.message}`,true); }
}
async function save() {
  if (!annotation) return;
  const savedImage=dataset.images[index];
  const result = await api('save', {image:savedImage, annotation, revision});
  revision = result.revision;
  if (result.state) dataset.state = result.state;
  if (!dataset.annotated.includes(savedImage)) dataset.annotated.push(savedImage);
  dirty = false; status('Annotation sauvegardée'); progress();
  let warning=result.warning || '';
  try {
    const refreshed=await api('scan',{root:dataset.root});
    const savedIndex=refreshed.images.indexOf(savedImage);
    if(savedIndex<0) throw new Error('L’image sauvegardée ne figure plus dans le dataset.');
    dataset=refreshed;index=savedIndex;
  } catch(e) {
    warning=[warning,`Liste des identités non actualisée : ${e.message}`].filter(Boolean).join('\n');
  }
  renderIdentityChoices();
  message(warning || 'JSON enregistré à côté de l’image.', Boolean(warning));
  progress();
}
function navigate(i) { run(async () => { if (discard()) await load(i); }); }
$('open').onsubmit = event => { event.preventDefault(); run(async () => {
  if (!discard()) return;
  const next = await api('scan', {root:$('root').value}); dataset = next; index = -1; annotation = null; dirty = false; $('workspace').hidden = true; img.removeAttribute('src');
  try { localStorage.setItem('lumen-annotation-root', next.root); } catch (_) { /* Storage may be disabled. */ }
  $('root').value = next.root; $('warnings').hidden = !next.warnings.length; $('warnings').querySelector('pre').textContent = next.warnings.join('\n');
  await prepareMaskBoxes(next);
  if (!next.images.length) { status('Aucune image'); message('Aucune image supportée dans ce dossier.'); return; }
  await load(next.resume_index);
}); };
$('create-form').onsubmit=createDataset;
$('browse-parent').onclick=()=>{
  const fallback=dataset?.root?.split('/').slice(0,-1).join('/') || '/';
  openDirectoryPicker($('dataset-parent').value.trim() || fallback,'parent')
    .catch(error=>message(error.message,true));
};
$('browse-root').onclick=()=>{
  openDirectoryPicker($('root').value.trim() || dataset?.root || '/','open')
    .catch(error=>message(error.message,true));
};
$('directory-select').onclick=()=>{
  const selectedPath=directoryBrowserPath;
  const mode=directoryPickerMode;
  $('directory-dialog').close();
  if(mode==='parent'){$('dataset-parent').value=selectedPath;return;}
  $('root').value=selectedPath;
  $('open').requestSubmit();
};
$('directory-cancel').onclick=()=>$('directory-dialog').close();
$('import-files').onchange=()=>{$('import-folder').value='';$('import-status').textContent=`${$('import-files').files.length} fichier(s) sélectionné(s)`;};
$('import-folder').onchange=()=>{$('import-files').value='';$('import-status').textContent=`${$('import-folder').files.length} fichier(s) sélectionné(s)`;};
try { $('root').value = localStorage.getItem('lumen-annotation-root') || ''; } catch (_) { /* Optional convenience. */ }
$('prev').onclick = () => navigate(index-1); $('next').onclick = () => navigate(index+1);
$('save').onclick = () => run(save); $('save-next').onclick = () => run(async () => { await save(); if (index+1 < dataset.images.length) await load(index+1); });
$('unannotated').onclick = () => { for (let n=1; n <= dataset.images.length; n++) { const i=(index+n)%dataset.images.length; if (!dataset.annotated.includes(dataset.images[i])) { navigate(i); return; } } message('Toutes les images sont annotées.'); };
$('last').onclick = () => { const i = dataset.images.indexOf(dataset.state.last_annotation); if (i >= 0) navigate(i); else message('Dernière annotation absente du dossier.',true); };
$('add-entity').onclick = () => { if (!busy) newEntity(); };
$('draw-track-occurrence').onclick = () => {
  if(busy)return;
  const identity=dataset?.identities.find(item=>item.id===$('track-identity').value);
  if(!identity){message('Choisissez une identité déjà annotée dans le dataset.',true);return;}
  const entity=newEntity();entity.phrase=identity.name;entity.role='primary_subject';entity.track={id:identity.id,name:identity.name};
  forceMask={target:selected,kind:'track'};forceDraw=false;skeletonTarget=null;renderEntities();draw();
  message(`Tracez le masque de ${identity.name} (${identity.id}) dans cette image.`);
};
$('add-attribute').onclick = () => { if (busy) return; annotation.attributes.push({entity:annotation.entities[selected]?.id || annotation.entities[0]?.id || '',property:'',value:''}); changed(); renderRecords(); };
$('add-relation').onclick = () => { if (busy) return; annotation.relations.push({subject:annotation.entities[0]?.id || '',predicate:'',object:annotation.entities[1]?.id || annotation.entities[0]?.id || ''}); changed(); renderRecords(); };
$('draw').onclick = () => { skeletonTarget = null;forceMask=null;forceDraw = true; selected = -1; renderEntities(); draw(); message('Glissez sur l’image pour créer une nouvelle entité.'); };
$('draw-mask').onclick = () => { skeletonTarget=null;forceDraw=false;forceMask={target:-1,kind:'decor'};selected=-1;renderEntities();draw();message('Glissez librement sur le décor pour tracer un nouveau masque polygonal.'); };
$('draw-ocr').onclick = () => { skeletonTarget=null;forceDraw=false;forceMask={target:-1,kind:'ocr'};selected=-1;renderEntities();draw();message('Glissez autour du texte à transcrire et traduire.'); };
function removeBox() { const entity=annotation?.entities[selected];if(!busy && entity?.bbox) { delete entity.bbox;delete entity.mask;if(entity.ocr)delete entity.ocr;if(entity.track)delete entity.track;changed();renderEntities();draw(); } }
$('remove-box').onclick = removeBox;
window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue=''; } });
canvas.addEventListener('keydown', e => { if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeBox(); } });
// Work in the displayed image rectangle, excluding object-fit letterboxing.
function rect() {
  const w=canvas.clientWidth, h=canvas.clientHeight, ratio=img.naturalWidth/img.naturalHeight;
  if (!Number.isFinite(ratio) || !w || !h) return null;
  const iw=Math.min(w,h*ratio), ih=iw/ratio; return {x:(w-iw)/2,y:(h-ih)/2,w:iw,h:ih};
}
function point(e, clamp=true) {
  const r=rect(); if (!r) return null; const c=canvas.getBoundingClientRect(), x=(e.clientX-c.left-r.x)/r.w, y=(e.clientY-c.top-r.y)/r.h;
  if (!clamp && (x<0 || x>1 || y<0 || y>1)) return null; return [Math.max(0,Math.min(1,x)),Math.max(0,Math.min(1,y))];
}
function handles(b) { return [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]],[(b[0]+b[2])/2,b[1]],[b[2],(b[1]+b[3])/2],[(b[0]+b[2])/2,b[3]],[b[0],(b[1]+b[3])/2]]; }
function updateLinkSimulation() {
  const enabled=$('simulate-links').checked;
  const tolerance=Number($('link-tolerance').value);
  if(!Number.isFinite(tolerance) || tolerance<0 || tolerance>10 || $('link-tolerance').value==='') {
    linkResults=[];$('link-results').textContent='La tolérance doit être entre 0 et 10 %.';linkResultsSignature='';return;
  }
  linkResults=enabled && annotation ? SkeletonLinks.assess(annotation,img.naturalWidth,img.naturalHeight,tolerance/100) : [];
  const signature=JSON.stringify([enabled,linkResults]);if(signature===linkResultsSignature)return;linkResultsSignature=signature;
  $('link-results').replaceChildren();if(!enabled)return;
  const labels={plausible:'Plausible',ambiguous:'Ambigu',misplaced:'À corriger',unknown:'Non vérifiable'};
  const summary=document.createElement('p');summary.textContent=linkResults.length ? Object.entries(labels).map(([key,name])=>`${name} : ${linkResults.filter(r=>r.status===key).length}`).join(' · ') : 'Aucune attache à vérifier. Ajoutez un squelette et liez ses points à des entités.';$('link-results').append(summary);
  for(const result of linkResults) {
    const row=document.createElement('div');row.className=`link-result ${result.status}`;
    const text=document.createElement('p');text.textContent=`${labels[result.status]} · ${result.owner} / ${result.point} → ${result.target}${result.implicit?' (propriétaire)':''} : ${result.reason}`;row.append(text);
    row.append(button('Voir cette attache',()=>{focusedLink={owner:result.owner,point:result.point,target:result.target};selected=annotation.entities.findIndex(e=>e.id===result.owner);renderEntities();draw();canvas.scrollIntoView({block:'center',behavior:'smooth'});}));
    if(result.suggestion && !result.implicit)row.append(button(`Remplacer par ${result.suggestion}`,()=>{
      if(busy)return;
      const owner=annotation.entities.find(e=>e.id===result.owner),point=owner?.skeleton?.points.find(p=>p.name===result.point);
      if(!point)return;
      point.entities=[...new Set(point.entities.map(id=>id===result.target?result.suggestion:id))];focusedLink=null;changed();renderEntities();draw();
    }));$('link-results').append(row);
  }
}
$('simulate-links').onchange=()=>{focusedLink=null;draw();};
$('link-tolerance').oninput=()=>draw();
function drawLinkSimulation(r) {
  for(const result of linkResults) {
    if(!result.position || !result.endpoint)continue;
    const target=annotation.entities.find(e=>e.id===result.target);if(!target?.bbox)continue;
    const focused=focusedLink && ['owner','point','target'].every(key=>focusedLink[key]===result[key]);
    ctx.save();ctx.strokeStyle=ctx.fillStyle=SkeletonLinks.colors[result.status];ctx.lineWidth=focused?4:2;ctx.setLineDash([6,4]);
    const b=target.bbox;ctx.strokeRect(r.x+b[0]*r.w,r.y+b[1]*r.h,(b[2]-b[0])*r.w,(b[3]-b[1])*r.h);
    const [x,y]=result.position,[tx,ty]=result.endpoint;
    ctx.beginPath();ctx.moveTo(r.x+x*r.w,r.y+y*r.h);ctx.lineTo(r.x+tx*r.w,r.y+ty*r.h);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.arc(r.x+x*r.w,r.y+y*r.h,focused?11:8,0,2*Math.PI);ctx.stroke();
    if(focused) {ctx.font='13px system-ui';ctx.fillText(`→ ${result.target}`,r.x+x*r.w+12,r.y+y*r.h+18);}ctx.restore();
  }
}
function draw() {
  updateLinkSimulation();
  const dpr=window.devicePixelRatio || 1; const width=Math.round(canvas.clientWidth*dpr), height=Math.round(canvas.clientHeight*dpr);
  if(canvas.width!==width || canvas.height!==height) {canvas.width=width;canvas.height=height;}
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight); const r=rect(); if (!r || !annotation) return;
  annotation.entities.forEach((entity,i) => {
    if(entity.mask?.type==='polygon' && Array.isArray(entity.mask.points) && entity.mask.points.length>=3) {
      const points=entity.mask.points,b=entity.bbox || maskBounds(points), x=r.x+b[0]*r.w, y=r.y+b[1]*r.h;
      ctx.beginPath();points.forEach((p,j)=>j?ctx.lineTo(r.x+p[0]*r.w,r.y+p[1]*r.h):ctx.moveTo(r.x+p[0]*r.w,r.y+p[1]*r.h));ctx.closePath();
      ctx.fillStyle=i===selected?'#ffd58a44':'#f0a45b33';ctx.strokeStyle=i===selected?'#ffd58a':'#f0a45b';ctx.lineWidth=i===selected?3:2;ctx.fill();ctx.stroke();
      ctx.font='13px system-ui';const text=entity.id,tw=ctx.measureText(text).width+10;ctx.fillStyle='#101820';ctx.fillRect(x,Math.max(r.y,y-21),tw,21);ctx.fillStyle='#e7edf3';ctx.fillText(text,x+5,Math.max(r.y,y-21)+15);
      if(i===selected)for(const p of points){ctx.beginPath();ctx.arc(r.x+p[0]*r.w,r.y+p[1]*r.h,3,0,Math.PI*2);ctx.fillStyle='#fff0c2';ctx.fill();}
      return;
    }
    if (!entity.bbox) return; const b=entity.bbox, x=r.x+b[0]*r.w, y=r.y+b[1]*r.h;
    ctx.strokeStyle=i===selected?'#7ce1ba':'#91bafa'; ctx.fillStyle=i===selected?'#7ce1ba22':'#91bafa15'; ctx.lineWidth=i===selected?2.5:1.5; ctx.fillRect(x,y,(b[2]-b[0])*r.w,(b[3]-b[1])*r.h); ctx.strokeRect(x,y,(b[2]-b[0])*r.w,(b[3]-b[1])*r.h);
    ctx.font='13px system-ui'; const text=entity.id, tw=ctx.measureText(text).width+10; ctx.fillStyle='#101820'; ctx.fillRect(x,Math.max(r.y,y-21),tw,21); ctx.fillStyle='#e7edf3'; ctx.fillText(text,x+5,Math.max(r.y,y-21)+15);
    if(i===selected) {ctx.fillStyle='#7ce1ba'; for(const [hx,hy] of handles(b)) ctx.fillRect(r.x+hx*r.w-4,r.y+hy*r.h-4,8,8);}
  });
  annotation.entities.forEach((entity,i) => {
    const points=entity.skeleton?.points || []; ctx.strokeStyle=ctx.fillStyle=i===selected?'#ffd58a':'#f3a4df'; ctx.lineWidth=2;
    for(const edge of entity.skeleton?.edges || []) {
      const a=points.find(p=>p.name===edge.from)?.position,b=points.find(p=>p.name===edge.to)?.position;
      if(a && b) {ctx.beginPath();ctx.moveTo(r.x+a[0]*r.w,r.y+a[1]*r.h);ctx.lineTo(r.x+b[0]*r.w,r.y+b[1]*r.h);ctx.stroke();if(edge.label)ctx.fillText(edge.label,r.x+(a[0]+b[0])/2*r.w,r.y+(a[1]+b[1])/2*r.h);}
    }
    for(const joint of points) {
      if(!joint.position) continue;
      for(const id of joint.entities || []) {
        const target=annotation.entities.find(e=>e.id===id),b=target?.bbox;
        if(!b || target===entity) continue;
        ctx.save();ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(r.x+joint.position[0]*r.w,r.y+joint.position[1]*r.h);ctx.lineTo(r.x+(b[0]+b[2])/2*r.w,r.y+(b[1]+b[3])/2*r.h);ctx.stroke();ctx.restore();
      }
      const [x,y]=joint.position, parent=points.find(p=>p.name===joint.parent);
      if(parent?.position) {ctx.beginPath();ctx.moveTo(r.x+x*r.w,r.y+y*r.h);ctx.lineTo(r.x+parent.position[0]*r.w,r.y+parent.position[1]*r.h);ctx.stroke();}
      ctx.beginPath();ctx.arc(r.x+x*r.w,r.y+y*r.h,5,0,Math.PI*2);ctx.fill();ctx.font='12px system-ui';ctx.fillText(joint.name,r.x+x*r.w+8,r.y+y*r.h-6);
    }
  });
  drawLinkSimulation(r);
  if (gesture?.type==='new') { const [x,y]=gesture.start, [xx,yy]=gesture.end; ctx.strokeStyle='#ffd58a'; ctx.strokeRect(r.x+x*r.w,r.y+y*r.h,(xx-x)*r.w,(yy-y)*r.h); }
  if (gesture?.type==='mask') { ctx.beginPath();gesture.points.forEach((p,i)=>i?ctx.lineTo(r.x+p[0]*r.w,r.y+p[1]*r.h):ctx.moveTo(r.x+p[0]*r.w,r.y+p[1]*r.h));ctx.strokeStyle='#ffd58a';ctx.fillStyle='#ffd58a33';ctx.lineWidth=2;ctx.closePath();ctx.fill();ctx.stroke(); }
}
img.onload = () => { $('dimensions').textContent=`${img.naturalWidth} × ${img.naturalHeight}`; draw(); };
img.onerror = () => { $('dimensions').textContent=''; ctx.clearRect(0,0,canvas.width,canvas.height); message('Image illisible ou format non affichable par ce navigateur.',true); };
new ResizeObserver(draw).observe($('stage'));
canvas.onpointerdown = e => {
  if (busy || !annotation || e.button!==0) return; const p=point(e,false), r=rect(); if(!p) return;
  canvas.focus();
  if(forceMask) { canvas.setPointerCapture(e.pointerId);gesture={type:'mask',points:[p],target:forceMask.target,kind:forceMask.kind};forceMask=null;draw();return; }
  if (skeletonTarget) {
    const {entity, joint, face} = skeletonTarget;
    if (face) {
      const scale=.08, aspect=img.naturalWidth/img.naturalHeight;
      for (const feature of face) feature.joint.position=[Math.max(0,Math.min(1,p[0]+feature.offset[0]*scale)),Math.max(0,Math.min(1,p[1]+feature.offset[1]*scale*aspect))];
    }
    if (annotation.entities.includes(entity) && entity.skeleton?.points.includes(joint)) { joint.position = p; changed(); renderEntities(); }
    skeletonTarget = null; draw(); return;
  }
  for (let i=annotation.entities.length-1; i>=0 && !forceDraw; i--) {
    const entity=annotation.entities[i];
    const joint=entity.skeleton?.points.find(j => j.position && Math.hypot((j.position[0]-p[0])*r.w,(j.position[1]-p[1])*r.h)<9);
    if(joint) { selected=i;gesture={type:'joint',joint,initial:[...joint.position]}; canvas.setPointerCapture(e.pointerId); renderEntities(); draw(); return; }
  }
  for(let i=annotation.entities.length-1;i>=0 && !forceDraw;i--) {
    const entity=annotation.entities[i];
    if(entity.mask?.type==='polygon' && pointInPolygon(p,entity.mask.points)) { selected=i;renderEntities();draw();return; }
  }
  canvas.setPointerCapture(e.pointerId);
  let hit=-1, handle=-1;
  if (!forceDraw && annotation.entities[selected]?.bbox && !annotation.entities[selected]?.mask) handle=handles(annotation.entities[selected].bbox).findIndex(h=>Math.abs(h[0]-p[0])*r.w<9 && Math.abs(h[1]-p[1])*r.h<9);
  if(handle>=0) gesture={type:'resize',start:p,initial:[...annotation.entities[selected].bbox],handle};
  else {
    if(!forceDraw) for(let i=annotation.entities.length-1;i>=0;i--) { const e=annotation.entities[i],b=e.bbox; if(!e.mask && b && p[0]>=b[0] && p[0]<=b[2] && p[1]>=b[1] && p[1]<=b[3]) {hit=i;break;} }
    if(hit>=0) { selected=hit;gesture={type:'move',start:p,initial:[...annotation.entities[hit].bbox]};renderEntities(); }
    else gesture={type:'new',start:p,end:p,target:selected};
  } draw();
};
canvas.onpointermove = e => {
  if(!gesture) return; const p=point(e); if(!p) return;
  if(gesture.type==='joint') gesture.joint.position=p;
  else if(gesture.type==='mask') {
    const last=gesture.points[gesture.points.length-1],r=rect();
    if(Math.hypot((p[0]-last[0])*r.w,(p[1]-last[1])*r.h)>=3)gesture.points.push(p);
  }
  else if(gesture.type==='new') gesture.end=p;
  else {
    const b=[...gesture.initial];
    if(gesture.type==='move') { const dx=Math.max(-b[0],Math.min(1-b[2],p[0]-gesture.start[0])),dy=Math.max(-b[1],Math.min(1-b[3],p[1]-gesture.start[1])); b[0]+=dx;b[2]+=dx;b[1]+=dy;b[3]+=dy; }
    else { const h=gesture.handle, ex=Math.min(.001,(b[2]-b[0])/2), ey=Math.min(.001,(b[3]-b[1])/2); if([0,3,7].includes(h)) b[0]=Math.min(p[0],b[2]-ex); if([1,2,5].includes(h)) b[2]=Math.max(p[0],b[0]+ex); if([0,1,4].includes(h)) b[1]=Math.min(p[1],b[3]-ey); if([2,3,6].includes(h)) b[3]=Math.max(p[1],b[1]+ey); }
    annotation.entities[selected].bbox=b;
  } draw();
};
canvas.onpointerup = e => {
  if(!gesture) return; const g=gesture; gesture=null;
  if(g.type==='joint') { if(JSON.stringify(g.initial)!==JSON.stringify(g.joint.position)) changed(); renderEntities(); }
  else if(g.type==='mask') {
    const p=point(e),last=g.points[g.points.length-1],r=rect();
    if(p && Math.hypot((p[0]-last[0])*r.w,(p[1]-last[1])*r.h)>=1)g.points.push(p);
    setMask(g.points,g.target,g.kind);
  }
  else if(g.type==='new') {
    const [x,y]=g.start,[xx,yy]=g.end, r=rect();
    if(Math.abs(xx-x)*r.w>=4 && Math.abs(yy-y)*r.h>=4) { const box=[Math.min(x,xx),Math.min(y,yy),Math.max(x,xx),Math.max(y,yy)]; if(g.target>=0) {annotation.entities[g.target].bbox=box;changed();renderEntities();} else newEntity(box); }
    forceDraw=false;
  } else if(JSON.stringify(g.initial)!==JSON.stringify(annotation.entities[selected].bbox)) {changed();renderEntities();}
  draw();
};
canvas.onpointercancel = () => { if(gesture?.type==='joint') gesture.joint.position=gesture.initial; else if(gesture && gesture.type!=='new' && gesture.type!=='mask') annotation.entities[selected].bbox=gesture.initial; gesture=null;draw(); };

function addTail(entity) {
  const skeleton = entity.skeleton ||= {name:'personnalisé',points:[]};
  const parts = ['base','segment 1','segment 2','extrémité'];
  let number = 1, names;
  do {
    const prefix = number === 1 ? 'queue' : `queue ${number}`;
    names = parts.map(part => `${prefix} · ${part}`);
    number++;
  } while (names.some(name => skeleton.points.some(point => point.name === name)));
  const anchor = skeleton.points.find(point => ['hanche','bassin','pelvis'].includes(point.name.toLowerCase()))?.name || null;
  names.forEach((name,i) => skeleton.points.push({name,parent:i ? names[i-1] : anchor,position:null}));
  selected = annotation.entities.indexOf(entity);
  skeletonTarget = null; forceDraw = false;
  changed(); renderEntities(); draw();
  message('Queue ajoutée : placez ses quatre points sur l’image. Les noms, segments et attaches restent modifiables.');
}

const jointRoles = ['point', 'tête', 'support', 'saisie', 'articulation majeure', 'détail facial'];
function jointRole(joint) {
  if (joint.role) return joint.role;
  const name=joint.name.toLowerCase();
  if (/^(tête|tete|head)(\s|$)/.test(name)) return 'tête';
  if (/^(main|hand)(\s|$)/.test(name)) return 'saisie';
  if (/^(pied|foot)(\s|$)/.test(name)) return 'support';
  if (/^(hanche|hanches|bassin|pelvis|hip)(\s|$)/.test(name)) return 'articulation majeure';
  return 'point';
}
function placeFace(entity, head) {
  const template=[['œil gauche',[-.25,-.15]],['œil droit',[.25,-.15]],['nez',[0,0]],['bouche',[0,.3]],['oreille gauche',[-.5,0]],['oreille droite',[.5,0]]];
  const face=template.map(([part,offset])=>{
    const name=`${head.name} · ${part}`;
    let joint=entity.skeleton.points.find(p=>p.name===name && p.parent===head.name);
    if(!joint) {
      let unique=name,n=2;while(entity.skeleton.points.some(p=>p.name===unique))unique=`${name} ${n++}`;
      joint={name:unique,parent:head.name,position:null,role:'détail facial'};
      entity.skeleton.points.push(joint);
    }
    return {joint,offset};
  });
  skeletonTarget={entity,joint:head,face};forceDraw=false;changed();renderEntities();
  message('Cliquez au centre du visage pour positionner le calque. Ajustez ensuite chaque œil, oreille, le nez et la bouche en glissant les points.');
}
function renderSkeleton(entity, card) {
  const section=document.createElement('div'); section.className='skeleton';
  if(!entity.skeleton) {
    section.append(button('+ Squelette humain',()=>{
      const names=[['hanche',null],['cou','hanche'],['tête','cou'],['épaule gauche','cou'],['bras gauche','épaule gauche'],['main gauche','bras gauche'],['épaule droite','cou'],['bras droit','épaule droite'],['main droite','bras droit'],['hanche gauche','hanche'],['jambe gauche','hanche gauche'],['pied gauche','jambe gauche'],['hanche droite','hanche'],['jambe droite','hanche droite'],['pied droit','jambe droite']];
      entity.skeleton={name:'humain',points:names.map(([name,parent])=>({name,parent,position:null,role:jointRole({name})}))}; changed();renderEntities();draw();
    }),button('+ Squelette personnalisé',()=>{entity.skeleton={name:'personnalisé',points:[]};changed();renderEntities();}));
  } else {
    const skeleton=entity.skeleton;
    section.append(label('Nom du squelette',input(skeleton.name,v=>{skeleton.name=v;},'Nom du squelette')));
    skeleton.points.forEach((joint,i)=>{
      const row=document.createElement('div');row.className='joint';
      row.append(label('Point nommé',input(joint.name,v=>{const old=joint.name;joint.name=v;for(const p of skeleton.points) if(p.parent===old)p.parent=v;for(const edge of skeleton.edges || []) {if(edge.from===old)edge.from=v;if(edge.to===old)edge.to=v;}draw();},'tête, bras, jambe…')),
        label('Relié à',select(joint.parent || '',skeleton.points.filter(p=>p!==joint).map(p=>p.name),v=>{joint.parent=v || null;draw();},'Point parent')),
        button(joint.position?'Replacer':'Placer',()=>{skeletonTarget={entity,joint};forceDraw=false;message(`Cliquez sur l’image pour placer « ${joint.name} ». Puis glissez le point pour le déplacer.`);}),
        button('×',()=>{skeleton.points.splice(i,1);if(skeleton.edges)skeleton.edges=skeleton.edges.filter(edge=>edge.from!==joint.name && edge.to!==joint.name);for(const p of skeleton.points)if(p.parent===joint.name)p.parent=null;skeletonTarget=null;changed();renderEntities();draw();}));
      const role=jointRole(joint);
      row.append(label('Nature du point',select(role,jointRoles,v=>{joint.role=v;renderEntities();draw();},'Nature du point')));
      if(role==='tête') row.append(button('Positionner le calque du visage',()=>placeFace(entity,joint)));
      const description=document.createElement('small');
      description.textContent={tête:'Tête : yeux, bouche, nez et oreilles.',support:'Membre de support du personnage : associez le sol ou un élément d’appui.',saisie:'Membre de saisie : associez les objets ou éléments tenus.', 'articulation majeure':'Articulation majeure du personnage.', 'détail facial':'Détail du calque facial : position ajustable.'}[role] || '';
      row.append(description);
      const bindings=document.createElement('div');bindings.className='point-bindings';
      const title=document.createElement('small');title.textContent=role==='saisie'?'Objets / éléments saisis':role==='support'?'Sol / éléments de support':'Entités liées à ce point';bindings.append(title);
      for(const target of annotation.entities) {
        const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=(joint.entities || []).includes(target.id);
        checkbox.onchange=()=>{joint.entities=checkbox.checked?[...new Set([...(joint.entities || []),target.id])]:(joint.entities || []).filter(id=>id!==target.id);changed();draw();};
        const l=label(target.id,checkbox);l.className='check';bindings.append(l);
      }
      row.append(bindings,button('Effacer la position',()=>{joint.position=null;changed();renderEntities();draw();}));
      row.querySelector('input').onchange = () => renderEntities();
      const coords=document.createElement('small');coords.textContent=joint.position?joint.position.map(v=>v.toFixed(3)).join(', '):'Non placé';row.append(coords);section.append(row);
    });
    const edgeTitle=document.createElement('h3');edgeTitle.textContent='Liaisons libres entre points';section.append(edgeTitle);
    (skeleton.edges || []).forEach((edge,i)=>{
      const row=document.createElement('div');row.className='record';const names=skeleton.points.map(p=>p.name);
      row.append(select(edge.from,names,v=>{edge.from=v;draw();},'Point de départ'),input(edge.label || '',v=>{edge.label=v;draw();},'Nom de la liaison'),select(edge.to,names,v=>{edge.to=v;draw();},'Point d’arrivée'),button('×',()=>{skeleton.edges.splice(i,1);changed();renderEntities();draw();}));section.append(row);
    });
    const addEdge=button('+ Liaison libre',()=>{(skeleton.edges ||= []).push({from:skeleton.points[0].name,to:skeleton.points[1].name,label:''});changed();renderEntities();draw();});addEdge.disabled=skeleton.points.length<2;section.append(addEdge);
    section.append(button('+ Point nommé' ,()=>{let n=1;while(skeleton.points.some(p=>p.name===`point_${n}`))n++;skeleton.points.push({name:`point_${n}`,parent:null,position:null});changed();renderEntities();}),button('Supprimer le squelette',()=>{if(!confirm('Supprimer ce squelette ?'))return;delete entity.skeleton;skeletonTarget=null;changed();renderEntities();draw();}));
  }
  section.append(button('+ Queue', () => addTail(entity)));
  card.append(section);
}
