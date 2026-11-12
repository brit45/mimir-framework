// Browser smoke: node Tests/test_organic_network_browser.mjs <generated fixture directory>
import {spawn} from 'node:child_process';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const digest=value=>createHash('sha256').update(value).digest('hex');
const root=resolve(process.argv[2]);
const profile=await mkdtemp(join(tmpdir(),'mimir-organic-browser-'));
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn){for(let i=0;i<100;i++){const x=await fn();if(x)return x;await delay(100)}throw Error('timeout')}
let browser,ws,logs='',exceptions=[];
try{
 browser=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank']);
 browser.stderr.on('data',d=>logs+=d);
 const port=await until(()=>logs.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/)?.[1]);
 const page=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(pathToFileURL(join(root,'reseau-organique.html')).href)}`,{method:'PUT'})).json();
 ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});
 let id=0;const pending=new Map();
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params)};
 const cdp=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
 const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 await cdp('Runtime.enable');await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
 await until(()=>evaluate(`document.getElementById('camera')?.textContent.includes('liens affichés')`));
 await evaluate(`document.fonts.ready`);await delay(250);
 assert(await evaluate(`document.getElementById('organic').checked`));
 const before=await evaluate(`document.querySelector('canvas').toDataURL()`);
 await evaluate(`document.getElementById('organic').click()`);await delay(100);
 const after=await evaluate(`document.querySelector('canvas').toDataURL()`);assert.notEqual(before,after);
 await evaluate(`document.getElementById('organic').click()`);await delay(100);
 assert.equal(digest(await evaluate(`document.querySelector('canvas').toDataURL()`)),digest(before),'Organic pixels did not restore after the style toggle');
 assert(await evaluate(`document.getElementById('axes').textContent.includes('volumes corticaux')`));
 const heatBefore=await evaluate(`document.querySelector('canvas').toDataURL()`);
 await evaluate(`document.getElementById('heatspots').click()`);await delay(100);
 assert.notEqual(digest(await evaluate(`document.querySelector('canvas').toDataURL()`)),digest(heatBefore),'Heatspot toggle did not change pixels');
 await evaluate(`document.getElementById('heatspots').click()`);await delay(100);
 assert.equal(digest(await evaluate(`document.querySelector('canvas').toDataURL()`)),digest(heatBefore),'Heatspot pixels did not restore');
 const screenshot=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(join(root,'reseau-organique.png'),Buffer.from(screenshot.data,'base64'));
 assert.equal(exceptions.length,0,JSON.stringify(exceptions));
 console.log('PASS Chromium: actual canvas, organic/classic toggle, deterministic restoration, cortical legend, no JS errors, screenshot');
}finally{ws?.close();if(browser&&browser.exitCode===null){browser.kill();await delay(400)}await rm(profile,{recursive:true,force:true,maxRetries:5,retryDelay:100})}
