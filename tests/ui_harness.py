"""Local UI fixture. Run with a Python containing Pillow/torch, then visit :8765.
This tests browser interaction only; it is not a ComfyUI server.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from test_layers import LayersTests, nodes

project=LayersTests().project()
payload=nodes.ui_payload(project,project['state'])['layers_project'][0]
ROOT=Path(__file__).resolve().parents[1]
APP='''export const app={
 extensionManager:{toast:{add:x=>alert(x.detail)}},
 queuePrompt:async()=>{document.querySelector('#result').textContent='Applied '+JSON.parse(window.testNode.widgets[0].value).layers.length+' layers';},
 registerExtension:async extension=>{
 class Node {constructor(){this.widgets=[{name:'editor_state',value:''}];this.graph={setDirtyCanvas(){}};}addDOMWidget(name,type,element,options){document.body.append(element);return this.addWidget(type,name,null,null);}addWidget(type,name,value,callback){const w={type,name,value,callback};this.widgets.push(w);return w;}}
 await extension.beforeRegisterNodeDef(Node,{name:'LayersEditor'});
 const node=new Node();node.onNodeCreated();node.onExecuted({layers_project:[await (await fetch('/fixture.json')).json()]});window.testNode=node;
 const b=document.createElement('button');b.textContent='Open layer editor';b.onclick=node.widgets.find(w=>w.name==='Open layer editor').callback;document.body.append(b);
 }};'''
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/':body=b'<html><body style="background:#0f172a;color:white;font:16px system-ui"><h1>Layers UI test fixture</h1><p id="result">Not applied</p><script type="module" src="/extensions/layers/layers.js"></script></body></html>';mime='text/html'
  elif self.path=='/fixture.json':body=json.dumps(payload).encode();mime='application/json'
  elif self.path=='/scripts/app.js':body=APP.encode();mime='text/javascript'
  elif self.path in ['/extensions/layers/layers.js','/extensions/layers/math.js','/extensions/layers/zoom.js','/extensions/layers/editor_status.js','/extensions/layers/layers.css']:
   path=ROOT/'web'/self.path.rsplit('/',1)[1];body=path.read_bytes();mime='text/css' if path.suffix=='.css' else 'text/javascript'
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',mime);self.end_headers();self.wfile.write(body)
if __name__=='__main__':HTTPServer(('127.0.0.1',8765),Handler).serve_forever()
