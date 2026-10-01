import test from 'node:test';
import assert from 'node:assert/strict';
import { hideEditorState, editorStatus } from '../web/editor_status.js';

test('hides internal JSON without losing serialized mask edits', () => {
    const state = JSON.stringify({layers:[{id:'layer-0',mask:'saved pixels'}]});
    const widget = {name:'editor_state',value:state,options:{serialize:true},inputEl:{style:{}},element:{style:{}}};
    hideEditorState({widgets:[widget]});
    assert.equal(widget.hidden,true);
    assert.equal(widget.inputEl.style.display,'none');
    assert.equal(widget.element.style.display,'none');
    assert.equal(widget.value,state);
    assert.equal(widget.options.serialize,true);
    assert.deepEqual(widget.computeSize(),[0,-4]);
    hideEditorState({widgets:[widget]});
    assert.equal(widget.value,state);
});
test('review status explains blank downstream output and the next action', () => {
    const text = editorStatus({review_required:true,state:{layers:[{name:'person'},{name:'chair'}]}});
    assert.match(text,/2 layers: person, chair/);
    assert.match(text,/Paused for review/);
    assert.match(text,/Apply & Run/);
    assert.doesNotMatch(text,/base64/);
});
test('completed execution is not incorrectly labeled paused', () => {
    assert.match(editorStatus({state:{layers:[]},review_required:false}),/Layers ready/);
    assert.match(editorStatus(null),/Run once/);
});
