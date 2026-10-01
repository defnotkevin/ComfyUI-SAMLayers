// Keep the backend/workflow value intact while hiding implementation data.
export function hideEditorState(node) {
    const widget = node.widgets?.find(w => w.name === 'editor_state');
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.inputEl) widget.inputEl.style.display = 'none';
    if (widget.element) widget.element.style.display = 'none';
    // Do not disable serialization: masks and transforms must survive saving.
}

export function editorStatus(payload) {
    if (!payload) return 'Run once to detect objects, then open the layer editor.';
    const layers = payload.state?.layers || [];
    const names = layers.map(layer => layer.name).join(', ');
    const action = payload.review_required
        ? 'Paused for review. Open layer editor, then Apply & Run to render and save.'
        : 'Layers ready. Open layer editor to make further changes.';
    return `${layers.length} layers: ${names}\n${action}`;
}
