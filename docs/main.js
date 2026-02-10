// Main thread logic
const worker = new Worker('worker.js', { type: 'module' });

const elements = {
    input: document.getElementById('text-input'),
    btn: document.getElementById('emojify-btn'),
    status: document.getElementById('status'),
    outputContainer: document.getElementById('output-container'),
    outputBox: document.getElementById('output-text'),
    progressContainer: document.getElementById('progress-container'),
    progressBar: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
};

// Check for WebGPU support
if (!navigator.gpu) {
    console.warn("WebGPU not available, falling back to WASM/CPU. This may be slower.");
}

// Start loading the model immediately
worker.postMessage({ type: 'load' });

// Message handling
worker.addEventListener('message', (event) => {
    const { type, data, error, output } = event.data;

    switch (type) {
        case 'progress':
            // data matches: { status: 'progress', file: '...', progress: 0-100, ... }
            if (data.status === 'progress') {
                elements.progressContainer.classList.remove('hidden');
                elements.progressBar.style.width = `${data.progress}%`;
                elements.progressText.textContent = `Loading ${data.file}... (${Math.round(data.progress)}%)`;
            } else if (data.status === 'done') {
                // One file done
                elements.progressText.textContent = `Loaded ${data.file}`;
            } else if (data.status === 'initiate') {
                elements.progressContainer.classList.remove('hidden');
                elements.progressText.textContent = `Downloading ${data.file}...`;
            }
            break;

        case 'ready':
            elements.progressContainer.classList.add('hidden');
            elements.status.textContent = "Model loaded & ready!";
            elements.status.style.color = "green";
            elements.btn.disabled = false;
            elements.btn.textContent = "Emojify! ✨";
            break;

        case 'update':
            // Streaming token received
            // If this is the FIRST token, reveal values and clear box
            if (elements.outputContainer.classList.contains('hidden')) {
                elements.outputContainer.classList.remove('hidden');
                elements.outputBox.textContent = "";
            }
            // Append token
            // FIX: 'output' is destructured directly from event.data, not 'data.output'
            elements.outputBox.textContent += output;
            break;

        case 'complete':
            elements.btn.disabled = false;
            elements.btn.textContent = "Emojify! ✨";
            // Check if output was sent in complete (legacy) or handled by stream
            if (output) {
                elements.outputContainer.classList.remove('hidden');
                elements.outputBox.textContent = output;
            }
            break;

        case 'error':
            elements.progressContainer.classList.add('hidden');
            elements.status.textContent = "Error: " + error;
            elements.status.style.color = "red";
            elements.btn.disabled = false;
            // Also alert for visibility if needed
            console.error(error);
            break;
    }
});

// UI Interactions
elements.btn.addEventListener('click', () => {
    const text = elements.input.value.trim();
    if (!text) return;

    elements.btn.disabled = true;
    elements.btn.textContent = "Thinking... 🤔";
    elements.outputContainer.classList.add('hidden');

    worker.postMessage({ type: 'generate', text });
});

// Enable 'Enter' key to submit
elements.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.btn.click();
    }
});
