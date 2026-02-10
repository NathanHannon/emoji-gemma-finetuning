import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest';

// Skip local check since we will be fetching from HF Hub
env.allowLocalModels = false;
// env.allowLocalModels = true;

// Define the model ID
const MODEL_ID = 'NathanHannon/emoji_gemma3.270m';
// FOR TESTING LOCAL: Points to the folder in the root directory (served by python server)
// const MODEL_ID = '../onnx_output_dir';

class PipelineSingleton {
    static task = 'text-generation';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = await pipeline(this.task, MODEL_ID, {
                progress_callback,
                // Use quantized model for performance and lower memory usage!
                dtype: 'q8',
                // device: 'webgpu', // Try WebGPU first, fall back to wasm
            });
        }
        return this.instance;
    }
}

// Message handler
self.addEventListener('message', async (event) => {
    const { text, type } = event.data;

    if (type === 'load') {
        try {
            await PipelineSingleton.getInstance((data) => {
                // Relay progress back to main thread
                self.postMessage({
                    type: 'progress',
                    data
                });
            });
            self.postMessage({ type: 'ready' });
        } catch (e) {
            console.error("Loading Error:", e);
            self.postMessage({ type: 'error', error: e.message || e.toString() });
        }
        return;
    }

    if (type === 'generate') {
        try {
            const translator = await PipelineSingleton.getInstance();

            // Format input as per Gemma chat template if needed, 
            // or just raw text depending on how we trained it.
            // Since we trained Role-based, let's try to mimic that simple format or just raw prompt
            // Assuming the model expects: 
            // <start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n

            // NOTE: Reverting to manual prompt construction because the ONNX export 
            // didn't include the chat_template in tokenizer_config.json automatically.
            const prompt = `<start_of_turn>user\n${text}<end_of_turn>\n<start_of_turn>model\n`;

            const output = await translator(prompt, {
                max_new_tokens: 20,
                do_sample: true,
                temperature: 0.6,
            });

            // The output is just the generated object
            let generatedText = output[0].generated_text;

            // CLEANUP LOGIC:
            // 1. Remove the prompt if the model returned the full text (standard behavior)
            // We strip valid special tokens or just the raw text "user...model" seen in the screenshot
            const promptRegex = /user\s+.*?\s+model\s*/si;

            if (generatedText.match(promptRegex)) {
                generatedText = generatedText.replace(promptRegex, '');
            } else if (generatedText.includes(prompt)) {
                generatedText = generatedText.replace(prompt, '');
            }

            // 2. Remove any remaining structural tokens
            generatedText = generatedText.replace(/<start_of_turn>/g, '')
                .replace(/<end_of_turn>/g, '')
                .replace(/model\n/g, '') // extra safety
                .trim();
            output: generatedText
        }
        catch (e) {
            console.error(e); // Log full error to console
            self.postMessage({ type: 'error', error: e.toString() });
        }
    }
});
