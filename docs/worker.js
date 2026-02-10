import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest';

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
                // Since files are at the root, we rely on standard name mapping or manual override
                // file: 'model_quantized.onnx', 
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

            // We need to inject the system prompt because the model was trained with it!
            // Based on the training script (train.py) and standard Gemma templates, 
            // the system prompt is prepended to the user's first message.
            const systemPrompt = "Translate this text to emoji: ";

            // Validating the exact format: "System Prompt\n\nUser Text"
            const prompt = `<start_of_turn>user\n${systemPrompt}\n\n${text}<end_of_turn>\n<start_of_turn>model\n`;

            console.log("Generating with prompt:", prompt);

            // Create a streamer to send tokens back as they are generated
            const streamer = new TextStreamer(translator.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true,
                callback_function: (token) => {
                    console.log("Token:", token);
                    self.postMessage({ type: 'update', output: token });
                }
            });

            const output = await translator(prompt, {
                max_new_tokens: 20,
                do_sample: true,
                temperature: 0.6,
                streamer,
            });
            console.log("Full output:", output);

            self.postMessage({
                type: 'complete',
                output: null
            });
        }
        catch (e) {
            console.error(e); // Log full error to console
            self.postMessage({ type: 'error', error: e.toString() });
        }
    }
});
