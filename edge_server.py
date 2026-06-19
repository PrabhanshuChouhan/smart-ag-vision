import gradio as gr
from ultralytics import YOLO
import openvino_genai as ov_genai

print("Loading Vision Model (YOLOv8 OpenVINO)...")
# Change this line to point to your NEW train-4 folder
vision_model = YOLO("runs/classify/train-4/weights/best_openvino_model/")
print("Loading Language Model (TinyLlama INT4 OpenVINO)...")
# Loading the LLM onto the CPU. On the DK-2500 board, you can change this to "GPU"
llm_pipeline = ov_genai.LLMPipeline("tinyllama_int4", "CPU")

def autonomous_agronomist(image):
    # 1. Vision AI: Analyze the image
    results = vision_model(image, verbose=False)
    
    top_class_id = results[0].probs.top1
    disease_name = results[0].names[top_class_id].replace("_", " ")
    confidence = results[0].probs.top1conf.item() * 100
    
    vision_output = f"Detection: {disease_name}\nConfidence: {confidence:.2f}%"

    # 2. Generative AI: Write the treatment plan dynamically
    prompt = (
        f"<|system|>\nYou are an expert agronomist. Keep your advice brief, professional, and actionable.</s>\n"
        f"<|user|>\nA farmer just scanned their crop and the AI detected {disease_name}. "
        f"Provide a 3-step action plan to treat this issue.</s>\n<|assistant|>\n"
    )

    print(f"Generating treatment plan for {disease_name}...")
    llm_output = llm_pipeline.generate(prompt, max_new_tokens=150)

    return vision_output, llm_output

# Build the Web Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Smart Ag Vision: Autonomous Edge Agronomist")
    gr.Markdown("Powered by Intel Core Ultra: Dual-Model Execution (Vision + GenAI)")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Leaf Photo")
            submit_btn = gr.Button("Analyze Crop", variant="primary")
        
        with gr.Column():
            vision_output = gr.Textbox(label="Vision AI Diagnostics (YOLOv8)")
            llm_output = gr.Textbox(label="Generative AI Agronomist (TinyLlama INT4)", lines=8)

    submit_btn.click(fn=autonomous_agronomist, inputs=image_input, outputs=[vision_output, llm_output])

if __name__ == "__main__":
    print("Starting Dual-Model Edge Server...")
    demo.launch(server_name="0.0.0.0", server_port=7860)