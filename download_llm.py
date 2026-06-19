from huggingface_hub import snapshot_download

print("Downloading INT4 Compressed LLM for Edge Inference...")
model_dir = snapshot_download(
    repo_id="OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
    local_dir="tinyllama_int4"
)
print("Download complete. Model saved to: tinyllama_int4/")