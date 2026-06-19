#!/bin/bash

echo "========================================================="
echo "        Smart Ag Vision - Edge Hardware Setup   "
echo "  Target Architecture: Intel DK-2500 Board (Ubuntu)      "
echo "========================================================="

# Step 1: Secure system dependencies
echo "[1/4] Updating system repositories..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install python3-venv python3-pip git wget -y

# Step 2: Establish the isolated environment
echo "[2/4] Initializing Python Virtual Environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

# Step 3: Install Vision AI stack (YOLOv8 + OpenVINO + Gradio)
echo "[3/4] Installing Vision dependencies and Intel OpenVINO..."
pip install --upgrade pip
pip install ultralytics openvino gradio
# Installing CPU-specific PyTorch to keep the edge footprint incredibly lightweight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install Local LLM stack (TinyLlama INT4 requirements)
echo "[4/4] Installing LLM dependencies (Transformers/Accelerate)..."
pip install transformers accelerate

echo "========================================================="
echo "  SYSTEM READY.                                          "
echo "                                                         "
echo "  To launch the autonomous agronomist pipeline:          "
echo "  1. source .venv/bin/activate                           "
echo "  2. python3 edge_server.py                              "
echo "========================================================="