# Smart Ag Vision Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Intel OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-informational.svg)](https://docs.openvino.ai/)

Hardware-accelerated, dual-model Edge AI pipeline engineered for real-time agricultural analytics, leaf diagnostics, and crop health monitoring.

## Overview

Modern precision agriculture requires instantaneous, on-site diagnostics without relying on cloud connectivity. The Smart Ag Vision Engine resolves this by bringing high-performance AI inference directly to the edge. By combining computer vision with a localized small language model (SLM), the system detects agricultural anomalies and explains them in natural language, suggesting actionable mitigation strategies.

## System Architecture

The pipeline leverages the Intel OpenVINO Toolkit to distribute computational workloads across the Neural Processing Unit (NPU) and integrated GPU, effectively bypassing CPU bottlenecks and minimizing thermal output.

```text
[ Multi-Modal Input ]
        |
        +---> (Leaf Image) -----> [ YOLOv8 Vision Engine ] ------> (Bounding Boxes & Class)
        |                              (OpenVINO IR)                     |
        |                                                                v
        +---> (Text Query) -----> [ TinyLlama-1.1B Engine ] <--- (Context Injection)
                                       (INT4 NNCF)                       |
                                                                         v
[ Gradio Edge Dashboard ] <-------------------------------------- [ Diagnostic Output ]
1. Vision Engine (YOLOv8)
Function: Analyzes high-resolution leaf samples to detect structural anomalies, pests, and localized necrotic spots (e.g., blight, rust).

Optimization: Converted to OpenVINO Intermediate Representation (IR) format for NPU/iGPU execution.

2. Language Engine (TinyLlama-1.1B)
Function: Acts as a localized AI agricultural agent, processing visual detection tokens and user prompts to provide farming guidance.

Optimization: Utilizes INT4 weight compression via OpenVINO's Neural Network Compression Framework (NNCF) to drastically reduce memory bandwidth requirements.

Repository Structure
The repository contains the complete pipeline for training, compiling, and deploying the models. Heavy dataset folders and raw model weights are strictly ignored to maintain a lightweight codebase.

Plaintext
smart-ag-vision/
├── edge_server.py           # Core orchestrator and web interface
├── setup_board.sh           # Edge environment initialization script
├── compile_model.py         # OpenVINO IR conversion (.pt -> .xml/.bin)
├── train_model.py           # YOLOv8 baseline training script
├── train_50_epochs.py       # YOLOv8 extended fine-tuning script
├── test_inference.py        # Static image latency benchmarking
├── live_inference.py        # Real-time webcam pipeline test
├── download_data.py         # Dataset retrieval utility
├── download_llm.py          # TinyLlama HuggingFace retrieval utility
├── requirements.txt         # Strict Python dependencies
└── .gitignore               # Configured to exclude heavy models and datasets
Quick Start
Prerequisites
OS: Ubuntu 22.04 LTS (or Windows WSL2 equivalent)

Hardware: Intel Core Ultra processor (Target: Intel DK-2500 NPU)

Dependencies: Python 3.10+, Intel NPU Drivers installed

1. Installation
Clone the repository and set up your virtual environment:

Bash
git clone [https://github.com/PrabhanshuChouhan/smart-ag-vision.git](https://github.com/PrabhanshuChouhan/smart-ag-vision.git)
cd smart-ag-vision

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2. Execution Pipeline
Follow these steps to prepare the hardware-accelerated models and launch the edge server:

Step A: Fetch Data and Base Models

Bash
python3 download_data.py
python3 download_llm.py
Step B: Compile Models for the NPU
Converts standard PyTorch models into optimized OpenVINO IR files.

Bash
python3 compile_model.py
Step C: Launch the Edge Dashboard

Bash
python3 edge_server.py
3. Network Access
Once the edge_server.py script is running, the interface binds to 0.0.0.0. It can be accessed securely from any device on the same local network by navigating to the edge board's IP address:

Plaintext
http://<BOARD_IP_ADDRESS>:7860
