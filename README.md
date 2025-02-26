🚀 AI-Based Logic Depth Predictor

📌 Overview

This repository contains an AI-powered tool to predict combinational logic depth of signals in an RTL module before synthesis. The model eliminates the need for time-consuming synthesis runs while maintaining high accuracy, enabling faster timing closure and design optimization.

🛠 Features

Graph-based RTL parsing for extracting signal dependencies.

Deep Learning (TensorFlow DNN) for fast and accurate combinational depth prediction.

Trained on OpenCores & FPGA/ASIC synthesis reports.

Detects timing violations without requiring full synthesis.

📊 Machine Learning Model Used

Model: TensorFlow Deep Neural Network (DNN)

Training Data: Extracted RTL features (Fan-in, Fan-out, Path Depth)

Training Size: 80% train / 20% test split

Evaluation Metrics:

Mean Absolute Error (MAE): ≤ 0.5

R² Score: ≥ 0.90

📌 Installation

Run the following command to install the required dependencies:
pip install tensorflow pandas numpy networkx pyverilog scikit-learn
🚀 Usage

1️⃣ Clone the Repository
git clone https://github.com/yourusername/GH-2025-Logic-Depth.git
cd GH-2025-Logic-Depth
2️⃣ Prepare RTL File

Place your Verilog RTL file in the same directory, or use the provided example_rtl.v.

3️⃣ Run the Predictor
python train_logic_depth_predictor.py
4️⃣ Expected Output

Extracted RTL Features (Fan-in, Fan-out, Logic Depth)

Model Performance Metrics (MAE, R² Score)

Trained Model Saved as tf_logic_depth_model.h5

📂 Sample Dataset

A synthetic dataset is included for training & testing.

Dataset File: dataset/logic_depth_dataset.csv


