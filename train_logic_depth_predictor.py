import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import networkx as nx
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -------------------------
# Step 1: Parse RTL and Build Logic Graph
# -------------------------
def extract_signals_from_verilog(file_path):
    """ Parses Verilog RTL code to extract Fan-in, Fan-out, and combinational paths. """
    with open(file_path, 'r') as file:
        code = file.readlines()

    signal_graph = nx.DiGraph()

    for line in code:
        line = line.strip()
        if line.startswith("//") or line == "":
            continue

        # Extract wire and reg signals
        if re.match(r"(wire|reg)\s+(\w+);", line):
            signal = re.findall(r"(wire|reg)\s+(\w+);", line)[0][1]
            signal_graph.add_node(signal)

        # Extract assignments (logic depth mapping)
        elif "assign" in line:
            parts = line.replace(";", "").split("=")
            if len(parts) == 2:
                left_signal = parts[0].strip()
                right_signals = re.findall(r"\b\w+\b", parts[1])
                for sig in right_signals:
                    if sig != left_signal:
                        signal_graph.add_edge(sig, left_signal)

    return signal_graph

# -------------------------
# Step 2: Extract Features
# -------------------------
def extract_features(signal_graph):
    """ Extracts Fan-in, Fan-out, and Logic Depth features from the graph """
    features = []

    for node in signal_graph.nodes():
        fan_in = len(list(signal_graph.predecessors(node)))
        fan_out = len(list(signal_graph.successors(node)))
        try:
            logic_depth = nx.shortest_path_length(signal_graph, source=list(signal_graph.nodes())[0], target=node)
        except nx.NetworkXNoPath:
            logic_depth = 0  # Assign default depth if no path exists

        features.append({
            'Signal': node,
            'Fan-in': fan_in,
            'Fan-out': fan_out,
            'Logic Depth': logic_depth
        })

    return pd.DataFrame(features)

# -------------------------
# Step 3: Train Deep Learning Model (TensorFlow)
# -------------------------
def train_logic_depth_predictor(dataset):
    """ Trains a Deep Learning model using TensorFlow to predict logic depth """
    X = dataset[['Fan-in', 'Fan-out']].values
    y = dataset['Logic Depth'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test), verbose=1)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"\n✅ Model Training Completed!\nMean Absolute Error: {mae:.4f}")

    model.save("tf_logic_depth_model.h5")
    print("\n💾 Model saved as 'tf_logic_depth_model.h5'")

    return model

# -------------------------
# Step 4: High-Speed Execution
# -------------------------
if __name__ == "__main__":
    verilog_file = "example_rtl.v"
    signal_graph = extract_signals_from_verilog(verilog_file)
    dataset = extract_features(signal_graph)
    print("\nExtracted Features:\n", dataset)
    model = train_logic_depth_predictor(dataset)
    print("\n🚀 Training Complete! Model is ready for predictions.")
