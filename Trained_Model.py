import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("tf_logic_depth_model.h5", custom_objects={'mse': tf.keras.losses.mse}) # Add custom_objects

# Example new data for prediction (Fan-in, Fan-out)
new_data = np.array([[3, 2],  # Example 1
                     [1, 4],  # Example 2
                     [2, 5]]) # Example 3

# Make predictions
predictions = model.predict(new_data)
print("\n🔍 Predicted Logic Depths:", predictions.flatten())
