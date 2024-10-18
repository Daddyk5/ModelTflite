import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('my_model.h5')

# Convert the model to TensorFlow Lite format with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations (weight and integer quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optionally, define representative dataset function for more aggressive quantization
def representative_dataset():
    # Replace this with a few input samples of your data
    for _ in range(100):
        yield [tf.random.normal([1, 224, 224, 3])]  # Example: image of shape (1, 224, 224, 3)

# Enable float16 or integer quantization
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.float16]

# Convert the model
tflite_model = converter.convert()

# Save the optimized TFLite model
with open('my_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully optimized and saved as 'my_model_optimized.tflite'")

