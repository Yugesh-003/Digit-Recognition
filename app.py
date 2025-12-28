from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def create_model():
    """Create model architecture matching your notebook"""
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(784,), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_weights():
    global model
    print("Training new model...")
    
    # Load MNIST data
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize data (CRITICAL: same as prediction input)
    x_train_flat = x_train.reshape(len(x_train), 28*28) / 255.0
    
    # Create and train model
    model = create_model()
    model.fit(x_train_flat, y_train, epochs=5, verbose=1)
    
    # Test accuracy
    x_test_flat = x_test.reshape(len(x_test), 28*28) / 255.0
    test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save only weights
    model.save_weights('mnist_weights.weights.h5')
    print("Weights saved")

def load_model_weights():
    global model
    try:
        model = create_model()
        model.load_weights('mnist_weights.weights.h5')
        print("Model weights loaded")
        return True
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    print("hehe")
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json
        image_data = data['image']
        
        # Convert to numpy array and normalize (0-1 range)
        image_array = np.array(image_data, dtype=np.float32)
        
        
        # Reshape for model input
        image_array = image_array.reshape(1, 784)
        
        print(f"Image shape: {image_array.shape}")
        print(f"Image min/max: {image_array.min():.3f}/{image_array.max():.3f}")
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        print(f"Predicted: {predicted_digit}, Confidence: {confidence:.1f}%")
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': predictions[0].tolist()
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load weights, if not found, train new model
    if not load_model_weights():
        train_and_save_weights()
    
    if model is None:
        print("ERROR: Model could not be loaded!")
        exit(1)
    
    print("Starting Flask server...")
    app.run(debug=True, port=5000)