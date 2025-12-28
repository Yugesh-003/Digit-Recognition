from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def create_model():
    """Create improved model architecture with dropout"""
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=(784,), activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_weights():
    global model
    print("Training improved model...")
    
    # Load MNIST data
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize and flatten data
    x_train_flat = x_train.reshape(len(x_train), 28*28) / 255.0
    x_test_flat = x_test.reshape(len(x_test), 28*28) / 255.0
    
    # Create and train model
    model = create_model()
    
    # Train with more epochs
    model.fit(x_train_flat, y_train, 
              epochs=12, 
              batch_size=32,
              validation_data=(x_test_flat, y_test),
              verbose=1)
    
    # Test accuracy
    test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
    print(f"Final test accuracy: {test_acc:.4f}")
    
    # Save weights
    model.save_weights('mnist_weights.weights.h5')
    print("Improved model weights saved")

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
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json
        image_data = data['image']
        
        # Convert to numpy array
        image_array = np.array(image_data, dtype=np.float32).reshape(28, 28)
        
        # Center the digit using center of mass
        if np.sum(image_array) > 0:
            # Find center of mass
            y_coords, x_coords = np.where(image_array > 0.1)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # Calculate shift to center (14,14 is center of 28x28)
                shift_x = 14 - center_x
                shift_y = 14 - center_y
                
                # Create centered image
                centered_image = np.zeros((28, 28))
                for y in range(28):
                    for x in range(28):
                        new_y = y - shift_y
                        new_x = x - shift_x
                        if 0 <= new_y < 28 and 0 <= new_x < 28:
                            centered_image[y, x] = image_array[new_y, new_x]
                
                image_array = centered_image
        
        # Normalize intensity
        if np.max(image_array) > 0:
            image_array = image_array / np.max(image_array)
        
        # Apply slight Gaussian blur for smoothing
        from scipy.ndimage import gaussian_filter
        image_array = gaussian_filter(image_array, sigma=0.5)
        
        # Reshape for model input
        image_array = image_array.reshape(1, 784)
        
        print(f"Image shape: {image_array.shape}")
        print(f"Image min/max: {image_array.min():.3f}/{image_array.max():.3f}")
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        print(f"Predicted: {predicted_digit}, Confidence: {confidence:.1f}%")
        print(f"All probabilities: {[f'{p:.3f}' for p in predictions[0]]}")
        
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