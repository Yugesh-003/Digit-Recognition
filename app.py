from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def create_and_train_model():
    global model
    print("Creating and training new model...")
    
    # Load MNIST data
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten data
    x_train_flat = x_train.reshape(60000, 784)
    x_test_flat = x_test.reshape(10000, 784)
    
    # Create model (same as your notebook)
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=(784,), activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    print("Training model...")
    model.fit(x_train_flat, y_train, epochs=5, batch_size=32, verbose=1)
    
    # Save model
    model.save('mnist_model.h5')
    print("Model saved as mnist_model.h5")
    
    return model

def load_model_file():
    global model
    try:
        model = keras.models.load_model('mnist_model.h5')
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json
        image_data = data['image']
        
        # Convert to numpy array and normalize
        image_array = np.array(image_data, dtype=np.float32)
        
        # Invert colors (canvas is white on black, MNIST is black on white)
        image_array = 1.0 - image_array
        
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
    # Try to load existing model, if not found, create and train new one
    if not load_model_file():
        create_and_train_model()
    
    if model is None:
        print("ERROR: Model could not be loaded or created!")
        exit(1)
    
    print("Starting Flask server...")
    app.run(debug=True, port=5000)