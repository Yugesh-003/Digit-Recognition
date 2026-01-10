from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def center_digit(image):
    """Center digit using center of mass and crop to bounding box"""
    # Find bounding box of non-zero pixels
    rows = np.any(image > 0.2, axis=1)
    cols = np.any(image > 0.2, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return image  # Return original if no content
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract digit region
    digit = image[rmin:rmax+1, cmin:cmax+1]
    
    # Calculate size and padding for centering
    h, w = digit.shape
    
    # Create square canvas large enough to fit digit
    canvas_size = max(h, w) 
    canvas = np.zeros((canvas_size, canvas_size))
    
    # Center the digit
    start_r = (canvas_size - h) // 2
    start_c = (canvas_size - w) // 2
    canvas[start_r:start_r+h, start_c:start_c+w] = digit
    
    # Resize to 28x28 using area interpolation
    from scipy.ndimage import zoom
    scale = 28.0 / canvas_size
    resized = zoom(canvas, scale, order=1)  # Bilinear interpolation
    
    # Apply light Gaussian blur to match MNIST stroke style
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(resized, sigma=0.5)
    
    return blurred

def load_model_():
    global model
    try:
        model=load_model('model.h5')
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
        
        image_array = np.array(image_data, dtype=np.float32).reshape(28, 28)

        # Normalize ONLY
        image_array /= 255.0
        image_array = np.clip(image_array, 0.0, 1.0)

        # Center and crop digit to match MNIST spatial alignment
        image_array = center_digit(image_array)

        # Reshape
        image_array = image_array.reshape(1, 28, 28, 1)

        # print("min:", image_array.min(), "max:", image_array.max())

        print(f"Image shape: {image_array.shape}")
        print(f"Image min/max: {image_array.min():.3f}/{image_array.max():.3f}")
        
        ink = np.sum(image_array > 0.1)
        if ink < 20:
            return jsonify({
                'prediction': None,
                'confidence': 0.0,
                'message': "No digit detected",
                'probabilities':0.0
            }) 
        
        
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
    load_model_()
    
    if model is None:
        print("ERROR: Model could not be loaded!")
        exit(1)
    
    print("Starting Flask server...")
    app.run(debug=True, port=5000)