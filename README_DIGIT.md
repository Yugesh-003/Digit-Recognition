# Digit Recognition Setup

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Start Flask Server
```bash
python app.py
```

## 3. Open HTML File
Open `digit_canvas.html` in your browser

## How it Works
1. The Flask server loads/trains the MNIST model from your notebook
2. Draw a digit in the canvas
3. Click "Predict" to send the image to the Flask API
4. The trained model returns the prediction and confidence

The model architecture matches your notebook:
- Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(10, softmax)
- Trained on MNIST dataset for 5 epochs