import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variable to store the model
model = None

# HTML template embedded in Python for single-file deployment
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Digit Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            text-align: center;
            max-width: 500px;
            width: 100%;
        }

        h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .canvas-container {
            position: relative;
            display: inline-block;
            margin-bottom: 30px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }

        #drawingCanvas {
            border: 3px solid #ddd;
            cursor: crosshair;
            background: white;
            display: block;
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        button {
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .clear-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
        }

        .clear-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(238, 90, 82, 0.4);
        }

        .predict-btn {
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            color: white;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(68, 160, 141, 0.4);
        }

        .predict-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .result {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            padding: 20px;
            border-radius: 15px;
            transition: all 0.3s ease;
        }

        .result.show {
            background: linear-gradient(45deg, #a8edea, #fed6e3);
            color: #333;
            transform: scale(1.05);
        }

        .loading {
            display: none;
            color: #667eea;
            font-style: italic;
        }

        .instructions {
            color: #666;
            margin-bottom: 20px;
            font-size: 16px;
            line-height: 1.5;
        }

        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            #drawingCanvas {
                width: 250px;
                height: 250px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ Digit Recognition</h1>
        <p class="instructions">Draw a digit (0-9) in the box below and click predict!</p>
        
        <div class="canvas-container">
            <canvas id="drawingCanvas" width="300" height="300"></canvas>
        </div>
        
        <div class="controls">
            <button id="clearBtn" class="clear-btn">Clear</button>
            <button id="predictBtn" class="predict-btn">Predict</button>
        </div>
        
        <div id="loading" class="loading">🤖 Analyzing your digit...</div>
        <div id="result" class="result"></div>
    </div>

    <script>
        const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        const clearBtn = document.getElementById('clearBtn');
        const predictBtn = document.getElementById('predictBtn');
        const resultDiv = document.getElementById('result');
        const loadingDiv = document.getElementById('loading');

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        // Set up canvas
        ctx.strokeStyle = '#000';
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.lineWidth = 8;

        // Drawing functions
        function startDrawing(e) {
            isDrawing = true;
            [lastX, lastY] = getMousePos(e);
        }

        function draw(e) {
            if (!isDrawing) return;
            
            const [currentX, currentY] = getMousePos(e);
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(currentX, currentY);
            ctx.stroke();
            
            [lastX, lastY] = [currentX, currentY];
        }

        function getMousePos(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            let clientX, clientY;
            
            if (e.touches) {
                clientX = e.touches[0].clientX;
                clientY = e.touches[0].clientY;
            } else {
                clientX = e.clientX;
                clientY = e.clientY;
            }
            
            return [
                (clientX - rect.left) * scaleX,
                (clientY - rect.top) * scaleY
            ];
        }

        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);

        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startDrawing(e);
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            draw(e);
        });
        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            isDrawing = false;
        });

        // Clear canvas
        clearBtn.addEventListener('click', () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            resultDiv.textContent = '';
            resultDiv.classList.remove('show');
        });

        // Predict function
        predictBtn.addEventListener('click', async () => {
            // Check if canvas is empty
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const isEmpty = !imageData.data.some((channel, i) => i % 4 === 3 && channel !== 0);
            
            if (isEmpty) {
                resultDiv.textContent = '⚠️ Please draw a digit first!';
                resultDiv.classList.add('show');
                return;
            }

            // Show loading
            loadingDiv.style.display = 'block';
            predictBtn.disabled = true;
            resultDiv.classList.remove('show');

            try {
                // Convert canvas to blob
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'digit.png');

                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.success) {
                        resultDiv.innerHTML = `
                            🎯 Predicted Digit: <strong>${data.prediction}</strong><br>
                            📊 Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong>
                        `;
                    } else {
                        resultDiv.textContent = `❌ Error: ${data.error}`;
                    }
                    
                    resultDiv.classList.add('show');
                }, 'image/png');
            } catch (error) {
                resultDiv.textContent = `❌ Network Error: ${error.message}`;
                resultDiv.classList.add('show');
            }

            // Hide loading
            loadingDiv.style.display = 'none';
            predictBtn.disabled = false;
        });
    </script>
</body>
</html>
"""

def create_cnn_model():
    """Create and compile a CNN model for digit recognition"""
    model = keras.Sequential([
        keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        
        # First Convolutional Block
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the model on MNIST dataset"""
    print("Loading and training model...")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create and train the model
    model = create_cnn_model()
    
    # Train the model (reduced epochs for faster deployment)
    model.fit(x_train, y_train, 
              epochs=3,  # Reduced for faster deployment
              batch_size=128,
              validation_data=(x_test, y_test),
              verbose=1)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model

def load_or_train_model():
    """Load existing model or train a new one"""
    model_path = '/app/models/digit_recognition_model.h5'
    
    try:
        # Try to load pre-trained model
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            print("Loaded existing model")
        else:
            raise FileNotFoundError("Model not found")
    except:
        # Train new model if loading fails
        print("Training new model...")
        model = train_model()
        # Save the trained model
        os.makedirs('/app/models', exist_ok=True)
        model.save(model_path)
        print("Model saved")
    
    return model

def preprocess_image(image_data):
    """Preprocess the image for prediction"""
    try:
        # Convert image data to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to 28x28 (MNIST standard)
        img_array = cv2.resize(img_array, (28, 28))
        
        # Invert colors (MNIST has white digits on black background)
        img_array = 255 - img_array
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_digit():
    """Predict the digit from uploaded image"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            })
        
        # Get image file
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            })
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'success': True,
            'prediction': predicted_digit,
            'confidence': confidence,
            'all_probabilities': prediction[0].tolist()
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health_check():
    """Health check endpoint for GCP"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading/training model...")
    
    # Load or train the model
    model = load_or_train_model()
    
    print("Model ready!")
    
    # Get port from environment variable (Cloud Run requirement)
    port = int(os.environ.get('PORT', 8080))
    
    print(f"Starting server on port {port}")
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)