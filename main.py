import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


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
        
        # Reshape for model input (adjust based on your model's expected input shape)
        img_array = img_array.reshape(1, 28, 28, 1)  # For CNN models
        # If your model expects flattened input, use: img_array.reshape(1, 784)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def segment_digits(thresh_image):
    """Segment individual digits from the image"""
    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours first, then sort by x-coordinate (left to right)
    valid_contours = []

    for contour in contours:
        # Filter out very small contours (noise)
        if cv2.contourArea(contour) < 100:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size and aspect ratio
        if w > 15 and h > 15 and w < 200 and h < 200:
            aspect_ratio = w / h
            if 0.2 <= aspect_ratio <= 3.0:
                valid_contours.append((contour, x, y, w, h))
    
    # FIXED: Sort by x-coordinate (left to right) instead of area
    valid_contours.sort(key=lambda item: item[1])  # Sort by x-coordinate

    digit_images = []
    bounding_boxes = []

    for contour, x, y, w, h in valid_contours:
        # Check for overlapping bounding boxes (remove duplicates)
        overlap = False
        for existing_x, existing_y, existing_w, existing_h in bounding_boxes:
            # Check if current box significantly overlaps with existing ones
            if (abs(x - existing_x) < 20 and abs(y - existing_y) < 20):
                overlap = True
                break

        if not overlap:
            digit_roi = thresh_image[y:y+h, x:x+w]
            pad = 10
            padded = cv2.copyMakeBorder(digit_roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

            digit_images.append(resized)
            bounding_boxes.append((x, y, w, h))

    return digit_images, bounding_boxes

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_digit():
    """Predict the digit from uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server logs.'
            })
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            })
        
        # Get image file
        image_file = request.files['image']
        img= image_file.read()
        
        #Preporcessing
        threshold_image, original_gray = preprocess_image(img)

        #Segmenting digits
        digits, bounding_boxes = segment_digits(threshold_image)
        
        if digits is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            })
        
        # Preprocess for model
        processed_digits = []
        for digit in digits:
            # Try inverting colors - MNIST has white digits on black background
            inverted = 255 - digit

            # Normalize
            normalized = inverted.astype('float32') / 255.0
            processed_digits.append(normalized)

        processed_digits = np.array(processed_digits)

        #adds channel dimension
        processed_digits = processed_digits.reshape(-1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(processed_digits)
        predicted_classes = np.argmax(prediction, axis=1)
        confidence_scores = np.max(prediction, axis=1)
        #predicted_digit = int(np.argmax(prediction[0]))
        #confidence = float(np.max(prediction[0]))
        
        #Threshold Confidence
        confidence_threshold = 0.5

        # Filter by confidence
        results = []
        for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidence_scores)):
            if confidence >= confidence_threshold:
                results.append({
                    'digit': int(pred_class),
                    'confidence': float(confidence),
                    'index': i  # Since we don't have bounding boxes in canvas case
                })
        
        # Prepare response based on results
        if results:
            # Sort by confidence (highest first) since we don't have x-coordinates
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return jsonify({
                'success': True,
                'high_confidence_results': results,
                'best_prediction': results[0],  # Highest confidence
                'total_confident_predictions': len(results),
                'confidence_threshold': confidence_threshold,
                'all_probabilities': prediction[0].tolist()
            })
        else:
            # No high confidence predictions
            return jsonify({
                'success': True,
                'high_confidence_results': [],
                'message': f'No predictions above {confidence_threshold} confidence threshold',
                'best_available': {
                    'digit': int(predicted_classes[0]),
                    'confidence': float(confidence_scores[0])
                },
                'confidence_threshold': confidence_threshold
            })
 
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    
    # Model Path
    model = load_model('./model.h5')
    
    if model is None:
        print("❌ Failed to load model. Server will start but predictions won't work.")
        print("Please check the model path and try again.")
    else:
        print("✅ Model loaded successfully! Ready to make predictions.")
    
    # Get port from environment variable (Cloud Run requirement)
    port = int(os.environ.get('PORT', 8080))
    
    print(f"🌐 Starting server on port {port}")
    print(f"📱 Access the app at: http://localhost:{port}")
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)