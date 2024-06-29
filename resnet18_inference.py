import os
import random
from PIL import Image
import torch
from torchvision import models, transforms
from flask import Flask, render_template, request, send_from_directory, jsonify
import logging
import io
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import Counter, Summary, Gauge, Histogram

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUESTS = Counter('http_requests_total', 'Total HTTP Requests (count)', ['method', 'endpoint', 'http_status'])
CURRENT_REQUESTS = Gauge('current_requests', 'Number of requests currently being processed')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')

# Load pre-trained ResNet18 model and weights
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.eval()

# Define transformations
preprocess = weights.transforms()

# Path to the folder containing the images
folder_path = '/app/imagenet-sample-images-master'

# Get a list of all files in the folder
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

@app.route('/images/<path:filename>')
def serve_image(filename):
    app.logger.info(f"Serving image: {filename}")
    return send_from_directory(folder_path, filename)

@app.route('/')
def home():
    return "ResNet18 Inference Service is running!"

@app.route('/metrics')
def metrics():
    return make_wsgi_app()
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

@REQUEST_TIME.time()
@REQUEST_LATENCY.time()
@app.route('/predict', methods=['POST'])
def predict():
    CURRENT_REQUESTS.inc()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # Read the image file
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Preprocess the image
            inp = preprocess(img).unsqueeze(0)
            
            # Pass the preprocessed image to the model and get predictions
            with torch.no_grad():
                preds = model(inp).squeeze(0)
            
            # Sort the predictions in descending order
            sorted_preds, indices = preds.sort(descending=True)
            
            # Get top 3 predictions
            top_predictions = [weights.meta["categories"][idx.item()] for idx in indices[:3]]
            
            return jsonify({"predictions": top_predictions})
        
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)