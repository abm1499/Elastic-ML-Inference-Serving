import os
import random
from PIL import Image
import torch
from torchvision import models, transforms
from flask import Flask, render_template, request, send_from_directory
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Randomly select a file from the folder
            random_file = random.choice(file_list)
            random_file_path = os.path.join(folder_path, random_file)
            app.logger.info(f"Selected file: {random_file_path}")

            # Load the randomly selected image
            img = Image.open(random_file_path)

            # Preprocess the image
            inp = preprocess(img).unsqueeze(0)

            # Pass the preprocessed image to the model and get predictions
            with torch.no_grad():
                preds = model(inp).squeeze(0)

            # Sort the predictions in descending order
            sorted_preds, indices = preds.sort(descending=True)

            # Get top 3 predictions
            top_predictions = [weights.meta["categories"][idx.item()] for idx in indices[:3]]

            return render_template('index.html', image_filename=random_file, predictions=top_predictions)
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)