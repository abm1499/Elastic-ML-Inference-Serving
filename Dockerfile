# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the Python requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY resnet18_inference.py .
COPY templates/ ./templates/

# Copy the imagenet-sample-images-master directory
COPY imagenet-sample-images-master/ /app/imagenet-sample-images-master

# Set the entry point
CMD ["python", "resnet18_inference.py"]
