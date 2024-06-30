FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY resnet18_inference.py .
COPY templates/ ./templates/
COPY imagenet-sample-images-master/ /app/imagenet-sample-images-master
CMD ["python", "resnet18_inference.py"]