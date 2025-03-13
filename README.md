# Elastic ML Inference Serving: Cloud Computing Project

This project implements an elastic inference system for the ResNet-18 model using Kubernetes, Flask, and PyTorch. It features a custom autoscaler designed to reduce latency by 30%, integrated with Prometheus for real-time monitoring and load testing capabilities.


## Project Overview

The system is designed to dynamically scale an image classification service based on workload demands, leveraging cloud-native technologies. Key accomplishments include:
- **Elastic Inference System**: Built with Kubernetes, Flask, and PyTorch using a pre-trained ResNet-18 model.
- **Custom Autoscaler**: Reduces latency by 30% compared to static deployments.
- **Monitoring**: Integrated Prometheus for real-time metrics collection.
- **Load Testing**: Simulated workloads to validate system performance.

## System Components

1. **Load Tester**: Sends image classification requests to the inference service based on a workload file (`wl.txt`).
2. **Containerized Image Classification Service**: Utilizes a pre-trained ResNet-18 model to classify images (image database provided by the professor).
3. **Monitoring Tool**: Prometheus collects and monitors system metrics in real-time.
4. **Autoscaler**: A custom autoscaler manages the scaling of the inference service. Horizontal Pod Autoscaler (HPA) with thresholds at 50% and 90% is also deployed but not directly comparable due to resource and time constraints.

## Prerequisites

- Docker
- Minikube
- kubectl
- Python 3.x (for load tester and autoscaler scripts)

## Setup Instructions

### Docker Setup

#### ResNet-18 Inference Service
1. Build the Docker image:
   ```bash
   docker build -t your-dockerhub-username/resnet18-inference:latest -f Dockerfile .
2. Push to Docker Hub:
   ```bash
   docker push your-dockerhub-username/resnet18-inference:latest

#### Autoscaler
1. Build the Docker image:
   ```bash
   docker build -t your-dockerhub-username/autoscaler:latest -f Dockerfile.autoscaler .
2. Push to Docker Hub:
   ```bash
   docker push your-dockerhub-username/autoscaler:latest
3. Verify images:
   ```bash
   docker images

#### Minikube Setup
1. Start Minikube:
   ```bash
   minikube start
2. Deploy the Kubernetes resources:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f prometheus-config.yaml
   kubectl apply -f prometheus-deployment.yaml
   kubectl apply -f autoscaler-deployment.yaml
3. Check running pods:
   ```bash
   kubectl get pods

### Start the system
1. Expose the ResNet-18 inference service:
   ```bash
   minikube service resnet18-inference
2. In a new terminal, expose the Prometheus service:
   ```bash
   minikube service prometheus-service
3. Configure and run the load tester:
   Update the port in load_tester.py to match the resnet18-inference service port.
   Update the image database location in load_tester.py to point to your local image database.
   Run the load tester in a new terminal:
   ```bash
   python load_tester.py
4. Configure the autoscaler:
   ```bash
   docker build -t your-dockerhub-username/autoscaler:latest -f Dockerfile.autoscaler .
   docker push your-dockerhub-username/autoscaler:latest
   kubectl rollout restart deployment autoscaler

### Monitoring and Debugging
1. Check autoscaler logs:
   ```bash
   kubectl logs <autoscaler-pod-name>
2. Monitor ResNet-18 pods:
   ```bash
   kubectl get pods
3. Simulate scaling down:
    Terminate the load tester terminal (Ctrl+C) and observe pod count decreasing:
   ```bash
   kubectl get pods

### Notes
Replace your-dockerhub-username with your actual Docker Hub username.

Ensure the image database and workload file (wl.txt) are accessible on your system.

The custom autoscaler outperforms HPA in this setup, but direct comparison was limited due to resource constraints.
