import time
import requests
from kubernetes import client, config
import logging
import sys
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

PROMETHEUS_URL = "http://10.102.122.62:9090"
QUERY = 'rate(request_latency_seconds_sum[1m]) / rate(request_latency_seconds_count[1m])'
DEPLOYMENT_NAME = 'resnet18-inference'
NAMESPACE = 'default'
SCALE_UP_THRESHOLD = 0.2
SCALE_DOWN_THRESHOLD = 0.1
MIN_REPLICAS = 1
MAX_REPLICAS = 10
SCALE_INTERVAL = 15

def get_metrics():
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': QUERY})
        logging.info(f"Prometheus response: {response.text}")
        result = response.json()['data']['result']
        if result:
            return float(result[0]['value'][1])
        else:
            logging.warning("No results from Prometheus query")
    except Exception as e:
        logging.exception(f"Error fetching metrics: {e}")
    return 0

def scale_deployment(api, deployment_name, namespace, replicas):
    try:
        api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body={'spec': {'replicas': replicas}}
        )
        logging.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
    except Exception as e:
        logging.exception(f"Error scaling deployment: {e}")

def main():
    logging.info("Autoscaler starting up...")
    
    try:
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            config.load_incluster_config()
            logging.info("Running inside Kubernetes cluster")
        else:
            config.load_kube_config()
            logging.info("Running outside Kubernetes cluster")
        
        api = client.AppsV1Api()
        logging.info("Kubernetes API client initialized")
    except Exception as e:
        logging.exception(f"Error initializing Kubernetes client: {e}")
        return

    while True:
        try:
            avg_latency = get_metrics()
            logging.info(f"Current average latency: {avg_latency}")
            
            current_replicas = api.read_namespaced_deployment_scale(
                name=DEPLOYMENT_NAME, namespace=NAMESPACE
            ).spec.replicas
            logging.info(f"Current replicas: {current_replicas}")
            
            if avg_latency > SCALE_UP_THRESHOLD and current_replicas < MAX_REPLICAS:
                logging.info(f"Attempting to scale up from {current_replicas} to {current_replicas + 1}")
                scale_deployment(api, DEPLOYMENT_NAME, NAMESPACE, current_replicas + 1)
            elif avg_latency < SCALE_DOWN_THRESHOLD and current_replicas > MIN_REPLICAS:
                logging.info(f"Attempting to scale down from {current_replicas} to {current_replicas - 1}")
                scale_deployment(api, DEPLOYMENT_NAME, NAMESPACE, current_replicas - 1)
            else:
                logging.info("No scaling action needed")
            
        except Exception as e:
            logging.exception(f"Error in main loop: {e}")
        
        time.sleep(SCALE_INTERVAL)

if __name__ == "__main__":
    main()
