"""
Control Backend API
Flask server that manages NN and Transformer services
Provides API endpoints for the frontend control panel
"""

import os
import sys
import subprocess
import signal
import time
import logging
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global state
service_processes = {
    'nn': None,
    'transformer': None
}
active_model = 'nn'

# Service configurations
NN_PORT = 5000
TRANSFORMER_PORT = 5001
NN_SCRIPT = "deployment/deploy_nn.py"
TRANSFORMER_SCRIPT = "deployment/deploy_transformer.py"


def is_service_running(port):
    """
    Check if a service is running on the specified port.

    Args:
        port (int): Port number to check

    Returns:
        bool: True if service is responding, False otherwise
    """
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_service(service_name, script_path, port):
    """
    Start a service by running its deployment script.

    Args:
        service_name (str): Name of the service ('nn' or 'transformer')
        script_path (str): Path to the deployment script
        port (int): Port number for the service

    Returns:
        bool: True if service started successfully, False otherwise
    """
    global service_processes

    try:
        # Check if already running
        if service_processes[service_name] is not None:
            if service_processes[service_name].poll() is None:
                logger.info(f"{service_name} service is already running")
                return True

        logger.info(f"Starting {service_name} service on port {port}...")

        # Start the service as a subprocess
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )

        service_processes[service_name] = process

        # Wait a moment and check if it's running
        time.sleep(2)

        if process.poll() is None:
            logger.info(f"{service_name} service started successfully")
            return True
        else:
            logger.error(f"{service_name} service failed to start")
            return False

    except Exception as e:
        logger.error(f"Error starting {service_name} service: {e}")
        return False


def stop_service(service_name):
    """
    Stop a running service.

    Args:
        service_name (str): Name of the service to stop

    Returns:
        bool: True if service stopped successfully, False otherwise
    """
    global service_processes

    try:
        process = service_processes.get(service_name)

        if process is None or process.poll() is not None:
            logger.info(f"{service_name} service is not running")
            service_processes[service_name] = None
            return True

        logger.info(f"Stopping {service_name} service...")

        # Try graceful shutdown first
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()

        # Wait for process to terminate
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()

        service_processes[service_name] = None
        logger.info(f"{service_name} service stopped")
        return True

    except Exception as e:
        logger.error(f"Error stopping {service_name} service: {e}")
        return False


# API Endpoints

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the status of all services and active model."""
    nn_running = is_service_running(NN_PORT)
    transformer_running = is_service_running(TRANSFORMER_PORT)

    return jsonify({
        'nn_running': nn_running,
        'transformer_running': transformer_running,
        'active_model': active_model,
        'timestamp': time.time()
    })


@app.route('/api/start_nn', methods=['POST'])
def start_nn():
    """Start the Neural Network service."""
    success = start_service('nn', NN_SCRIPT, NN_PORT)
    return jsonify({
        'success': success,
        'message': 'NN service started' if success else 'Failed to start NN service'
    }), 200 if success else 500


@app.route('/api/stop_nn', methods=['POST'])
def stop_nn():
    """Stop the Neural Network service."""
    success = stop_service('nn')
    return jsonify({
        'success': success,
        'message': 'NN service stopped' if success else 'Failed to stop NN service'
    }), 200 if success else 500


@app.route('/api/start_transformer', methods=['POST'])
def start_transformer():
    """Start the Transformer service."""
    success = start_service('transformer', TRANSFORMER_SCRIPT, TRANSFORMER_PORT)
    return jsonify({
        'success': success,
        'message': 'Transformer service started' if success else 'Failed to start Transformer service'
    }), 200 if success else 500


@app.route('/api/stop_transformer', methods=['POST'])
def stop_transformer():
    """Stop the Transformer service."""
    success = stop_service('transformer')
    return jsonify({
        'success': success,
        'message': 'Transformer service stopped' if success else 'Failed to stop Transformer service'
    }), 200 if success else 500


@app.route('/api/set_active_model', methods=['POST'])
def set_active_model():
    """Set the active model (nn or transformer)."""
    global active_model

    data = request.get_json()
    model = data.get('model', 'nn')

    if model not in ['nn', 'transformer']:
        return jsonify({
            'success': False,
            'message': 'Invalid model. Must be "nn" or "transformer"'
        }), 400

    active_model = model
    logger.info(f"Active model set to: {active_model}")

    return jsonify({
        'success': True,
        'active_model': active_model,
        'message': f'Active model set to {active_model}'
    })


@app.route('/api/test_predict', methods=['POST'])
def test_predict():
    """Test prediction using the specified model."""
    data = request.get_json()
    model = data.get('model', 'nn')

    # Generate random test state
    test_state = np.random.rand(128).tolist()

    # Determine which port to use
    port = NN_PORT if model == 'nn' else TRANSFORMER_PORT

    try:
        # Call the prediction service
        response = requests.post(
            f"http://localhost:{port}/predict",
            json={'state': test_state},
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'action': result.get('action', 'unknown'),
                'model': model
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Prediction service returned error: {response.status_code}'
            }), 500

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling prediction service: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to connect to {model} service. Is it running?'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})


def cleanup():
    """Cleanup function to stop all services on shutdown."""
    logger.info("Shutting down control backend...")
    stop_service('nn')
    stop_service('transformer')


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup)

    logger.info("Starting Control Backend API on port 8000...")
    logger.info("Frontend should connect to http://localhost:8000")

    try:
        app.run(host='0.0.0.0', port=8000, debug=False)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup()
