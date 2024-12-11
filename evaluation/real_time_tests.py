import time
import numpy as np
import requests

# API endpoints
NN_API_URL = "http://localhost:5000/predict"
TRANSFORMER_API_URL = "http://localhost:5001/predict"

# Generate a random input state
def generate_random_state(input_size=128):
    return list(np.random.rand(input_size))

# Test API response time
def test_model_latency(api_url, num_requests=100, input_size=128):
    total_time = 0
    for _ in range(num_requests):
        state = generate_random_state(input_size)
        start_time = time.time()
        response = requests.post(api_url, json={"state": state})
        total_time += time.time() - start_time
        if response.status_code != 200:
            print(f"Error: {response.json()}")
    average_latency = total_time / num_requests
    return average_latency

# Test both models
nn_latency = test_model_latency(NN_API_URL)
transformer_latency = test_model_latency(TRANSFORMER_API_URL)

print(f"Neural Network Average Latency: {nn_latency:.4f} seconds")
print(f"Transformer Average Latency: {transformer_latency:.4f} seconds")

# Save results
with open("results/latency_results.txt", "w") as f:
    f.write(f"Neural Network Average Latency: {nn_latency:.4f} seconds\n")
    f.write(f"Transformer Average Latency: {transformer_latency:.4f} seconds\n")
print("Latency results saved to results/latency_results.txt")
