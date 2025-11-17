"""
Real-Time Performance Testing Module
Comprehensive testing of model APIs for latency, throughput, and reliability
"""

import time
import numpy as np
import requests
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceTester:
    """Test and benchmark model API performance."""

    def __init__(self, nn_url="http://localhost:5000", transformer_url="http://localhost:5001"):
        """
        Initialize performance tester.

        Args:
            nn_url (str): Neural Network API base URL
            transformer_url (str): Transformer API base URL
        """
        self.nn_url = nn_url
        self.transformer_url = transformer_url
        self.input_size = 128

    def generate_random_state(self) -> List[float]:
        """Generate a random input state."""
        return np.random.rand(self.input_size).tolist()

    def test_single_request(self, api_url: str, timeout: int = 5) -> Tuple[float, bool, str]:
        """
        Test a single API request.

        Args:
            api_url (str): API endpoint URL
            timeout (int): Request timeout in seconds

        Returns:
            tuple: (latency, success, response_action)
        """
        state = self.generate_random_state()
        start_time = time.time()

        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"state": state},
                timeout=timeout
            )
            latency = time.time() - start_time

            if response.status_code == 200:
                action = response.json().get("action", "unknown")
                return latency, True, action
            else:
                return latency, False, f"Error: {response.status_code}"

        except requests.exceptions.Timeout:
            return timeout, False, "Timeout"
        except requests.exceptions.RequestException as e:
            return time.time() - start_time, False, f"Error: {str(e)}"

    def test_latency(self, api_url: str, num_requests: int = 100) -> Dict:
        """
        Test API latency with multiple requests.

        Args:
            api_url (str): API endpoint URL
            num_requests (int): Number of requests to make

        Returns:
            dict: Latency statistics
        """
        logger.info(f"Testing latency for {api_url} with {num_requests} requests...")

        latencies = []
        successes = 0
        errors = []

        for i in range(num_requests):
            latency, success, response = self.test_single_request(api_url)
            latencies.append(latency)

            if success:
                successes += 1
            else:
                errors.append(response)

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{num_requests} requests completed")

        # Calculate statistics
        latencies_array = np.array(latencies)
        stats = {
            'num_requests': num_requests,
            'successes': successes,
            'failures': num_requests - successes,
            'success_rate': (successes / num_requests) * 100,
            'avg_latency': np.mean(latencies_array),
            'median_latency': np.median(latencies_array),
            'min_latency': np.min(latencies_array),
            'max_latency': np.max(latencies_array),
            'std_latency': np.std(latencies_array),
            'p95_latency': np.percentile(latencies_array, 95),
            'p99_latency': np.percentile(latencies_array, 99),
            'errors': errors[:10]  # Store first 10 errors
        }

        return stats

    def test_concurrent_load(self, api_url: str, num_workers: int = 10, requests_per_worker: int = 10) -> Dict:
        """
        Test API under concurrent load.

        Args:
            api_url (str): API endpoint URL
            num_workers (int): Number of concurrent workers
            requests_per_worker (int): Requests per worker

        Returns:
            dict: Load test results
        """
        logger.info(f"Testing concurrent load: {num_workers} workers, {requests_per_worker} requests each")

        start_time = time.time()
        latencies = []
        successes = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers * requests_per_worker):
                future = executor.submit(self.test_single_request, api_url)
                futures.append(future)

            for future in as_completed(futures):
                latency, success, _ = future.result()
                latencies.append(latency)
                if success:
                    successes += 1

        total_time = time.time() - start_time
        total_requests = num_workers * requests_per_worker

        stats = {
            'num_workers': num_workers,
            'requests_per_worker': requests_per_worker,
            'total_requests': total_requests,
            'total_time': total_time,
            'requests_per_second': total_requests / total_time,
            'successes': successes,
            'failures': total_requests - successes,
            'success_rate': (successes / total_requests) * 100,
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95)
        }

        return stats

    def test_health_endpoint(self, api_url: str) -> Tuple[bool, str]:
        """
        Test health check endpoint.

        Args:
            api_url (str): API base URL

        Returns:
            tuple: (is_healthy, message)
        """
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                return True, "Service is healthy"
            else:
                return False, f"Health check failed: {response.status_code}"
        except Exception as e:
            return False, f"Health check error: {str(e)}"

    def run_comprehensive_test(self, num_latency_tests: int = 100,
                              num_workers: int = 10, requests_per_worker: int = 10) -> Dict:
        """
        Run comprehensive performance tests on both models.

        Args:
            num_latency_tests (int): Number of latency test requests
            num_workers (int): Number of concurrent workers for load test
            requests_per_worker (int): Requests per worker in load test

        Returns:
            dict: Complete test results
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE PERFORMANCE TESTS")
        logger.info("=" * 60)

        results = {
            'timestamp': datetime.now().isoformat(),
            'neural_network': {},
            'transformer': {}
        }

        # Test Neural Network
        logger.info("\n=== Testing Neural Network API ===")

        # Health check
        nn_healthy, nn_health_msg = self.test_health_endpoint(self.nn_url)
        logger.info(f"Health check: {nn_health_msg}")
        results['neural_network']['health'] = {'healthy': nn_healthy, 'message': nn_health_msg}

        if nn_healthy:
            # Latency test
            nn_latency_stats = self.test_latency(self.nn_url, num_latency_tests)
            results['neural_network']['latency'] = nn_latency_stats
            logger.info(f"NN Latency - Avg: {nn_latency_stats['avg_latency']:.4f}s, "
                       f"P95: {nn_latency_stats['p95_latency']:.4f}s, "
                       f"Success Rate: {nn_latency_stats['success_rate']:.1f}%")

            # Load test
            nn_load_stats = self.test_concurrent_load(self.nn_url, num_workers, requests_per_worker)
            results['neural_network']['load'] = nn_load_stats
            logger.info(f"NN Load Test - RPS: {nn_load_stats['requests_per_second']:.2f}, "
                       f"Success Rate: {nn_load_stats['success_rate']:.1f}%")

        # Test Transformer
        logger.info("\n=== Testing Transformer API ===")

        # Health check
        tf_healthy, tf_health_msg = self.test_health_endpoint(self.transformer_url)
        logger.info(f"Health check: {tf_health_msg}")
        results['transformer']['health'] = {'healthy': tf_healthy, 'message': tf_health_msg}

        if tf_healthy:
            # Latency test
            tf_latency_stats = self.test_latency(self.transformer_url, num_latency_tests)
            results['transformer']['latency'] = tf_latency_stats
            logger.info(f"Transformer Latency - Avg: {tf_latency_stats['avg_latency']:.4f}s, "
                       f"P95: {tf_latency_stats['p95_latency']:.4f}s, "
                       f"Success Rate: {tf_latency_stats['success_rate']:.1f}%")

            # Load test
            tf_load_stats = self.test_concurrent_load(self.transformer_url, num_workers, requests_per_worker)
            results['transformer']['load'] = tf_load_stats
            logger.info(f"Transformer Load Test - RPS: {tf_load_stats['requests_per_second']:.2f}, "
                       f"Success Rate: {tf_load_stats['success_rate']:.1f}%")

        return results

    def save_results(self, results: Dict, output_dir: str = "results"):
        """
        Save test results to files.

        Args:
            results (dict): Test results
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        json_path = os.path.join(output_dir, "performance_test_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

        # Save readable text summary
        txt_path = os.path.join(output_dir, "latency_results.txt")
        with open(txt_path, 'w') as f:
            f.write("AI Gameplay Bot - Performance Test Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Test Date: {results['timestamp']}\n\n")

            # Neural Network results
            if 'latency' in results.get('neural_network', {}):
                nn_stats = results['neural_network']['latency']
                f.write("NEURAL NETWORK MODEL\n")
                f.write("-" * 60 + "\n")
                f.write(f"Average Latency:     {nn_stats['avg_latency']:.4f} seconds\n")
                f.write(f"Median Latency:      {nn_stats['median_latency']:.4f} seconds\n")
                f.write(f"Min Latency:         {nn_stats['min_latency']:.4f} seconds\n")
                f.write(f"Max Latency:         {nn_stats['max_latency']:.4f} seconds\n")
                f.write(f"P95 Latency:         {nn_stats['p95_latency']:.4f} seconds\n")
                f.write(f"P99 Latency:         {nn_stats['p99_latency']:.4f} seconds\n")
                f.write(f"Success Rate:        {nn_stats['success_rate']:.2f}%\n\n")

            # Transformer results
            if 'latency' in results.get('transformer', {}):
                tf_stats = results['transformer']['latency']
                f.write("TRANSFORMER MODEL\n")
                f.write("-" * 60 + "\n")
                f.write(f"Average Latency:     {tf_stats['avg_latency']:.4f} seconds\n")
                f.write(f"Median Latency:      {tf_stats['median_latency']:.4f} seconds\n")
                f.write(f"Min Latency:         {tf_stats['min_latency']:.4f} seconds\n")
                f.write(f"Max Latency:         {tf_stats['max_latency']:.4f} seconds\n")
                f.write(f"P95 Latency:         {tf_stats['p95_latency']:.4f} seconds\n")
                f.write(f"P99 Latency:         {tf_stats['p99_latency']:.4f} seconds\n")
                f.write(f"Success Rate:        {tf_stats['success_rate']:.2f}%\n\n")

            # Load test results
            if 'load' in results.get('neural_network', {}):
                nn_load = results['neural_network']['load']
                f.write("LOAD TEST RESULTS\n")
                f.write("-" * 60 + "\n")
                f.write(f"NN Requests/Second:   {nn_load['requests_per_second']:.2f}\n")

                if 'load' in results.get('transformer', {}):
                    tf_load = results['transformer']['load']
                    f.write(f"TF Requests/Second:   {tf_load['requests_per_second']:.2f}\n")

        logger.info(f"Text summary saved to {txt_path}")


def main():
    """Main execution function."""
    tester = PerformanceTester()

    # Run comprehensive tests
    results = tester.run_comprehensive_test(
        num_latency_tests=100,
        num_workers=10,
        requests_per_worker=10
    )

    # Save results
    tester.save_results(results)

    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE TESTING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
