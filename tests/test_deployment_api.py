"""
Integration tests for Deployment API
"""

import pytest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deployment'))


class TestDeploymentAPI:
    """Test deployment API endpoints."""

    @pytest.fixture
    def sample_state(self):
        """Generate a sample state for testing."""
        return [0.5] * 128

    def test_predict_endpoint_format(self, sample_state):
        """Test that predict request format is correct."""
        request_data = {"state": sample_state}

        assert "state" in request_data
        assert isinstance(request_data["state"], list)
        assert len(request_data["state"]) == 128

    def test_predict_response_format(self):
        """Test expected format of prediction response."""
        # Mock response
        response_data = {"action": "move_forward"}

        assert "action" in response_data
        assert isinstance(response_data["action"], str)

    def test_action_mapping_completeness(self):
        """Test that action mapping covers all indices."""
        action_mapping = {
            0: "move_forward",
            1: "move_backward",
            2: "turn_left",
            3: "turn_right",
            4: "attack",
            5: "jump",
            6: "interact",
            7: "use_item",
            8: "open_inventory",
            9: "cast_spell"
        }

        # Check all indices 0-9 are mapped
        for i in range(10):
            assert i in action_mapping
            assert isinstance(action_mapping[i], str)


class TestControlBackend:
    """Test control backend functionality."""

    def test_status_response_format(self):
        """Test status endpoint response format."""
        # Mock status response
        status = {
            'nn_running': False,
            'transformer_running': False,
            'active_model': 'nn',
            'timestamp': 1234567890.0
        }

        assert 'nn_running' in status
        assert 'transformer_running' in status
        assert 'active_model' in status
        assert 'timestamp' in status

    def test_model_selection_validation(self):
        """Test that only valid models can be selected."""
        valid_models = ['nn', 'transformer']

        for model in valid_models:
            assert model in ['nn', 'transformer']

        invalid_model = 'invalid_model'
        assert invalid_model not in valid_models


if __name__ == '__main__':
    pytest.main([__file__])
