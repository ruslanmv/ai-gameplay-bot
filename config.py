"""
Configuration Module for AI Gameplay Bot
Centralized configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = ENVIRONMENT == 'development'

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Service Ports
NN_PORT = int(os.getenv('NN_PORT', 5000))
TRANSFORMER_PORT = int(os.getenv('TRANSFORMER_PORT', 5001))
CONTROL_PORT = int(os.getenv('CONTROL_PORT', 8000))

# Model Paths
MODELS_DIR = BASE_DIR / 'models'
NN_MODEL_PATH = MODELS_DIR / 'neural_network' / 'nn_model.pth'
NN_FINETUNED_PATH = MODELS_DIR / 'neural_network' / 'nn_model_finetuned.pth'
TRANSFORMER_MODEL_PATH = MODELS_DIR / 'transformer' / 'transformer_model.pth'
TRANSFORMER_FINETUNED_PATH = MODELS_DIR / 'transformer' / 'transformer_model_finetuned.pth'

# Data Paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEEDBACK_DIR = BASE_DIR / 'feedback'
RESULTS_DIR = BASE_DIR / 'results'

# Model Configuration
NN_CONFIG = {
    'input_size': 128,
    'hidden_size': 64,
    'output_size': 10,
    'dropout_rate': 0.3
}

TRANSFORMER_CONFIG = {
    'input_dim': 128,
    'num_classes': 10,
    'num_heads': 4,
    'num_layers': 3,
    'hidden_dim': 256,
    'sequence_length': 10
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'val_split': 0.2,
    'early_stopping_patience': 10
}

# Action Mapping
ACTION_MAPPING = {
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

# Reverse mapping (action name to index)
ACTION_NAME_TO_INDEX = {v: k for k, v in ACTION_MAPPING.items()}

# API Configuration
API_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'backoff_factor': 0.3
}

# Monitoring
ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'
METRICS_PORT = int(os.getenv('METRICS_PORT', 9090))

# Feature flags
FEATURE_FLAGS = {
    'enable_caching': True,
    'enable_metrics': ENABLE_MONITORING,
    'enable_debug_mode': DEBUG,
    'enable_profiling': False
}


def get_model_path(model_type='nn', finetuned=True):
    """
    Get the path to a model file.

    Args:
        model_type (str): 'nn' or 'transformer'
        finetuned (bool): Whether to get finetuned or base model

    Returns:
        Path: Path to the model file
    """
    if model_type == 'nn':
        return NN_FINETUNED_PATH if finetuned else NN_MODEL_PATH
    elif model_type == 'transformer':
        return TRANSFORMER_FINETUNED_PATH if finetuned else TRANSFORMER_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_action_name(action_index):
    """
    Get action name from index.

    Args:
        action_index (int): Action index

    Returns:
        str: Action name
    """
    return ACTION_MAPPING.get(action_index, f"unknown_action_{action_index}")


def get_action_index(action_name):
    """
    Get action index from name.

    Args:
        action_name (str): Action name

    Returns:
        int: Action index or None if not found
    """
    return ACTION_NAME_TO_INDEX.get(action_name.lower())


class Config:
    """Configuration class for easy access to all settings."""

    def __init__(self):
        self.environment = ENVIRONMENT
        self.debug = DEBUG
        self.log_level = LOG_LEVEL
        self.nn_port = NN_PORT
        self.transformer_port = TRANSFORMER_PORT
        self.control_port = CONTROL_PORT

    def __repr__(self):
        return f"Config(environment='{self.environment}', debug={self.debug})"


# Global config instance
config = Config()
