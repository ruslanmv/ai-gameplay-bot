"""
Logging Configuration Module
Provides centralized logging setup for the entire application
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

# Import config
try:
    from config import LOG_LEVEL, LOG_DIR, ENVIRONMENT
except ImportError:
    LOG_LEVEL = 'INFO'
    LOG_DIR = Path('logs')
    ENVIRONMENT = 'development'

# Create logs directory if it doesn't exist
LOG_DIR = Path(LOG_DIR)
LOG_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record):
        """Format the log record with colors."""
        if sys.stdout.isatty():  # Only use colors for terminal output
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    log_file: str = None,
    level: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name (str): Logger name
        log_file (str, optional): Log file name (saved in LOG_DIR)
        level (str, optional): Logging level (defaults to LOG_LEVEL from config)
        console (bool): Whether to add console handler

    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Set level
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_path = LOG_DIR / log_file

        # Rotating file handler (max 10MB, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_app_logging():
    """
    Set up logging for the entire application.
    Creates loggers for different components.
    """
    # Main application log
    app_logger = setup_logger(
        'ai_gameplay_bot',
        log_file='app.log',
        console=True
    )

    # Neural network service log
    nn_logger = setup_logger(
        'nn_service',
        log_file='nn_service.log',
        console=ENVIRONMENT == 'development'
    )

    # Transformer service log
    transformer_logger = setup_logger(
        'transformer_service',
        log_file='transformer_service.log',
        console=ENVIRONMENT == 'development'
    )

    # Control backend log
    control_logger = setup_logger(
        'control_backend',
        log_file='control_backend.log',
        console=True
    )

    # Training log
    training_logger = setup_logger(
        'training',
        log_file='training.log',
        console=True
    )

    # Error log (only errors and critical)
    error_logger = setup_logger(
        'errors',
        log_file='errors.log',
        level='ERROR',
        console=False
    )

    app_logger.info(f"Logging initialized for environment: {ENVIRONMENT}")
    app_logger.info(f"Log files stored in: {LOG_DIR}")

    return {
        'app': app_logger,
        'nn': nn_logger,
        'transformer': transformer_logger,
        'control': control_logger,
        'training': training_logger,
        'errors': error_logger
    }


# Create default loggers
loggers = setup_app_logging()


# Convenience function
def get_logger(name: str = 'ai_gameplay_bot') -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


# Exception logging decorator
def log_exceptions(logger_name='ai_gameplay_bot'):
    """
    Decorator to log exceptions in functions.

    Usage:
        @log_exceptions('my_logger')
        def my_function():
            # ... code that might raise exceptions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator


if __name__ == '__main__':
    # Test logging setup
    logger = get_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    print(f"\nLogs are saved to: {LOG_DIR}")
