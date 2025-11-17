"""
Input Mapping Module
Maps predicted actions to keyboard/mouse inputs for game control
"""

import time
import platform
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Enumeration of supported action types."""
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ATTACK = 4
    JUMP = 5
    INTERACT = 6
    USE_ITEM = 7
    OPEN_INVENTORY = 8
    CAST_SPELL = 9


class KeyboardController:
    """
    Abstract keyboard controller that handles keyboard input simulation.
    Platform-specific implementations should be created based on the OS.
    """

    def __init__(self):
        self.os_type = platform.system()
        self.active_keys = set()
        logger.info(f"Initialized KeyboardController for {self.os_type}")

    def press_key(self, key: str):
        """
        Press a key.

        Args:
            key (str): Key to press
        """
        try:
            if self.os_type == "Windows":
                self._press_key_windows(key)
            elif self.os_type == "Linux":
                self._press_key_linux(key)
            elif self.os_type == "Darwin":  # macOS
                self._press_key_mac(key)
            self.active_keys.add(key)
            logger.debug(f"Pressed key: {key}")
        except Exception as e:
            logger.error(f"Error pressing key {key}: {e}")

    def release_key(self, key: str):
        """
        Release a key.

        Args:
            key (str): Key to release
        """
        try:
            if self.os_type == "Windows":
                self._release_key_windows(key)
            elif self.os_type == "Linux":
                self._release_key_linux(key)
            elif self.os_type == "Darwin":
                self._release_key_mac(key)
            self.active_keys.discard(key)
            logger.debug(f"Released key: {key}")
        except Exception as e:
            logger.error(f"Error releasing key {key}: {e}")

    def release_all_keys(self):
        """Release all currently pressed keys."""
        for key in list(self.active_keys):
            self.release_key(key)

    def _press_key_windows(self, key: str):
        """Windows-specific key press implementation."""
        try:
            import win32api
            import win32con
            key_code = self._get_windows_keycode(key)
            win32api.keybd_event(key_code, 0, 0, 0)
        except ImportError:
            logger.warning("win32api not available. Install pywin32 for Windows support.")

    def _release_key_windows(self, key: str):
        """Windows-specific key release implementation."""
        try:
            import win32api
            import win32con
            key_code = self._get_windows_keycode(key)
            win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        except ImportError:
            logger.warning("win32api not available. Install pywin32 for Windows support.")

    def _press_key_linux(self, key: str):
        """Linux-specific key press implementation."""
        try:
            from pynput.keyboard import Controller, Key
            controller = Controller()
            controller.press(key)
        except ImportError:
            logger.warning("pynput not available. Install pynput for Linux support.")

    def _release_key_linux(self, key: str):
        """Linux-specific key release implementation."""
        try:
            from pynput.keyboard import Controller, Key
            controller = Controller()
            controller.release(key)
        except ImportError:
            logger.warning("pynput not available. Install pynput for Linux support.")

    def _press_key_mac(self, key: str):
        """macOS-specific key press implementation."""
        try:
            from pynput.keyboard import Controller, Key
            controller = Controller()
            controller.press(key)
        except ImportError:
            logger.warning("pynput not available. Install pynput for macOS support.")

    def _release_key_mac(self, key: str):
        """macOS-specific key release implementation."""
        try:
            from pynput.keyboard import Controller, Key
            controller = Controller()
            controller.release(key)
        except ImportError:
            logger.warning("pynput not available. Install pynput for macOS support.")

    def _get_windows_keycode(self, key: str) -> int:
        """
        Get Windows virtual key code for a given key.

        Args:
            key (str): Key name

        Returns:
            int: Virtual key code
        """
        key_map = {
            'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44,
            'space': 0x20, 'shift': 0x10, 'ctrl': 0x11,
            'e': 0x45, 'f': 0x46, 'q': 0x51, 'r': 0x52,
            'tab': 0x09, 'esc': 0x1B, '1': 0x31, '2': 0x32,
            'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28
        }
        return key_map.get(key.lower(), 0x00)


class ActionMapper:
    """
    Maps predicted action indices to keyboard/mouse inputs.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ActionMapper with custom or default configuration.

        Args:
            config (dict, optional): Custom action-to-key mapping
        """
        self.controller = KeyboardController()
        self.action_mapping = config or self._get_default_mapping()
        self.last_action = None
        self.action_duration = 0.1  # Default action duration in seconds
        logger.info("ActionMapper initialized with default configuration")

    def _get_default_mapping(self) -> Dict[int, List[str]]:
        """
        Get default action-to-key mapping.

        Returns:
            dict: Mapping of action indices to key combinations
        """
        return {
            ActionType.MOVE_FORWARD.value: ['w'],
            ActionType.MOVE_BACKWARD.value: ['s'],
            ActionType.TURN_LEFT.value: ['a'],
            ActionType.TURN_RIGHT.value: ['d'],
            ActionType.ATTACK.value: ['space'],
            ActionType.JUMP.value: ['space'],
            ActionType.INTERACT.value: ['e'],
            ActionType.USE_ITEM.value: ['f'],
            ActionType.OPEN_INVENTORY.value: ['tab'],
            ActionType.CAST_SPELL.value: ['q']
        }

    def execute_action(self, action_index: int, duration: Optional[float] = None):
        """
        Execute an action by pressing the corresponding keys.

        Args:
            action_index (int): Index of the action to execute
            duration (float, optional): Duration to hold the keys (seconds)
        """
        if action_index not in self.action_mapping:
            logger.warning(f"Unknown action index: {action_index}")
            return

        # Release previous action keys
        if self.last_action is not None and self.last_action in self.action_mapping:
            for key in self.action_mapping[self.last_action]:
                self.controller.release_key(key)

        # Execute new action
        keys = self.action_mapping[action_index]
        action_name = ActionType(action_index).name if action_index < len(ActionType) else "UNKNOWN"

        logger.info(f"Executing action: {action_name} (index: {action_index})")

        for key in keys:
            self.controller.press_key(key)

        # Hold keys for specified duration
        if duration is not None:
            time.sleep(duration)
            for key in keys:
                self.controller.release_key(key)
        else:
            # Store current action for release in next iteration
            self.last_action = action_index

    def execute_action_sequence(self, action_sequence: List[int], duration_per_action: float = 0.1):
        """
        Execute a sequence of actions.

        Args:
            action_sequence (list): List of action indices to execute
            duration_per_action (float): Duration for each action in seconds
        """
        logger.info(f"Executing action sequence of length {len(action_sequence)}")

        for action in action_sequence:
            self.execute_action(action, duration=duration_per_action)
            time.sleep(0.05)  # Small delay between actions

        self.controller.release_all_keys()

    def stop_all_actions(self):
        """Stop all current actions and release all keys."""
        logger.info("Stopping all actions")
        self.controller.release_all_keys()
        self.last_action = None

    def update_mapping(self, new_mapping: Dict[int, List[str]]):
        """
        Update the action-to-key mapping.

        Args:
            new_mapping (dict): New mapping configuration
        """
        self.action_mapping.update(new_mapping)
        logger.info("Action mapping updated")

    def get_action_name(self, action_index: int) -> str:
        """
        Get the name of an action by its index.

        Args:
            action_index (int): Action index

        Returns:
            str: Action name
        """
        try:
            return ActionType(action_index).name
        except ValueError:
            return f"CUSTOM_ACTION_{action_index}"


# Global action mapper instance
_action_mapper = None


def get_action_mapper(config: Optional[Dict] = None) -> ActionMapper:
    """
    Get or create the global ActionMapper instance.

    Args:
        config (dict, optional): Custom action-to-key mapping

    Returns:
        ActionMapper: The global action mapper instance
    """
    global _action_mapper
    if _action_mapper is None:
        _action_mapper = ActionMapper(config)
    return _action_mapper


def main():
    """
    Test the input mapping functionality.
    """
    print("Testing Input Mapping...")

    mapper = get_action_mapper()

    # Test individual actions
    print("\nTesting individual actions:")
    for action in ActionType:
        print(f"  - {action.name}")
        mapper.execute_action(action.value, duration=0.2)
        time.sleep(0.3)

    # Test action sequence
    print("\nTesting action sequence:")
    sequence = [
        ActionType.MOVE_FORWARD.value,
        ActionType.TURN_RIGHT.value,
        ActionType.JUMP.value,
        ActionType.ATTACK.value
    ]
    mapper.execute_action_sequence(sequence, duration_per_action=0.2)

    # Stop all actions
    mapper.stop_all_actions()
    print("\nTesting completed!")


if __name__ == "__main__":
    main()
