"""
Unit tests for Input Mapping Module
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from input_mapping import ActionType, ActionMapper, get_action_mapper


class TestActionType:
    """Test ActionType enum."""

    def test_action_types_exist(self):
        """Test that all expected action types exist."""
        expected_actions = [
            'MOVE_FORWARD', 'MOVE_BACKWARD', 'TURN_LEFT', 'TURN_RIGHT',
            'ATTACK', 'JUMP', 'INTERACT', 'USE_ITEM', 'OPEN_INVENTORY', 'CAST_SPELL'
        ]

        for action in expected_actions:
            assert hasattr(ActionType, action)

    def test_action_values(self):
        """Test that action values are sequential integers."""
        assert ActionType.MOVE_FORWARD.value == 0
        assert ActionType.CAST_SPELL.value == 9


class TestActionMapper:
    """Test ActionMapper class."""

    def test_mapper_initialization(self):
        """Test that ActionMapper initializes correctly."""
        mapper = ActionMapper()
        assert mapper is not None
        assert mapper.action_mapping is not None
        assert len(mapper.action_mapping) > 0

    def test_default_mapping(self):
        """Test that default mapping contains all action types."""
        mapper = ActionMapper()
        for action_type in ActionType:
            assert action_type.value in mapper.action_mapping

    def test_custom_mapping(self):
        """Test initialization with custom mapping."""
        custom_map = {0: ['custom_key']}
        mapper = ActionMapper(config=custom_map)
        assert mapper.action_mapping[0] == ['custom_key']

    def test_get_action_name(self):
        """Test getting action name by index."""
        mapper = ActionMapper()

        name = mapper.get_action_name(ActionType.ATTACK.value)
        assert name == 'ATTACK'

        name = mapper.get_action_name(999)
        assert 'CUSTOM_ACTION' in name

    def test_update_mapping(self):
        """Test updating action mapping."""
        mapper = ActionMapper()
        new_mapping = {ActionType.JUMP.value: ['space', 'shift']}
        mapper.update_mapping(new_mapping)

        assert mapper.action_mapping[ActionType.JUMP.value] == ['space', 'shift']


class TestGlobalActionMapper:
    """Test global action mapper instance."""

    def test_get_action_mapper_singleton(self):
        """Test that get_action_mapper returns same instance."""
        mapper1 = get_action_mapper()
        mapper2 = get_action_mapper()

        assert mapper1 is mapper2


if __name__ == '__main__':
    pytest.main([__file__])
