"""
Unit tests for Video Processing Module
"""

import pytest
import os
import sys
import shutil
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from video_processing import extract_frames


class TestVideoProcessing:
    """Test video processing functions."""

    @pytest.fixture
    def temp_video_path(self, tmp_path):
        """Create a temporary test video."""
        video_path = tmp_path / "test_video.mp4"

        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        # Write 90 frames (3 seconds at 30fps)
        for i in range(90):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            out.write(frame)

        out.release()
        return str(video_path)

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        return str(output_dir)

    def test_extract_frames_basic(self, temp_video_path, temp_output_dir):
        """Test basic frame extraction."""
        extract_frames(temp_video_path, temp_output_dir, frame_rate=1)

        # Check that frames were extracted
        frames = os.listdir(temp_output_dir)
        assert len(frames) > 0
        assert all(f.endswith('.jpg') for f in frames)

    def test_extract_frames_count(self, temp_video_path, temp_output_dir):
        """Test that correct number of frames are extracted."""
        extract_frames(temp_video_path, temp_output_dir, frame_rate=1)

        frames = os.listdir(temp_output_dir)
        # Should extract approximately 3 frames (3 seconds at 1fps)
        assert 2 <= len(frames) <= 4

    def test_extract_frames_creates_directory(self, temp_video_path, tmp_path):
        """Test that extraction creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new_frames"
        extract_frames(temp_video_path, str(output_dir), frame_rate=1)

        assert output_dir.exists()

    def test_extract_frames_naming(self, temp_video_path, temp_output_dir):
        """Test that frames are named correctly."""
        extract_frames(temp_video_path, temp_output_dir, frame_rate=1)

        frames = sorted(os.listdir(temp_output_dir))
        assert frames[0].startswith('frame_')
        assert frames[0].endswith('.jpg')


if __name__ == '__main__':
    pytest.main([__file__])
