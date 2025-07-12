# test_utilities.py

"""Test suite for v7p3r_utilities module."""

import unittest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from v7p3r_utilities import (
    ensure_directory_exists,
    get_timestamp,
    get_resource_path,
    get_project_root
)

class TestV7P3RUtilities(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_ensure_directory_exists_str(self):
        """Test directory creation with string path."""
        test_path = f"{self.temp_dir}/test_dir"
        ensure_directory_exists(test_path)
        self.assertTrue(Path(test_path).exists())
        self.assertTrue(Path(test_path).is_dir())
    
    def test_ensure_directory_exists_path(self):
        """Test directory creation with Path object."""
        test_path = Path(self.temp_dir) / "test_dir"
        ensure_directory_exists(test_path)
        self.assertTrue(test_path.exists())
        self.assertTrue(test_path.is_dir())
    
    def test_ensure_directory_exists_nested(self):
        """Test nested directory creation."""
        test_path = Path(self.temp_dir) / "test_dir" / "nested" / "path"
        ensure_directory_exists(test_path)
        self.assertTrue(test_path.exists())
        self.assertTrue(test_path.is_dir())
    
    def test_get_timestamp_format(self):
        """Test timestamp format."""
        timestamp = get_timestamp()
        # Try to parse the timestamp
        try:
            datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%SS")
        except ValueError:
            self.fail("Timestamp has incorrect format")
    
    def test_get_resource_path_str(self):
        """Test resource path resolution with string."""
        resource = get_resource_path("test.txt")
        self.assertIsInstance(resource, Path)
        self.assertTrue(str(resource).endswith("test.txt"))
    
    def test_get_resource_path_path(self):
        """Test resource path resolution with Path."""
        resource = get_resource_path(Path("test.txt"))
        self.assertIsInstance(resource, Path)
        self.assertTrue(str(resource).endswith("test.txt"))
    
    def test_get_project_root(self):
        """Test project root path."""
        root = get_project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue((root / "v7p3r.py").exists())

if __name__ == '__main__':
    unittest.main()
