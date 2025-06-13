#!/usr/bin/env python3
import os
import tempfile
import shutil
import unittest
import sys

# Add the parent directory to sys.path so we can import prep_context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prep_context


class TestPrepContextMulti(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories with sample files for testing."""
        # Create two temporary directories
        self.temp_dir1 = tempfile.mkdtemp(prefix="test_dir1_")
        self.temp_dir2 = tempfile.mkdtemp(prefix="test_dir2_")

        # Create sample files in first directory
        self.file1_path = os.path.join(self.temp_dir1, "file1.cs")
        with open(self.file1_path, 'w', encoding='utf-8') as f:
            f.write("// File 1 content\nclass Class1 { }")

        self.file2_path = os.path.join(self.temp_dir1, "file2.py")
        with open(self.file2_path, 'w', encoding='utf-8') as f:
            f.write("# File 2 content\nclass Class2:\n    pass")

        # Create sample files in second directory
        self.file3_path = os.path.join(self.temp_dir2, "file3.cs")
        with open(self.file3_path, 'w', encoding='utf-8') as f:
            f.write("// File 3 content\nclass Class3 { }")

        self.file4_path = os.path.join(self.temp_dir2, "file4.js")
        with open(self.file4_path, 'w', encoding='utf-8') as f:
            f.write("// File 4 content\nfunction test() { }")

        # Create a subdirectory in second directory
        self.subdir = os.path.join(self.temp_dir2, "subdir")
        os.makedirs(self.subdir)
        self.file5_path = os.path.join(self.subdir, "file5.cs")
        with open(self.file5_path, 'w', encoding='utf-8') as f:
            f.write("// File 5 content\nclass Class5 { }")

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir1):
            shutil.rmtree(self.temp_dir1)
        if os.path.exists(self.temp_dir2):
            shutil.rmtree(self.temp_dir2)

    def test_find_files_in_directories_cs_only(self):
        """Test finding .cs files in multiple directories."""
        directories = [self.temp_dir1, self.temp_dir2]
        files = prep_context.find_files_in_directories(
            directories, extensions=['.cs'], recursive=False)

        # Should find file1.cs and file3.cs (but not file5.cs without recursive)
        expected_files = {self.file1_path, self.file3_path}
        actual_files = set(files)

        self.assertEqual(actual_files, expected_files)
        self.assertEqual(len(files), len(actual_files))  # Ensure no duplicates

    def test_find_files_in_directories_recursive(self):
        """Test recursive search in multiple directories."""
        directories = [self.temp_dir1, self.temp_dir2]
        files = prep_context.find_files_in_directories(
            directories, extensions=['.cs'], recursive=True)

        # Should find file1.cs, file3.cs, and file5.cs
        expected_files = {self.file1_path, self.file3_path, self.file5_path}
        actual_files = set(files)

        self.assertEqual(actual_files, expected_files)
        self.assertEqual(len(files), len(actual_files))  # Ensure no duplicates

    def test_find_files_in_directories_multiple_extensions(self):
        """Test finding files with multiple extensions."""
        directories = [self.temp_dir1, self.temp_dir2]
        files = prep_context.find_files_in_directories(
            directories, extensions=['.cs', '.py', '.js'], recursive=False)

        # Should find file1.cs, file2.py, file3.cs, file4.js
        expected_files = {self.file1_path, self.file2_path, self.file3_path, self.file4_path}
        actual_files = set(files)

        self.assertEqual(actual_files, expected_files)
        self.assertEqual(len(files), len(actual_files))  # Ensure no duplicates

    def test_relative_to_roots(self):
        """Test the _relative_to_roots helper function."""
        roots = [self.temp_dir1, self.temp_dir2]

        # Test file from first root
        rel_path1 = prep_context._relative_to_roots(self.file1_path, roots)
        expected1 = os.path.join(os.path.basename(self.temp_dir1), "file1.cs")
        self.assertEqual(rel_path1, expected1)

        # Test file from second root
        rel_path3 = prep_context._relative_to_roots(self.file3_path, roots)
        expected3 = os.path.join(os.path.basename(self.temp_dir2), "file3.cs")
        self.assertEqual(rel_path3, expected3)

        # Test file from subdirectory
        rel_path5 = prep_context._relative_to_roots(self.file5_path, roots)
        expected5 = os.path.join(os.path.basename(self.temp_dir2), "subdir", "file5.cs")
        self.assertEqual(rel_path5, expected5)

    def test_create_context_content_multiple_roots(self):
        """Test creating context content with multiple root directories."""
        directories = [self.temp_dir1, self.temp_dir2]
        files = [self.file1_path, self.file3_path]

        context = prep_context.create_context_content(files, directories)

        # Check that both directory names appear in headers
        dir1_name = os.path.basename(self.temp_dir1)
        dir2_name = os.path.basename(self.temp_dir2)

        self.assertIn(f"{dir1_name}\\file1.cs", context)  # Windows path separator
        self.assertIn(f"{dir2_name}\\file3.cs", context)  # Windows path separator

        # Check that file contents are present
        self.assertIn("// File 1 content", context)
        self.assertIn("// File 3 content", context)
        self.assertIn("class Class1", context)
        self.assertIn("class Class3", context)

        # Check header format
        expected_header1 = f"--------------\n{dir1_name}\\file1.cs\n--------------"
        expected_header3 = f"--------------\n{dir2_name}\\file3.cs\n--------------"
        self.assertIn(expected_header1, context)
        self.assertIn(expected_header3, context)

    def test_create_context_content_backward_compatibility(self):
        """Test that single directory (string) still works for backward compatibility."""
        files = [self.file1_path]

        # Pass single directory as string (old API)
        context = prep_context.create_context_content(files, self.temp_dir1)

        # Should work just like before
        self.assertIn("file1.cs", context)
        self.assertIn("// File 1 content", context)
        self.assertIn("class Class1", context)

    def test_empty_directories_list(self):
        """Test behavior with empty directories list."""
        files = prep_context.find_files_in_directories([], extensions=['.cs'])
        self.assertEqual(files, [])

    def test_nonexistent_directory(self):
        """Test behavior with nonexistent directory."""
        nonexistent = "/path/that/does/not/exist"
        directories = [self.temp_dir1, nonexistent]

        # Should still find files from the valid directory
        files = prep_context.find_files_in_directories(
            directories, extensions=['.cs'], recursive=False)

        expected_files = {self.file1_path}
        actual_files = set(files)
        self.assertEqual(actual_files, expected_files)


if __name__ == '__main__':
    unittest.main()
