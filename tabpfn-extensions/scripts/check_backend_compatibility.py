#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys


def scan_for_device_hardcoding(file_path: str) -> list[int]:
    """Scan file for hardcoded device parameters.

    Args:
        file_path: Path to the file to scan

    Returns:
        List of line numbers where 'device="cpu"' or similar is found
    """
    device_patterns = [
        r'device\s*=\s*[\'"]cpu[\'"]',
        r'device\s*=\s*[\'"]cuda[\'"]',
    ]

    hardcoded_lines = []

    with open(file_path) as f:
        for i, line in enumerate(f.readlines(), 1):
            for pattern in device_patterns:
                if re.search(pattern, line):
                    hardcoded_lines.append(i)

    return hardcoded_lines


def scan_for_import_try_except(file_path: str) -> bool:
    """Check if file has proper try-except for TabPFN imports.

    Args:
        file_path: Path to the file to scan

    Returns:
        True if file has proper try-except pattern, False otherwise
    """
    with open(file_path) as f:
        content = f.read()

    # Check for presence of try-except pattern for imports
    import_pattern = re.compile(
        r"try\s*:"
        r"(?:.*?)"  # Match content in between
        r"from\s+tabpfn\s+import"
        r"(?:.*?)"  # Match any content
        r"except\s+ImportError",
        re.DOTALL,
    )

    return bool(import_pattern.search(content))


def find_py_files(directory: str, exclude_dirs: set[str]) -> list[str]:
    """Find all Python files in the directory.

    Args:
        directory: Directory to search for Python files
        exclude_dirs: Set of directory names to exclude from search

    Returns:
        List of paths to Python files
    """
    py_files = []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    return py_files


def check_compatibility():
    """Check all Python files for compatibility issues."""
    # Directory to scan
    src_dir = "src/tabpfn_extensions"

    # Directories to exclude
    exclude_dirs = {"__pycache__", ".git", "venv", "tests", "examples"}

    # Find all Python files
    py_files = find_py_files(src_dir, exclude_dirs)

    # Results
    issues = {
        "device_hardcoding": [],
        "missing_import_compatibility": [],
    }

    # Scan each file
    for file_path in py_files:
        hardcoded_lines = scan_for_device_hardcoding(file_path)
        if hardcoded_lines:
            issues["device_hardcoding"].append((file_path, hardcoded_lines))

        has_import_compatibility = scan_for_import_try_except(file_path)
        if not has_import_compatibility and "import" in open(file_path).read():
            issues["missing_import_compatibility"].append(file_path)

    # Print results
    print(f"Scanned {len(py_files)} Python files")

    print("\nFiles with hardcoded device parameters:")
    for file_path, lines in issues["device_hardcoding"]:
        print(f"  {file_path} (lines: {', '.join(map(str, lines))})")

    print("\nFiles missing TabPFN backend compatibility imports:")
    for file_path in issues["missing_import_compatibility"]:
        print(f"  {file_path}")

    # Return non-zero exit code if issues were found
    if issues["device_hardcoding"] or issues["missing_import_compatibility"]:
        print(
            "\n⚠️  Backend compatibility issues found. Please fix them before committing.",
        )
        return 1
    else:
        print("\n✅ No backend compatibility issues found.")
        return 0


if __name__ == "__main__":
    sys.exit(check_compatibility())
