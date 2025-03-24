"""
Test runner for ALF-T5 project
Run this script to execute all tests or specific test modules
"""

import unittest
import sys
import os

def run_tests(pattern=None):
    """
    Run tests with optional pattern filter
    
    Args:
        pattern: Optional pattern to filter test files
    """
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if pattern:
        test_suite = unittest.defaultTestLoader.discover('.', pattern=pattern)
    else:
        test_suite = unittest.defaultTestLoader.discover('.')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        print(f"Running tests matching pattern: {pattern}")
        success = run_tests(pattern)
    else:
        print("Running all tests")
        success = run_tests()
    
    sys.exit(0 if success else 1) 