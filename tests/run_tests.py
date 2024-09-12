import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    print("Running tests...")
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    print(f"Tests run: {result.testsRun}, Errors: {len(result.errors)}, Failures: {len(result.failures)}")
