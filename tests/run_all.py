import os
import unittest

loader = unittest.TestLoader()
tests = loader.discover(os.path.dirname(__file__), "*.py")
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
