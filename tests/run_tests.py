from numpy.testing import run_module_suite
import sys

# from test_astrometry import *
from test_simulator import *
from test_extractor import *
from test_fitter import *

if __name__ == "__main__":
    # Run tests
    args = sys.argv

    # If no args were specified, add arg to only do non-slow tests
    if len(args) == 1:
        print("Running tests that are not tagged as 'slow'. "
              "Use '--all' to run all tests.")
        args.append("-a!slow")
    run_module_suite(argv=args)
