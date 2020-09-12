# Pytest runs this this file to add the path of this top level directory to the path
# enabling imports for python files in nested test folders

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))