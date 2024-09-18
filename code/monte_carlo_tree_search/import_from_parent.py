import sys
import os

def import_from_parent():
    """
    Enables file to import from parent directory.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))