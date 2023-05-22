import os
import sys
import unittest
from pathlib import Path
path_root = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
sys.path.append(str(path_root))