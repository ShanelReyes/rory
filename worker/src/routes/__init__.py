import os
import sys
from pathlib import Path
path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
print(path_root)
sys.path.append(str(path_root))