import os
import sys
from pathlib import Path
path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
newPath        = str(path_root) +"/src"
print(newPath)
sys.path.append(newPath)