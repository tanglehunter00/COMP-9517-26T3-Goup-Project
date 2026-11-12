"""M2 entry point; shares the matched non-EMA M4 training protocol."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    shared = Path(__file__).resolve().parents[1] / 'Yolo Improeved'
    sys.path.insert(0, str(shared))
    sys.argv.extend(['--baseline'])
    runpy.run_path(str(shared / 'train.py'), run_name='__main__')
