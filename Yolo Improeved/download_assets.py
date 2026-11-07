"""Download official COCO initialization and the pinned upstream source wheel."""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    (HERE/'checkpoints').mkdir(exist_ok=True)
    checkpoint = HERE/'checkpoints/yolo11s.pt'
    if not checkpoint.exists():
        tmp = checkpoint.with_suffix('.download')
        urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt', tmp)
        tmp.replace(checkpoint)
    wheel = HERE/'vendor/ultralytics-8.3.221-py3-none-any.whl'
    if not wheel.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'download', 'ultralytics==8.3.221',
                        '--no-deps', '--dest', str(HERE/'vendor')], check=True)
    expected = json.loads((HERE/'vendor/SHA256.json').read_text())
    for name, digest in expected.items():
        path = HERE / name.replace('\\', '/')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f'SHA256 mismatch: {path}')
        print(f'OK {path.name}: {actual}')


if __name__ == '__main__':
    main()
