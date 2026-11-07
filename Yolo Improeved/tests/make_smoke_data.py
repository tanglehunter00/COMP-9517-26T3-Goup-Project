"""Generate synthetic diagnostics under this method's work/ folder, never real data."""
from pathlib import Path
import cv2
import numpy as np


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]/'work/synthetic'
    rng = np.random.default_rng(42)
    for split, count in [('train', 17), ('val', 8), ('test', 8)]:
        (root/'images'/split).mkdir(parents=True, exist_ok=True)
        (root/'labels'/split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            path = root/'images'/split/f'{i:03}.jpg'
            image = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(path), image)
            (root/'labels'/split/f'{i:03}.txt').write_text('0 0.5 0.5 0.2 0.2\n' if i%3 else '')
    print(root)
