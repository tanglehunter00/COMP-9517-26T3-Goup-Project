# Upstream sources

- Ultralytics 8.3.221: https://github.com/ultralytics/ultralytics/tree/v8.3.221
  Official PyPI wheel; AGPL-3.0, full license in ULTRALYTICS_LICENSE.
  Configs adapt ultralytics/cfg/models/11/yolo11.yaml.
- DySample: https://github.com/tiny-smart/dysample/blob/main/dysample.py
  Downloaded verbatim; MIT, full license in DYSAMPLE_LICENSE.
  Exact content is pinned by SHA256.json; no runtime source downloads.
- COCO initialization: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
  Exact checkpoint hash in SHA256.json.

download_assets.py checks all three hashes. Downloaded wheel and extracted
source are ignored local assets; install the pinned package to run.
