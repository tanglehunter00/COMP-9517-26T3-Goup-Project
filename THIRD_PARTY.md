# Third-party sources and attribution

The project combines existing methods with task-specific implementation,
integration and experiments. P2 and DySample are not claimed as newly invented
methods. Preserve these notices and the bundled licenses when redistributing.

| Component | Source and use |
| --- | --- |
| Ultralytics 8.3.221 | https://github.com/ultralytics/ultralytics/tree/v8.3.221 ; model construction, losses, augmentation and utilities; AGPL-3.0 |
| YOLO11 model YAML | The P2 configurations adapt `ultralytics/cfg/models/11/yolo11.yaml` in the pinned release |
| DySample | https://github.com/tiny-smart/dysample/blob/main/dysample.py ; vendored implementation under MIT |
| PyTorch / torchvision | https://pytorch.org/ and https://github.com/pytorch/vision ; tensor/autograd operations and Faster R-CNN ResNet50-FPN V2 |
| scikit-learn | https://scikit-learn.org/ ; Random Forest training |
| scikit-image | https://scikit-image.org/ ; reference HOG features |
| OpenCV | https://opencv.org/ ; image loading, resizing, colour conversion and corruption |
| NumPy / Pillow | https://numpy.org/ and https://python-pillow.org/ ; numerical and image metadata operations |
| COCO evaluation | https://github.com/cocodataset/cocoapi ; pycocotools evaluation |
| faster-coco-eval | https://github.com/MiXaiLL76/faster_coco_eval ; compiled evaluation backend with parity tests |
| Other runtime utilities | PyYAML, joblib, psutil, threadpoolctl and their transitive dependencies retain their respective upstream licenses |

`Yolo Improeved/vendor/SOURCES.md` lists exact upstream URLs.
`ULTRALYTICS_LICENSE` and `DYSAMPLE_LICENSE` are included in that directory.
`SHA256.json` pins the vendored DySample file, official YOLO checkpoint and
Ultralytics wheel. The local DySample hash is checked after normalizing CRLF to LF.

## Downloaded assets, excluded from the code archive

- COCO YOLO11s initialization:
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
- torchvision Faster R-CNN ResNet50-FPN V2 COCO initialization:
  https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
- Dataset: course-provided Beach Bird Detection Dataset v1.0; Kaggle source
  https://www.kaggle.com/datasets/liugalhigh/beach-bird .

The project archive is source code, not a redistribution of these data or model
weights. The final academic report must also include the appropriate method and
dataset citations. This notice is not a replacement for the report bibliography.
