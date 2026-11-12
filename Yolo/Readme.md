# M2: standard YOLO11s

Uses the same non-EMA training loop, augmentation, checkpoint selection, evaluator,
and inference as M4; only the architecture differs. Compatible COCO tensors are
transferred directly without P2 branch remapping. All layers train.

From repository root, after installing `requirements.txt` and downloading assets:

```powershell
python Yolo/train.py --root <dataset> --out "Yolo Improeved/runs/m2"
python Yolo/train.py --root <dataset> --out "Yolo Improeved/runs/m2" --resume "Yolo Improeved/runs/m2/last.pt"
python "Yolo Improeved/infer.py" --root <dataset> --weights "Yolo Improeved/runs/m2/best.pt" --split val --out "Yolo Improeved/runs/m2/val.csv"
```

After all configurations complete, use `--split test --benchmark` for final test
export and timing. Evaluate CSVs with `Yolo Improeved/evaluate.py`. Fixed full runs
use 100 epochs; `--smoke` is a separate two-epoch diagnostic only.
