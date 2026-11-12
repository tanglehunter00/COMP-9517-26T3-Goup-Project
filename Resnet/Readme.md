# M3: Faster R-CNN ResNet50-FPN V2

AMP overflow handling: reduce the loss scale and recompute the same accumulation
group, restoring model buffers and CPU/CUDA RNG before retries. No scheduled
optimizer update is dropped, and BatchNorm statistics advance only once per
accepted microbatch. Finite losses and gradients remain required; persistent
overflow after 16 retries fails explicitly. Logs include AMP retries and scale.

`pipeline.py` implements the root protocol using torch 2.6.0 / torchvision 0.21.0.
Install the matching CUDA wheels before the root requirements. COCO_V1 weights
download automatically. The predictor has background + bird classes. The last
three ResNet stages train; backbone BatchNorm affine parameters and running
statistics stay frozen after each call to `train()`.

Input images and targets are letterboxed onto a 640 canvas with RGB 114; internal
min/max resize is 640. Training uses batch 2, accumulation 2 (including the final
partial group), SGD 0.0025, momentum 0.9, decay 0.0001, one-epoch linear warmup and
cosine decay to 0.000025. Only horizontal flip 0.5 augments training images.
Validation/inference uses batch 8, AMP FP16, RoI hard NMS 0.6 and max 300 outputs;
RPN NMS stays 0.7. There is no second output NMS.

```powershell
python Resnet/pipeline.py train --root <dataset> --out Resnet/runs/m3
python Resnet/pipeline.py train --root <dataset> --out Resnet/runs/m3 --resume Resnet/runs/m3/last.pt
python Resnet/pipeline.py infer --root <dataset> --weights Resnet/runs/m3/best.pt --split val --out Resnet/runs/m3/val.csv
```

After all training completes, infer with `--split test --benchmark`. CSV boxes use
original coordinates and dataset class 0; metrics use the shared COCO evaluator.
Benchmark preloads images, warms up 20 batches, repeats the full set three times,
and labels non-T4 hardware explicitly. Checkpoints save optimizer, AMP, update
schedule, RNG and dataset fingerprint. Best selection is AP50, then AP50:95, then
earlier epoch. Resume requires the same run directory and data fingerprint.

Diagnostics: `--smoke` uses two epochs on synthetic data and marks checkpoints;
inference rejects them unless `--allow-smoke` is explicit. `--stop-after-epoch 1`
exercises interruption and resume. Run tests with
`python -m unittest discover -s Resnet -p test_pipeline.py -v`.
