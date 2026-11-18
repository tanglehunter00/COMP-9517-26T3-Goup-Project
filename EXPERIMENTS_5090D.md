# Recorded RTX 5090 D comparison

The main comparison records are `reports/MAIN_comparison.csv` and JSON.
They describe existing seed-42 team measurements, not fresh integration runs.
The original feature branches contain separate experiments; see INTEGRATION.md.

For the complete code project install the root requirements, audit the official
dataset, then run:

```text
python project.py run --root DATASET --run runs/reproduction_seed42
python project.py diagnostics --root DATASET --run runs/reproduction_seed42
python project.py robustness --root DATASET --run runs/reproduction_seed42
```

The current common entry permits other CUDA GPUs and records actual hardware.
For an explicit recorded-hardware guard, call `run_serial_benchmark.py` directly
without `--allow-other-cuda`. All matched deep configurations use actual batch 8
and accumulation 2. Original Faster R-CNN's batch-2 T4 records are separate.

Complete environment, commands and limitations are in the root README.
