# Serial all-method execution

Use `python project.py run --root DATASET --run RUN`.
Preview without dependencies or side effects with
`python project.py --dry-run run --root DATASET --run RUN`.

The queue runs M2, M3, P2-only and M4 training sequentially, then RF sample
construction and two-round fitting. After all training completes it runs
validation/test inference, the shared evaluator and full-test timing.
A run-directory file lock prevents two queue owners. Stage/log/checkpoint
records support resuming the same run. Never delete a lock file to bypass an
active owner. Use a new directory for different data or settings.

`comparison.csv` and JSON summarize the completed run; `status.json`,
`stages.json`, per-stage logs, per-method loss logs, epoch CSVs and source
snapshots retain the execution record. Two CPU threads are configured inside
queue children. RF's CPU runtime and deep GPU throughput identify different
hardware. Actual GPU names are recorded.

The direct runner retains its RTX-5090-family guard unless
`--allow-other-cuda` is supplied. The common entry supplies that flag for portable
CUDA execution. Training still requires a CUDA device, the ML dependencies,
initial weights, official dataset and sufficient checkpoint storage.
