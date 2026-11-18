# Code submission scope

`submission/COMP9517_Group104_Code.zip` contains the integrated code project.
Extract it into a new folder and follow the root README. Rebuild with
`python tools/package_submission.py`; the manifest lists every packaged file's
size and SHA256, and the checksum sidecar identifies the ZIP.

## Included computational work

- All five matched configurations: RF, YOLO11s, Faster R-CNN, P2-only and P2 + DySample.
- The previously separate original YOLO branch: baseline/protocol training,
  custom trainer, checkpoint resume and evaluation.
- The original Faster R-CNN branch: dataset/model, training, COCO evaluation,
  runtime benchmarking, distortion augmentation and local swirl-warp evaluation.
- One portable command line, a native-YOLO adapter, common dependency entry,
  all-method serial runner and shared evaluator.
- Frozen-checkpoint diagnostics, corruption comparisons and P2 bootstrap analysis.
- Tests, English execution/integration guides, upstream notices and licenses,
  configuration files, compact comparison records and epoch logs.

See INTEGRATION.md for audited branch commits, source locations and experiment
provenance. Original branch histories are retained in Git. Do not claim that
packaging or integrating code is authorship of all underlying methods.

## Exclusions required for a clean code hand-in

The ZIP excludes trained/initialization weights, model binaries, dataset/input
images, result figures, videos, environments, wheels, Git metadata and large raw
prediction dumps. It also excludes empty historical scaffolds and
Word/PDF/report-authoring artefacts. The complete Git repository retains the
historical research records outside this clean code archive.

Initial weights download during setup. Reproduction creates fresh checkpoints
and predictions. Existing comparison tables alone do not provide the per-image
predictions needed for a fresh bootstrap analysis; obtain the appropriate run
directory from the team or reproduce it.

## Submission tasks outside this code integration

The presentation video is not yet recorded according to the team's latest
status. Final English report readiness has not been verified here. Check the
course specification's report/video requirements and consistency with the
chosen experiment. One designated member must submit the report, video and
source ZIP through the course submission system; GitHub upload is separate.

The supplied specification limits code ZIP to 25 MB, report PDF to 10 MB and MP4
to 100 MB. The source builder enforces the stricter decimal 25,000,000-byte cap.
One seed per model and two dense test images still limit the conclusions;
integration does not add training seeds or new experimental evidence.
