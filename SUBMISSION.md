# Submission scope and status

## Source deliverable

`submission/COMP9517_Group104_Code.zip` is the integrated English code submission.
Extract it into a new folder, then follow the included root README. Its
`MANIFEST.json` lists every packaged file's size and SHA256. Sidecars next to the
ZIP record the same manifest and the archive checksum.

Rebuild from the repository root with `python tools/package_submission.py`.
The builder uses an explicit allowlist and refuses missing required files,
invalid Python syntax, non-English CJK text, modified vendored DySample content,
and archives larger than 25,000,000 bytes. Generated ZIPs are excluded from their
own input list. Root requirements for the older experiment are excluded.

The code archive includes:
- RF sample building, training, inference and evaluation helper.
- YOLO11s, P2-only and P2 + DySample source/configurations and shared evaluator.
- Faster R-CNN training and inference.
- Serial experiment runner and validation diagnostic/robustness scripts.
- P2 saved-prediction analysis and existing regression tests.
- English execution instructions, dependency profile and third-party notices.
- Compact existing comparison/diagnostic tables and epoch logs.

Excluded by design: dataset/input images, output/result images, trained or
initialization weights, model binaries, virtual environments, package wheels,
large prediction dumps, Git history and report-authoring artefacts.
Initial weights download during setup. Reproduction creates fresh trained
checkpoints and predictions. To analyse the existing RTX 5090 D predictions
without retraining, obtain the full run directory from the team; summary tables
alone are insufficient.

## Completed project work documented in the repository

The team has recorded training/evaluation for RF, YOLO11s, Faster R-CNN, P2-only,
and P2 + DySample. Existing records include test comparisons, epoch logs,
validation interventions and corruption comparisons. The P2 analysis script
supports matched-image comparisons with paired bootstrap intervals. These are
the team's existing results; consolidation does not claim new model training.

## Remaining work before the course submission

- **Presentation video: not yet recorded**, according to the team's latest status.
  Every official member must participate and speak. Check the specification's
  duration, webcam/identity, format, resolution and file-size requirements.
- **Final English report: final readiness has not been verified in this
  consolidation.** The repository contains report material, including Chinese
  documents. Check the final IEEE two-column PDF, page limit, citations and
  agreement with the selected experiment before submission.
- Confirm all official members' contributions and the final report/video files.
  This code integration does not certify individual work performed by others.
- One designated group member must upload the final report, video and source ZIP
  through the course submission system. A GitHub upload does not submit them.

The supplied project specification limits source ZIP to 25 MB, report PDF to
10 MB and video MP4 to 100 MB, with at most 10 minutes of video and 10 main report
pages (references may follow). Recheck the course's current submission page
before the designated member submits.

## Evidence limits retained

One seed per configuration and two dense test images limit claims of stable
superiority. Validation perturbations and offset removal provide conditional
evidence about frozen checkpoints; they do not establish a novel method or a
training-time causal mechanism. P2 and DySample are existing techniques and are
credited as such. Additional seeds remain an extension rather than completed
work.
