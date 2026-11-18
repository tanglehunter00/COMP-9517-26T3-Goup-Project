"""CPU analysis of saved M2, P2-only and M4 predictions.

Uses the score-ordered one-to-one matching rule in Yolo Improeved/evaluate.py.
Outputs fixed-confidence recall and paired image-bootstrap intervals.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ('m2', 'p2-only', 'm4')
IOU_THRESHOLDS = (.5, .75)
COMPARISONS = (('p2-only', 'm2'), ('m4', 'p2-only'), ('m4', 'm2'))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_truth(root, split):
    paths = sorted(p for p in (root / 'images' / split).glob('*')
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Expected nonempty image directory with unique image stems')
    truth = {}
    fingerprint = hashlib.sha256()
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        label = root / 'labels' / split / (path.stem + '.txt')
        label_bytes = label.read_bytes()
        values = np.empty((0, 5), dtype=np.float32)
        if label_bytes.strip():
            values = np.loadtxt(label, ndmin=2, dtype=np.float32)
        if values.shape[1] != 5 or not np.isfinite(values).all():
            raise ValueError(f'Invalid label array in {label}')
        if np.any(values[:, 0] != 0) or np.any(values[:, 3:] <= 0):
            raise ValueError(f'Expected bird class 0 and positive box sizes in {label}')
        low = values[:, 1:3] - values[:, 3:] / 2
        high = values[:, 1:3] + values[:, 3:] / 2
        if np.any(low < -1e-6) or np.any(high > 1 + 1e-6):
            raise ValueError(f'Out-of-bounds labels in {label}')
        # Preserve the shared evaluator's float32 label-to-pixel arithmetic.
        boxes = np.array([
            [(xc - bw / 2) * width, (yc - bh / 2) * height, bw * width, bh * height]
            for _, xc, yc, bw, bh in values
        ], dtype=float).reshape(-1, 4)
        truth[path.stem] = boxes
        fingerprint.update(f'{path.name}\0{width}\0{height}\0'.encode())
        fingerprint.update(label_bytes)
        fingerprint.update(b'\0')
    return truth, fingerprint.hexdigest()


def load_predictions(path, truth, confidence):
    predictions = {image_id: [] for image_id in truth}
    seen = {image_id: 0 for image_id in truth}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'image_id', 'category_id', 'x', 'y', 'width', 'height', 'score'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing prediction columns in {path}')
        for row in reader:
            image_id = row['image_id']
            if image_id not in truth or row['category_id'] != '0':
                raise ValueError(f'Unexpected image or category in {path}')
            box = np.array([float(row[k]) for k in ['x', 'y', 'width', 'height']])
            score = float(row['score'])
            if not np.isfinite([*box, score]).all() or min(box[2:]) <= 0 or not 0 <= score <= 1:
                raise ValueError(f'Invalid prediction values in {path}')
            seen[image_id] += 1
            if seen[image_id] > 300:
                raise ValueError(f'More than 300 predictions for {image_id}')
            if score >= confidence:
                predictions[image_id].append((score, box))
    return {key: sorted(rows, key=lambda row: -row[0]) for key, rows in predictions.items()}


def match_counts(boxes, predictions, threshold):
    """Return TP, FP, FN, small TP and small GT for one image."""
    used = np.zeros(len(boxes), dtype=bool)
    false_positives = 0
    for _, box in predictions:
        intersection = np.maximum(
            0, np.minimum(box[:2] + box[2:], boxes[:, :2] + boxes[:, 2:])
            - np.maximum(box[:2], boxes[:, :2])
        ).prod(1)
        overlaps = intersection / np.maximum(
            box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - intersection, 1e-9
        )
        candidates = np.flatnonzero((~used) & (overlaps >= threshold))
        if candidates.size:
            used[candidates[np.argmax(overlaps[candidates])]] = True
        else:
            false_positives += 1
    small = boxes[:, 2] * boxes[:, 3] < 32 ** 2
    return [int(used.sum()), false_positives, len(boxes) - int(used.sum()),
            int(used[small].sum()), int(small.sum())]


def recall_values(totals):
    """Undefined recall is NaN so empty bootstrap denominators can be excluded."""
    denominators = np.stack((totals[..., 0] + totals[..., 2], totals[..., 4]), axis=-1)
    numerators = totals[..., [0, 3]]
    return np.divide(numerators, denominators,
                     out=np.full(numerators.shape, np.nan), where=denominators > 0)


def paired_bootstrap(counts, samples, seed):
    """Pair every model on the same resampled images, including empty images."""
    rng = np.random.default_rng(seed)
    draws = np.empty((len(METHODS), len(IOU_THRESHOLDS), samples, 2))
    image_count = counts.shape[2]
    # Small chunks keep memory bounded when many replicates are requested.
    for start in range(0, samples, 100):
        stop = min(start + 100, samples)
        indices = rng.integers(0, image_count, size=(stop - start, image_count))
        for method in range(len(METHODS)):
            for threshold in range(len(IOU_THRESHOLDS)):
                totals = counts[method, threshold][indices].sum(axis=1)
                draws[method, threshold, start:stop] = recall_values(totals)
    points = recall_values(counts.sum(axis=2))
    intervals = []
    for target, reference in COMPARISONS:
        target_index, reference_index = METHODS.index(target), METHODS.index(reference)
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            for metric_index, metric in enumerate(('all_recall', 'small_recall')):
                differences = 100 * (draws[target_index, threshold_index, :, metric_index]
                                     - draws[reference_index, threshold_index, :, metric_index])
                valid = differences[np.isfinite(differences)]
                point = 100 * (points[target_index, threshold_index, metric_index]
                               - points[reference_index, threshold_index, metric_index])
                interval = np.quantile(valid, [.025, .975]).tolist() if valid.size else [None, None]
                intervals.append(dict(target=target, reference=reference, iou=threshold, metric=metric,
                                      difference_pp=float(point) if np.isfinite(point) else None,
                                      lower_pp=interval[0], upper_pp=interval[1],
                                      valid_replicates=int(valid.size), requested_replicates=samples))
    return intervals


def verify_reference(directory, split, totals, confidence):
    path = directory / f'{split}_metrics.json'
    if confidence != .25 or not path.is_file():
        return dict(status='unavailable', path=str(path))
    reference = json.loads(path.read_text(encoding='utf-8'))
    actual = totals[:3].tolist()
    expected = [reference[key] for key in ('TP', 'FP', 'FN')]
    if actual != expected:
        raise ValueError(f'Count mismatch for {directory.name}: {actual} versus {expected}')
    return dict(status='matched', path=str(path), sha256=file_hash(path))


def display(value, percent=False):
    if value is None:
        return 'N/A'
    return f'{value * 100:.2f}%' if percent else f'{value:+.2f}'


def markdown_report(result):
    lines = ['# P2 ablation analysis', '', result['label'], '',
             f"Split {result['split']}. Images {result['images']}. Bird boxes {result['objects']}. "
             f"Small boxes {result['small_objects']}. Confidence {result['confidence']}.", '',
             '## Fixed confidence recall', '',
             '| Configuration | All IoU 0.50 | All IoU 0.75 | Small IoU 0.50 | Small IoU 0.75 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        values = result['methods'][method]
        cells = [values['0.5']['recall'], values['0.75']['recall'],
                 values['0.5']['small_recall'], values['0.75']['small_recall']]
        lines.append('| ' + ' | '.join([method] + [display(x, True) for x in cells]) + ' |')
    lines += ['', '## Paired image bootstrap', '',
              f"Percentile intervals from {result['bootstrap_samples']} resamples, seed {result['seed']}. "
              'Differences are target minus reference in percentage points.', '',
              '| Target | Reference | IoU | Recall group | Difference | 95% interval | Valid draws |',
              '| --- | --- | ---: | --- | ---: | --- | ---: |']
    for row in result['bootstrap']:
        interval = f"{display(row['lower_pp'])} to {display(row['upper_pp'])}"
        lines.append(f"| {row['target']} | {row['reference']} | {row['iou']:.2f} | {row['metric']} | "
                     f"{display(row['difference_pp'])} | {interval} | {row['valid_replicates']} |")
    lines += ['', '## Interpretation', '',
              'Matching follows the shared evaluator using descending confidence and one-to-one assignment. '
              'Small targets have original-image box area below 1,024 square pixels. '
              'These are fixed-confidence recall measures. COCO AP-small and AR-small use their own evaluation settings.', '',
              'The bootstrap samples whole images and uses identical draws for every model. '
              'Intervals describe test-image sampling uncertainty for the supplied predictions. '
              'Training-seed variability and the mechanism behind a difference require additional experiments. '
              'Draws containing no relevant ground truth have undefined recall and are excluded from that interval.', '',
              'Input paths, SHA256 fingerprints and reference-count checks are recorded in summary.json.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Dataset containing images/ and labels/')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[2] / 'results')
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    parser.add_argument('--out', required=True, type=Path, help='Directory for JSON, CSV and Markdown output')
    parser.add_argument('--label', required=True, help='Identify the experiment and hardware being analysed')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261118)
    args = parser.parse_args()
    if not .001 <= args.confidence <= 1 or args.samples < 1 or args.seed < 0:
        parser.error('confidence must be in [0.001, 1], samples positive and seed nonnegative')
    truth, fingerprint = load_truth(args.root, args.split)
    counts = np.empty((len(METHODS), len(IOU_THRESHOLDS), len(truth), 5), dtype=np.int64)
    inputs, checks, metrics, per_image = {}, {}, {}, []
    for method_index, method in enumerate(METHODS):
        directory = args.results / method
        path = directory / f'preds_{args.split}.csv.gz'
        if not path.is_file():
            path = directory / f'preds_{args.split}.csv'
        predictions = load_predictions(path, truth, args.confidence)
        inputs[method] = dict(path=str(path.resolve()), sha256=file_hash(path))
        metrics[method] = {}
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            rows = [match_counts(boxes, predictions[image_id], threshold)
                    for image_id, boxes in truth.items()]
            counts[method_index, threshold_index] = rows
            totals = np.sum(rows, axis=0)
            recall = recall_values(totals)
            metrics[method][str(threshold)] = dict(
                TP=int(totals[0]), FP=int(totals[1]), FN=int(totals[2]),
                small_TP=int(totals[3]), small_GT=int(totals[4]),
                recall=float(recall[0]) if np.isfinite(recall[0]) else None,
                small_recall=float(recall[1]) if np.isfinite(recall[1]) else None)
            for image_id, row in zip(truth, rows):
                per_image.append([method, image_id, threshold, *row])
        checks[method] = verify_reference(directory, args.split, counts[method_index, 0].sum(0), args.confidence)
    totals = counts[0, 0].sum(0)
    result = dict(label=args.label, split=args.split, confidence=args.confidence,
                  images=len(truth), objects=int(totals[0] + totals[2]), small_objects=int(totals[4]),
                  dataset_root=str(args.root.resolve()), label_geometry_sha256=fingerprint,
                  bootstrap_samples=args.samples, seed=args.seed, inputs=inputs,
                  reference_checks=checks, methods=metrics,
                  bootstrap=paired_bootstrap(counts, args.samples, args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.out / 'analysis.md').write_text(markdown_report(result), encoding='utf-8')
    with (args.out / 'per_image.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'image_id', 'iou', 'TP', 'FP', 'FN', 'small_TP', 'small_GT'])
        writer.writerows(per_image)
    print(f'Analysed {len(truth)} {args.split} images. Wrote results to {args.out.resolve()}')


if __name__ == '__main__':
    main()
