"""Checks for assignment, empty targets and paired resampling regressions."""
import importlib.util
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location('p2_analysis', Path(__file__).resolve().parents[1] / 'analyze_p2_ablation.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class P2AnalysisTests(unittest.TestCase):
    def test_csv_and_gzip_inputs_match_and_validate_image_ids(self):
        content = 'image_id,category_id,x,y,width,height,score\na,0,0,0,10,10,0.9\n'
        truth = {'a': np.array([[0., 0., 10., 10.]]), 'empty': np.empty((0, 4))}
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / 'preds.csv'
            compressed = Path(directory) / 'preds.csv.gz'
            plain.write_text(content, encoding='utf-8')
            with gzip.open(compressed, 'wt', encoding='utf-8') as handle:
                handle.write(content)
            for path in [plain, compressed]:
                predictions = analysis.load_predictions(path, truth, .25)
                self.assertEqual(predictions['empty'], [])
                self.assertEqual(len(predictions['a']), 1)
                self.assertEqual(analysis.match_counts(truth['a'], predictions['a'], .5), [1, 0, 0, 1, 1])
            plain.write_text(content.replace('a,0,', 'unknown,0,'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unexpected image'):
                analysis.load_predictions(plain, truth, .25)

    def test_duplicate_predictions_and_negative_images(self):
        box = np.array([0., 0., 10., 10.])
        predictions = [(.9, box), (.8, box)]
        self.assertEqual(analysis.match_counts(box[None, :], predictions, .5), [1, 1, 0, 1, 1])
        self.assertEqual(analysis.match_counts(np.empty((0, 4)), predictions, .5), [0, 2, 0, 0, 0])

    def test_iou_threshold_changes_localisation_match(self):
        truth = np.array([[0., 0., 10., 10.]])
        predictions = [(.9, np.array([0., 0., 6., 10.]))]
        self.assertEqual(analysis.match_counts(truth, predictions, .5)[0], 1)
        self.assertEqual(analysis.match_counts(truth, predictions, .75)[0], 0)

    def test_identical_models_have_zero_paired_intervals(self):
        # Different per-image recall makes independent resampling detectable.
        images = np.array([[1, 0, 0, 1, 1], [0, 0, 1, 0, 1]])
        counts = np.tile(images, (3, 2, 1, 1))
        first = analysis.paired_bootstrap(counts, 203, 42)
        self.assertEqual(first, analysis.paired_bootstrap(counts, 203, 42))
        for row in first:
            self.assertEqual((row['difference_pp'], row['lower_pp'], row['upper_pp']), (0., 0., 0.))
            self.assertEqual(row['valid_replicates'], 203)

    def test_absent_small_targets_produce_undefined_intervals(self):
        images = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
        counts = np.tile(images, (3, 2, 1, 1))
        for row in analysis.paired_bootstrap(counts, 25, 42):
            if row['metric'] == 'small_recall':
                self.assertIsNone(row['difference_pp'])
                self.assertIsNone(row['lower_pp'])
                self.assertIsNone(row['upper_pp'])
                self.assertEqual(row['valid_replicates'], 0)


if __name__ == '__main__':
    unittest.main()
