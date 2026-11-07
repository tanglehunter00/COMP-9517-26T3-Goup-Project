import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vendor.dysample import DySample
from model import build_model
from evaluate import evaluate
from train import lr_at


class PipelineTests(unittest.TestCase):
    def test_dysample_backward_rectangular(self):
        x = torch.randn(2, 16, 5, 7, requires_grad=True)
        m = DySample(16)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 16, 10, 14))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(m.offset.weight.grad.abs().sum().item(), 0)

    def test_heads_and_transfer(self):
        for ablation in [False, True]:
            m, report = build_model(ablation, Path(__file__).resolve().parents[1]/'checkpoints/yolo11s.pt')
            self.assertEqual(m.stride.tolist(), [4., 8., 16., 32.])
            self.assertGreater(len(report['loaded']), 400)
            m.eval()
            with torch.no_grad():
                result = m(torch.rand(1, 3, 64, 64))[0]
            self.assertEqual(tuple(result.shape), (1, 5, 340))

    def test_perfect_and_empty_metrics(self):
        gt = dict(images=[dict(id=1, width=100, height=100), dict(id=2, width=100, height=100)],
                  annotations=[dict(id=1, image_id=1, category_id=0, bbox=[10,10,10,10], area=100, iscrowd=0)],
                  categories=[dict(id=0, name='bird')], info={})
        rows = [dict(image_id=1, category_id=0, bbox=[10,10,10,10], score=.9)]
        good = evaluate(gt, rows)
        self.assertAlmostEqual(good['AP50'], 1.)
        self.assertAlmostEqual(good['AP-small'], 1.)
        self.assertEqual(good['F1'], 1.)
        self.assertEqual(good['false_positives_per_negative_image'], 0.)
        self.assertIsNone(good['density']['dense']['AP50'])
        self.assertEqual(evaluate(gt, [])['AP50'], 0.)
        bad = evaluate(gt, rows+[dict(image_id=2, category_id=0, bbox=[10,10,10,10], score=.9)])
        self.assertEqual(bad['false_positives_per_negative_image'], 1.)

    def test_scheduler_endpoints(self):
        self.assertAlmostEqual(lr_at(0, 1000, 30), .0001)
        self.assertAlmostEqual(lr_at(29, 1000, 30), .001)
        self.assertAlmostEqual(lr_at(999, 1000, 30), .00001)


if __name__ == '__main__':
    unittest.main()
