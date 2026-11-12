import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import build_model, freeze_bn, lr_at, Birds, predict_batch


class ResnetTests(unittest.TestCase):
    def test_protocol_and_backbone_freeze(self):
        model = build_model(False)
        self.assertEqual(model.transform.min_size, (640,))
        self.assertEqual(model.transform.max_size, 640)
        self.assertEqual(model.roi_heads.nms_thresh, .6)
        self.assertEqual(model.roi_heads.detections_per_img, 300)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        model.train(); freeze_bn(model)
        for name, parameter in model.backbone.body.named_parameters():
            if name.startswith(('conv1.', 'bn1.', 'layer1.')):
                self.assertFalse(parameter.requires_grad)
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertFalse(module.training)
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_learning_rate(self):
        self.assertAlmostEqual(lr_at(0, 1000, 10), .00025)
        self.assertAlmostEqual(lr_at(9, 1000, 10), .0025)
        self.assertAlmostEqual(lr_at(999, 1000, 10), .000025)


if __name__ == '__main__':
    unittest.main()
