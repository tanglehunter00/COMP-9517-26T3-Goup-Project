import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import build_model, freeze_bn, lr_at, Birds, predict_batch, update_group


class ToyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad(): self.linear.weight.fill_(1.); self.linear.bias.zero_()

    def forward(self, images, targets):
        torch.rand(1, device=images[0].device)
        return {'loss': self.linear(self.bn(torch.stack(images))).float().square().mean()}


class ResnetTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP test')
    def test_amp_overflow_replays_group_without_extra_updates(self):
        device = torch.device('cuda')
        torch.manual_seed(42)
        model = ToyDetector().to(device)
        reference = ToyDetector().to(device); reference.load_state_dict(model.state_dict())
        batches = [([torch.tensor([1.,2.],device=device),torch.tensor([3.,4.],device=device)], [])]*2
        optimizer = torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        expected_optimizer = torch.optim.SGD(reference.parameters(),lr=.01,momentum=.9)
        scaler = torch.amp.GradScaler('cuda',init_scale=2.**20)
        expected_scaler = torch.amp.GradScaler('cuda',init_scale=1.)
        rng = torch.cuda.get_rng_state_all()
        _, count, retries = update_group(model,batches,optimizer,scaler,device)
        actual_rng = torch.cuda.get_rng_state_all()
        torch.cuda.set_rng_state_all(rng)
        update_group(reference,batches,expected_optimizer,expected_scaler,device)
        self.assertGreater(retries,0)
        self.assertEqual(count,4)
        self.assertEqual(model.bn.num_batches_tracked.item(),2)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual_rng,torch.cuda.get_rng_state_all())))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,reference.state_dict()[key],atol=1e-5,rtol=1e-5)

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
