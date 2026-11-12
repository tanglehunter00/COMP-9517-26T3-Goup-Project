"""Verify accelerated evaluation and compiled NMS preserve the reference rules."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torchvision
import evaluate as metrics
from ultralytics.utils.nms import TorchNMS
from faster_coco_eval import COCO, COCOeval_faster


class BackendTests(unittest.TestCase):
    def test_compiled_greedy_nms(self):
        generator = torch.Generator().manual_seed(42)
        origins = torch.rand(100, 2, generator=generator)*100
        boxes = torch.cat([origins, origins+torch.rand(100, 2, generator=generator)*30+1], 1)
        scores = torch.rand(100, generator=generator)
        self.assertTrue(torch.equal(TorchNMS.nms(boxes, scores, .6), torchvision.ops.nms(boxes, scores, .6)))

    def test_coco_backend_parity(self):
        gt = dict(images=[dict(id=i, width=640, height=640) for i in range(1,5)],
            categories=[dict(id=0, name='bird')], info={}, annotations=[
                dict(id=1, image_id=1, category_id=0, bbox=[0,0,20,20], area=400, iscrowd=0),
                dict(id=2, image_id=2, category_id=0, bbox=[10,10,50,50], area=2500, iscrowd=0),
                dict(id=3, image_id=3, category_id=0, bbox=[20,20,100,100], area=10000, iscrowd=0)])
        rows = [dict(image_id=i, category_id=0, bbox=box, score=score) for i,box,score in [
            (1,[1,1,20,20],.9), (1,[100,100,10,10],.3), (2,[10,10,50,50],.8),
            (3,[20,20,100,100],.7), (4,[10,10,10,10],.5)]]
        from pycocotools.coco import COCO as ReferenceCOCO
        from pycocotools.cocoeval import COCOeval as ReferenceEval
        original = metrics.COCO, metrics.COCOeval
        try:
            metrics.COCO, metrics.COCOeval = ReferenceCOCO, ReferenceEval
            reference = metrics.evaluate(copy.deepcopy(gt), copy.deepcopy(rows))
            metrics.COCO, metrics.COCOeval = COCO, COCOeval_faster
            accelerated = metrics.evaluate(copy.deepcopy(gt), copy.deepcopy(rows))
        finally:
            metrics.COCO, metrics.COCOeval = original
        for key in ['AP50','AP50:95','AP75','AP-small','AR-small','precision','recall','F1']:
            self.assertAlmostEqual(reference[key], accelerated[key], places=12)
        for group in reference['density']:
            for key, value in reference['density'][group].items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, accelerated['density'][group][key], places=12)
                else:
                    self.assertEqual(value, accelerated['density'][group][key])


if __name__=='__main__':
    unittest.main()
