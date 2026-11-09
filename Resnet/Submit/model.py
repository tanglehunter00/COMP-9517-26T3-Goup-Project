import torch
from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import FrozenBatchNorm2d


PROTOCOL_KWARGS = dict(
    min_size=640, max_size=640,
    rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
    rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000,
    rpn_nms_thresh=0.7,
    box_batch_size_per_image=512, box_positive_fraction=0.25,
    box_score_thresh=0.001,
    box_nms_thresh=0.6,
    box_detections_per_img=300,
)


def freeze_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                for key in ("weight", "bias", "running_mean", "running_var"):
                    getattr(frozen, key).copy_(getattr(child, key))
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)


def get_faster_rcnn_model(num_classes=2, pretrained=True):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        trainable_backbone_layers=3, **PROTOCOL_KWARGS)
    freeze_batchnorm(model.backbone.body)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
