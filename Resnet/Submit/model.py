import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(weights_path=None, num_classes=2):
    """
    Builds and configures Faster R-CNN ResNet50-FPN V2 according to the fixed protocol:
    - Trainable backbone layers = 3
    - Freezes backbone BatchNorm running statistics and affine parameters
    - Replaces predictor for background + bird (num_classes = 2)
    - Configures RPN pre/post-NMS limits, IoU thresholds, and RoI sampling
    """
    # Initialize Faster R-CNN V2 with 3 trainable backbone layers
    # (trainable_backbone_layers=3 freezes layers 0, 1, and part of 2 in resnet50_fpn_v2)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None, trainable_backbone_layers=3
    )

    # Load official COCO checkpoint weights if provided
    if weights_path:
        print(f"Loading checkpoint weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # Freeze backbone BatchNorm running stats and affine parameters explicitly
    def set_bn_eval_and_freeze(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    model.backbone.apply(set_bn_eval_and_freeze)

    # Replace the box predictor head for our 2 classes (0: background, 1: bird)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Configure RPN parameters as per spec:
    # - pre/post-NMS limits: 2000/1000 during training, 1000/1000 during evaluation
    # - RPN NMS IoU: 0.7
    model.rpn.pre_nms_top_n = {"training": 2000, "testing": 1000}
    model.rpn.post_nms_top_n = {"training": 1000, "testing": 1000}
    model.rpn.nms_thresh = 0.7

    # Configure RoI sampling and NMS as per spec:
    # - RoI sampling 512 proposals per image with positive fraction 0.25
    # - RoI output greedy hard NMS IoU 0.6
    model.roi_heads.batch_size_per_image = 512
    model.roi_heads.positive_fraction = 0.25
    model.roi_heads.nms_thresh = 0.6  # RoI output NMS

    return model
