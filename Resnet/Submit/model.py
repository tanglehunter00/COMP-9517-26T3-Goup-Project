import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(weights_path=None, num_classes=2):
    """
    Builds and configures Faster R-CNN ResNet50-FPN V2 according to the fixed protocol:
    - RPN pre/post-NMS limits: 2000/1000 (train), 1000/1000 (eval)
    - RPN NMS IoU: 0.7
    - RoI sampling: 512 proposals, positive fraction 0.25
    - RoI output NMS: 0.6
    """
    # Initialize with keyword arguments for RPN and RoI settings
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        box_nms_thresh=0.6
    )

    # Load official COCO checkpoint weights if provided
    if weights_path:
        print(f"Loading checkpoint weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # Configure trainable backbone layers = 3 (freeze stem, layer1, and layer2)
    for name, parameter in model.backbone.named_parameters():
        if any(prefix in name for prefix in ["conv1", "bn1", "body.layer1", "body.layer2"]):
            parameter.requires_grad = False

    # Freeze backbone BatchNorm running stats and affine parameters explicitly
    def set_bn_eval_and_freeze(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    model.backbone.apply(set_bn_eval_and_freeze)

    # Replace the box predictor head for 2 classes (0: background, 1: bird)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model