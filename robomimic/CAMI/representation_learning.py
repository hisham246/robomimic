# i sketched out an algo for the encoders
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from transformers import AutoImageProcessor, AutoModel
 
from robomimic.models.obs_core import EncoderCore
 
 
class DinoV2Core(EncoderCore):
    """
    DINOv2 visual encoder for RoboMimic RGB observations.
    Returns a compact feature vector for each image.
    """
 
    def __init__(
        self,
        input_shape,
        model_name="facebook/dinov2-base",
        feature_dimension=128,
        freeze_backbone=True,
        use_cls_token=True,
    ):
        super().__init__(input_shape=input_shape)
        self.model_name = model_name
        self.feature_dimension = feature_dimension
        self.use_cls_token = use_cls_token
 
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
 
        hidden_dim = self.backbone.config.hidden_size  # 768 for dinov2-base
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dimension),
        )
 
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
 
        self.image_size = int(self.backbone.config.image_size)
        self.register_buffer(
            "image_mean",
            torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.processor.image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
 
    def output_shape(self, input_shape=None):
        return [self.feature_dimension]
 
    def _to_bchw(self, x):
        """
        Accept either:
          [B, C, H, W]
          [B, H, W, C]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(x.shape)}")
 
        # channel-last
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
 
    def _preprocess(self, x):
        x = self._to_bchw(x)
 
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if x.max() > 1.0:
                x = x / 255.0
 
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
 
        x = (x - self.image_mean) / self.image_std
        return x
 
    def forward(self, inputs):
        x = self._preprocess(inputs)
 
        if any(p.requires_grad for p in self.backbone.parameters()):
            outputs = self.backbone(pixel_values=x)
        else:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
 
        tokens = outputs.last_hidden_state  # [B, 1 + num_patches, hidden_dim]
 
        if self.use_cls_token:
            feat = tokens[:, 0, :]              # CLS token
        else:
            feat = tokens[:, 1:, :].mean(dim=1) # mean patch feature
 
        z = self.projector(feat)
        z = F.normalize(z, dim=-1)
        return z
 
 
import torch
import torch.nn as nn

from robomimic.models.base_nets import EncoderCore


class ForceEncoderCore(EncoderCore):
    """
    Simple 3-layer MLP force encoder.

    Intended to match the paper's description:
    force input -> 3-layer MLP -> 64-D feature vector

    Notes:
    - Assumes input is a 1D wrench / force vector, e.g. shape [6]
    - No final L2 normalization, since this is for policy conditioning
      rather than a contrastive embedding by default
    - Hidden sizes are chosen reasonably because the paper text
      specifies the 3-layer MLP and 64-D output, but not hidden widths
    """

    def __init__(
        self,
        input_shape,
        feature_dimension=64,
        hidden_dims=(64, 64),
    ):
        super().__init__(input_shape=input_shape)

        assert len(input_shape) == 1, f"Expected force shape like [6], got {input_shape}"
        assert len(hidden_dims) == 2, "For a 3-layer MLP, hidden_dims should contain exactly 2 values."

        in_dim = int(input_shape[0])
        h1, h2 = hidden_dims

        self.feature_dimension = feature_dimension

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, feature_dimension),
        )

    def output_shape(self, input_shape=None):
        return [self.feature_dimension]

    def forward(self, inputs):
        return self.net(inputs.float())
 
# we could use DINOv2 since it does better than CLIP, and a force encoder via an MLP
 
 
# i also made an algo but tbd because im not sure how this would work, but the flow is as follows:
 
# camera image -> DINOv2 -> proj -> z_v (128)
# wrench (or maybe torque?)-> MLP -> z_f (128)
# contact -> mask (binary)
# CaMI = InfoNCE(z_v[contact], z_f[contact])
# policy input = concat(z_v, z_f * contact, proprio (we may need this butidk), contact)