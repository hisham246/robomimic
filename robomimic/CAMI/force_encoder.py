import torch.nn as nn
from robomimic.models.obs_core import EncoderCore


class ForceEncoderCore(EncoderCore):
    def __init__(self, input_shape, feature_dimension=64):
        super().__init__(input_shape=input_shape)

        assert len(input_shape) == 1, f"Expected force shape like [6], got {input_shape}"
        in_dim = int(input_shape[0])
        self.feature_dimension = feature_dimension

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dimension),
        )

    def output_shape(self, input_shape=None):
        return [self.feature_dimension]

    def forward(self, inputs):
        return self.net(inputs.float())