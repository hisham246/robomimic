import torch
import json
from pprint import pprint

ckpt_path = "/home/aist/hisham/cami/cami_models/bc_rnn_cami_tool_hang_100_percent_success_0.64.pth"
ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

print("\nTop-level keys:")
print(list(ckpt.keys()))

if "config" in ckpt:
    print("\nFound config:\n")
    cfg = ckpt["config"]
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    pprint(cfg)
else:
    print("\nNo 'config' key found.")