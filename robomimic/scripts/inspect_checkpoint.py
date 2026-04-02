import torch
import json
from pprint import pprint

ckpt_path = "/home/hisham246/Downloads/square_ph_image_epoch_540_succ_78.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

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