import torch, json
ckpt_path = "/home/aist/hisham/cami/cami_models/bc_rnn_cami_square_100_percent_success_0.88.pth"
ckpt = torch.load(ckpt_path,
                  map_location="cpu", weights_only=False)
cfg = json.loads(ckpt["config"])
json.dump(cfg, open("config.json", "w"), indent=4)