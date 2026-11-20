"""
Derive ManiFlow + CaMI configs from your working ManiFlow JSON, so dataset, normalization,
observation and training settings stay identical to the ManiFlow baseline.

    python make_maniflow_cami_jsons.py path/to/maniflow.json [out_dir]

Writes:
    maniflow_cami.json          full run (CaMI on, class defaults)
    maniflow_cami_control.json  same code path, CaMI off (plain ManiFlow control)
    maniflow_cami_smoke.json    3 epochs, no rollouts / saving / wandb
"""
import copy
import json
import os
import sys

src = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(src))
base = json.load(open(src))

base["algo_name"] = "maniflow_cami"
keys = list(base["train"].get("dataset_keys", ["actions", "rewards", "dones"]))
for k in ("contact_label", "force"):
    if k not in keys:
        keys.append(k)
base["train"]["dataset_keys"] = keys
base.setdefault("algo", {}).setdefault("cami", {})["enabled"] = True


def write(cfg, name, exp_name):
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["name"] = exp_name
    path = os.path.join(out_dir, name)
    json.dump(cfg, open(path, "w"), indent=4)
    print("wrote", path)


write(base, "maniflow_cami.json", "maniflow_cami")

ctrl = copy.deepcopy(base)
ctrl["algo"]["cami"]["enabled"] = False
write(ctrl, "maniflow_cami_control.json", "maniflow_cami_control")

smoke = copy.deepcopy(base)
smoke["train"]["num_epochs"] = 3
smoke["experiment"]["rollout"]["enabled"] = False
smoke["experiment"]["save"]["enabled"] = False
smoke["experiment"]["logging"]["log_wandb"] = False
smoke["experiment"]["validate"] = False
write(smoke, "maniflow_cami_smoke.json", "maniflow_cami_smoke")