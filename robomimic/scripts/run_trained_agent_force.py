"""
Evaluate a trained policy and analyze injected force distributions / trajectories.
"""

import argparse
import json
import os
from copy import deepcopy

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.CAMI.force_encoder import ForceEncoderCore
from robomimic.CAMI.force_modality import ForceModality
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper


# -------------------------
# Global force logs
# -------------------------
FORCE_LOG_RAW = []
FORCE_LOG_NORM = []
FORCE_LOG_FIRST_RAW = []
FORCE_LOG_FIRST_NORM = []

# Per-episode sequences
FORCE_EPISODE_RAW = []
FORCE_EPISODE_NORM = []


# -------------------------
# Optional debug hook
# -------------------------
def attach_force_encoder_shape_hook(policy):
    handles = []

    module_root = None
    module_root_name = None

    if isinstance(policy, torch.nn.Module):
        module_root = policy
        module_root_name = "policy"
    elif hasattr(policy, "policy"):
        algo = policy.policy
        if isinstance(algo, torch.nn.Module) and hasattr(algo, "named_modules"):
            module_root = algo
            module_root_name = "policy.policy"
        elif hasattr(algo, "nets") and "policy" in algo.nets:
            module_root = algo.nets["policy"]
            module_root_name = 'policy.policy.nets["policy"]'

    if module_root is None:
        print("[DEBUG] Could not find module root for hook attachment")
        return handles

    print(f"[DEBUG] Hook search root: {module_root_name} ({type(module_root)})")

    def make_hook(name):
        def hook(module, inputs, output):
            print(f"\n[FORCE ENCODER HOOK] module: {name}")
            if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                print("  input shape :", tuple(inputs[0].shape))
            else:
                print("  input shape : <non-tensor or unavailable>")

            if isinstance(output, torch.Tensor):
                print("  output shape:", tuple(output.shape))
            elif isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        print(f"  output[{i}] shape:", tuple(out.shape))
                    else:
                        print(f"  output[{i}] type :", type(out))
            else:
                print("  output type :", type(output))
        return hook

    found = False
    for name, module in module_root.named_modules():
        if isinstance(module, ForceEncoderCore):
            found = True
            print(f"[DEBUG] Found ForceEncoderCore at: {name}")
            handles.append(module.register_forward_hook(make_hook(name)))

    if not found:
        print("[DEBUG] No ForceEncoderCore found under", module_root_name)

    return handles

def compute_eval_force_from_current_state(env, bias_alpha=0.1):
    """
    Recompute force exactly like the eval overwrite path, but return it directly.
    """
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    F_raw, T_raw = base_env._read_raw_ft_sensor()

    if getattr(base_env, "_bias_F_sensor", None) is None:
        base_env._bias_F_sensor = F_raw.copy()
        base_env._bias_T_sensor = T_raw.copy()

    ncon = int(base_env.sim.data.ncon)
    if ncon == 0:
        a = float(bias_alpha)
        base_env._bias_F_sensor = (1.0 - a) * base_env._bias_F_sensor + a * F_raw
        base_env._bias_T_sensor = (1.0 - a) * base_env._bias_T_sensor + a * T_raw

    F = F_raw - base_env._bias_F_sensor
    T = T_raw - base_env._bias_T_sensor

    force_vec = np.concatenate([F, T], axis=0).astype(np.float32)
    return (
        force_vec,
        ncon,
        F_raw.copy(),
        T_raw.copy(),
        base_env._bias_F_sensor.copy(),
        base_env._bias_T_sensor.copy(),
    )


def compare_dataset_vs_env_force(
    dataset_path,
    env,
    demo_key,
    out_dir,
    normalize_fn,
    bias_alpha=0.1,
    max_steps=None,
):
    """
    Compare stored dataset obs/force against env-recomputed force at exactly the same restored states.
    """
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        states = demo["states"][()]
        dataset_force = demo["obs/force"][()]

        if max_steps is not None:
            states = states[:max_steps]
            dataset_force = dataset_force[:max_steps]

    # unwrap robomimic / wrapper stack to raw robosuite env
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # reset force bias before replaying this demo
    reset_force_bias_state(env)

    eval_force = []
    ncon_list = []
    raw_F_list = []
    raw_T_list = []
    bias_F_list = []
    bias_T_list = []

    # replay sequentially, preserving running bias state
    for t in range(states.shape[0]):
        base_env.sim.set_state_from_flattened(states[t])
        base_env.sim.forward()

        f_eval, ncon, rawF, rawT, biasF, biasT = compute_eval_force_from_current_state(
            env, bias_alpha=bias_alpha
        )

        eval_force.append(f_eval)
        ncon_list.append(ncon)
        raw_F_list.append(rawF)
        raw_T_list.append(rawT)
        bias_F_list.append(biasF)
        bias_T_list.append(biasT)

    eval_force = np.asarray(eval_force, dtype=np.float32)
    ncon_list = np.asarray(ncon_list, dtype=np.int32)
    raw_F_list = np.asarray(raw_F_list, dtype=np.float32)
    raw_T_list = np.asarray(raw_T_list, dtype=np.float32)
    bias_F_list = np.asarray(bias_F_list, dtype=np.float32)
    bias_T_list = np.asarray(bias_T_list, dtype=np.float32)

    dataset_norm = normalize_fn(dataset_force)
    eval_norm = normalize_fn(eval_force)

    abs_err = np.abs(dataset_force - eval_force)
    norm_abs_err = np.abs(dataset_norm - eval_norm)

    dim_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    np.save(os.path.join(out_dir, f"{demo_key}_dataset_force.npy"), dataset_force)
    np.save(os.path.join(out_dir, f"{demo_key}_eval_force.npy"), eval_force)
    np.save(os.path.join(out_dir, f"{demo_key}_dataset_force_norm.npy"), dataset_norm)
    np.save(os.path.join(out_dir, f"{demo_key}_eval_force_norm.npy"), eval_norm)
    np.save(os.path.join(out_dir, f"{demo_key}_ncon.npy"), ncon_list)

    with open(os.path.join(out_dir, f"{demo_key}_force_compare_stats.txt"), "w") as f:
        f.write(f"demo_key: {demo_key}\n")
        f.write(f"num_steps: {states.shape[0]}\n\n")

        for i, name in enumerate(dim_names):
            rmse = np.sqrt(np.mean((dataset_force[:, i] - eval_force[:, i]) ** 2))
            f.write(
                f"{name} raw: "
                f"mae={abs_err[:, i].mean():.6f}, "
                f"rmse={rmse:.6f}, "
                f"dataset_mean={dataset_force[:, i].mean():.6f}, "
                f"eval_mean={eval_force[:, i].mean():.6f}\n"
            )

        f.write("\n")

        for i, name in enumerate(dim_names):
            rmse = np.sqrt(np.mean((dataset_norm[:, i] - eval_norm[:, i]) ** 2))
            f.write(
                f"{name} norm: "
                f"mae={norm_abs_err[:, i].mean():.6f}, "
                f"rmse={rmse:.6f}, "
                f"dataset_mean={dataset_norm[:, i].mean():.6f}, "
                f"eval_mean={eval_norm[:, i].mean():.6f}\n"
            )

    x = np.arange(states.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(x, dataset_force[:, i], label="dataset", linewidth=1.5)
        ax.plot(x, eval_force[:, i], label="eval", linewidth=1.0, alpha=0.8)
        ax.set_title(dim_names[i])
        ax.set_xlabel("timestep")
        ax.set_ylabel("raw")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{demo_key}_raw_overlay.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(x, dataset_norm[:, i], label="dataset", linewidth=1.5)
        ax.plot(x, eval_norm[:, i], label="eval", linewidth=1.0, alpha=0.8)
        ax.set_title(dim_names[i])
        ax.set_xlabel("timestep")
        ax.set_ylabel("normalized")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{demo_key}_norm_overlay.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 3))
    plt.plot(x, ncon_list, linewidth=1.2)
    plt.title("sim.data.ncon during matched-state replay")
    plt.xlabel("timestep")
    plt.ylabel("ncon")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{demo_key}_ncon.png"), dpi=200)
    plt.close(fig)

    print(f"[FORCE COMPARE] Saved comparison for {demo_key} to {out_dir}")
    
# -------------------------
# Force utilities
# -------------------------
def compute_force_stats(hdf5_path):
    all_force = []
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        for demo in demos:
            force = f[f"data/{demo}/obs/force"][()]
            all_force.append(force)
    all_force = np.concatenate(all_force, axis=0)
    mean = all_force.mean(axis=0)
    std = np.maximum(all_force.std(axis=0), 1e-6)
    return mean, std


def reset_force_bias_state(env):
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    base_env._bias_F_sensor = None
    base_env._bias_T_sensor = None


def normalize_force_array(force_arr):
    """
    force_arr: [N, 6] raw force array
    returns : [N, 6] normalized + clipped like ForceModality
    """
    mean = np.asarray(ForceModality.FORCE_MEAN, dtype=np.float32)
    std = np.asarray(ForceModality.FORCE_STD, dtype=np.float32)
    eps = float(getattr(ForceModality, "EPS", 1e-6))
    clip = float(getattr(ForceModality, "CLIP", 5.0))

    norm = (force_arr.astype(np.float32) - mean) / (std + eps)
    norm = np.clip(norm, -clip, clip)
    return norm


def interp_sequence_to_normalized_time(seq, num_points=100):
    """
    seq: [T, D]
    returns: [num_points, D]
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]

    if T == 1:
        return np.repeat(seq, num_points, axis=0)

    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, num_points)

    out = np.zeros((num_points, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(x_new, x_old, seq[:, d])
    return out


# -------------------------
# Plotting
# -------------------------
def plot_force_distributions(raw_force, norm_force, out_dir, prefix="rollout"):
    os.makedirs(out_dir, exist_ok=True)
    dim_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    # Raw histograms
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.hist(raw_force[:, i], bins=100, alpha=0.8)
        ax.set_title(f"Raw {dim_names[i]}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_raw_force_hist.png"), dpi=200)
    plt.close(fig)

    # Normalized histograms
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.hist(norm_force[:, i], bins=100, alpha=0.8)
        ax.set_title(f"Normalized {dim_names[i]}")
        ax.set_xlabel("z-scored value")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_normalized_force_hist.png"), dpi=200)
    plt.close(fig)

    # Norm histograms
    raw_force_norm = np.linalg.norm(raw_force[:, :3], axis=1)
    raw_torque_norm = np.linalg.norm(raw_force[:, 3:], axis=1)
    norm_force_norm = np.linalg.norm(norm_force[:, :3], axis=1)
    norm_torque_norm = np.linalg.norm(norm_force[:, 3:], axis=1)

    fig = plt.figure(figsize=(10, 6))
    plt.hist(raw_force_norm, bins=100, alpha=0.7, label="||F|| raw")
    plt.hist(raw_torque_norm, bins=100, alpha=0.7, label="||T|| raw")
    plt.legend()
    plt.xlabel("norm")
    plt.ylabel("count")
    plt.title("Raw force / torque norm distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_raw_norm_hist.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.hist(norm_force_norm, bins=100, alpha=0.7, label="||F|| normalized")
    plt.hist(norm_torque_norm, bins=100, alpha=0.7, label="||T|| normalized")
    plt.legend()
    plt.xlabel("norm")
    plt.ylabel("count")
    plt.title("Normalized force / torque norm distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_normalized_norm_hist.png"), dpi=200)
    plt.close(fig)

    # Stats text file
    stats_path = os.path.join(out_dir, f"{prefix}_force_stats.txt")
    with open(stats_path, "w") as f:
        f.write("Raw force stats\n")
        for i, name in enumerate(dim_names):
            vals = raw_force[:, i]
            f.write(
                f"{name}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
                f"min={vals.min():.6f}, max={vals.max():.6f}, "
                f"p1={np.percentile(vals,1):.6f}, p50={np.percentile(vals,50):.6f}, p99={np.percentile(vals,99):.6f}\n"
            )

        f.write("\nNormalized force stats\n")
        for i, name in enumerate(dim_names):
            vals = norm_force[:, i]
            clipped_frac = np.mean(np.abs(vals) >= float(getattr(ForceModality, "CLIP", 5.0)) - 1e-6)
            f.write(
                f"{name}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
                f"min={vals.min():.6f}, max={vals.max():.6f}, "
                f"p1={np.percentile(vals,1):.6f}, p50={np.percentile(vals,50):.6f}, p99={np.percentile(vals,99):.6f}, "
                f"clipped_frac={clipped_frac:.6f}\n"
            )


def plot_first_step_histograms(first_raw, first_norm, out_dir, prefix="rollout_firststep"):
    os.makedirs(out_dir, exist_ok=True)
    dim_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.hist(first_norm[:, i], bins=60, alpha=0.8)
        ax.set_title(f"First-step normalized {dim_names[i]}")
        ax.set_xlabel("z-scored value")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_normalized_hist.png"), dpi=200)
    plt.close(fig)


def plot_force_time_profile(force_episode_list, out_dir, prefix="eval_rollouts", num_points=100):
    """
    force_episode_list: list of [T_i, 6] normalized force sequences
    Produces mean ± std across normalized time.
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(force_episode_list) == 0:
        print("No per-episode force sequences to plot.")
        return

    interp_list = [interp_sequence_to_normalized_time(seq, num_points=num_points)
                   for seq in force_episode_list]
    arr = np.stack(interp_list, axis=0)  # [N, P, 6]

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    x = np.linspace(0.0, 1.0, num_points)

    # Force components
    fig = plt.figure(figsize=(12, 7))
    labels = ["Fx_norm", "Fy_norm", "Fz_norm"]
    for i, label in enumerate(labels):
        plt.plot(x, mean[:, i], label=label, linewidth=2)
        plt.fill_between(x, mean[:, i] - std[:, i], mean[:, i] + std[:, i], alpha=0.2)

    plt.title("Normalized force components across all rollouts (mean ± std)")
    plt.xlabel("Normalized time")
    plt.ylabel("Z-scored force")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_force_time_profile.png"), dpi=200)
    plt.close(fig)

    # Torque components
    fig = plt.figure(figsize=(12, 7))
    labels = ["Tx_norm", "Ty_norm", "Tz_norm"]
    for i, label in enumerate(labels, start=3):
        plt.plot(x, mean[:, i], label=label, linewidth=2)
        plt.fill_between(x, mean[:, i] - std[:, i], mean[:, i] + std[:, i], alpha=0.2)

    plt.title("Normalized torque components across all rollouts (mean ± std)")
    plt.xlabel("Normalized time")
    plt.ylabel("Z-scored torque")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_torque_time_profile.png"), dpi=200)
    plt.close(fig)

    np.save(os.path.join(out_dir, f"{prefix}_time_profile_mean.npy"), mean)
    np.save(os.path.join(out_dir, f"{prefix}_time_profile_std.npy"), std)


# -------------------------
# Rollout
# -------------------------
def inject_force_into_obs(env, obs, debug_prefix="", bias_alpha=0.1, verbose=False):
    """
    Always overwrite obs['force'] so eval matches training dataset semantics:

        force = force_rawbias
              = raw Mujoco FT sensor
              - EMA bias updated only when ncon == 0
              - unscaled
              - shape [Fx, Fy, Fz, Tx, Ty, Tz]

    This intentionally ignores any env-provided obs['force'].
    """
    try:
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        F_raw, T_raw = base_env._read_raw_ft_sensor()

        if getattr(base_env, "_bias_F_sensor", None) is None:
            base_env._bias_F_sensor = F_raw.copy()
            base_env._bias_T_sensor = T_raw.copy()

        ncon = int(base_env.sim.data.ncon)
        if ncon == 0:
            a = float(bias_alpha)
            base_env._bias_F_sensor = (1.0 - a) * base_env._bias_F_sensor + a * F_raw
            base_env._bias_T_sensor = (1.0 - a) * base_env._bias_T_sensor + a * T_raw

        F = F_raw - base_env._bias_F_sensor
        T = T_raw - base_env._bias_T_sensor

        force_vec = np.concatenate([F, T], axis=0).astype(np.float32)
        norm_vec = normalize_force_array(force_vec[None])[0]

        if verbose:
            print(f"[FORCE INJECT] {debug_prefix}")
            print("  raw force vec :", force_vec)
            print("  norm force vec:", norm_vec)
            print("  norm abs max  :", np.max(np.abs(norm_vec)))
            print("  clipped dims  :", np.abs(norm_vec) >= float(getattr(ForceModality, "CLIP", 5.0)) - 1e-6)

        # ALWAYS overwrite env-provided force
        obs["force"] = force_vec

        FORCE_LOG_RAW.append(force_vec.copy())
        FORCE_LOG_NORM.append(norm_vec.copy())

        if verbose:
            print(f"[FORCE INJECT] {debug_prefix}")
            print("  ncon:", ncon)
            print("  raw force :", F_raw)
            print("  raw torque:", T_raw)
            print("  bias force :", base_env._bias_F_sensor)
            print("  bias torque:", base_env._bias_T_sensor)
            print("  corr force :", force_vec)

    except Exception as e:
        print(f"[FORCE INJECT DEBUG] {debug_prefix} failed to inject force: {e}")

    return obs

def attach_force_encoder_value_hook(policy, max_prints=10):
    handles = []
    counter = {"n": 0}

    module_root = None
    if isinstance(policy, torch.nn.Module):
        module_root = policy
    elif hasattr(policy, "policy"):
        algo = policy.policy
        if isinstance(algo, torch.nn.Module) and hasattr(algo, "named_modules"):
            module_root = algo
        elif hasattr(algo, "nets") and "policy" in algo.nets:
            module_root = algo.nets["policy"]

    if module_root is None:
        print("[DEBUG] Could not find module root for force value hook")
        return handles

    def pre_hook(name):
        def hook(module, inputs):
            if counter["n"] >= max_prints:
                return
            if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                return

            x = inputs[0].detach().cpu()
            print(f"\n[FORCE VALUE HOOK] module={name}")
            print("  shape:", tuple(x.shape))
            print("  min  :", float(x.min()))
            print("  max  :", float(x.max()))
            print("  mean :", float(x.mean()))
            print("  std  :", float(x.std()))
            flat = x.reshape(-1)
            print("  first values:", flat[: min(12, flat.numel())].numpy())
            counter["n"] += 1
        return hook

    for name, module in module_root.named_modules():
        if isinstance(module, ForceEncoderCore):
            handles.append(module.register_forward_pre_hook(pre_hook(name)))

    return handles

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5,
            return_obs=False, camera_names=None, force_verbose=False):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    # deterministic initial state
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    # IMPORTANT:
    # after final reset_to, clear bias state and then build the actual first obs
    reset_force_bias_state(env)
    obs = inject_force_into_obs(env, obs, debug_prefix="after reset_to", verbose=force_verbose)

    # start policy only after first observation is finalized
    policy.start_episode()

    total_reward = 0.0
    video_count = 0
    success = False

    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    ep_force_raw = []
    ep_force_norm = []

    if "force" in obs:
        ep_force_raw.append(obs["force"].copy())
        ep_force_norm.append(normalize_force_array(obs["force"][None])[0].copy())
        FORCE_LOG_FIRST_RAW.append(obs["force"].copy())
        FORCE_LOG_FIRST_NORM.append(normalize_force_array(obs["force"][None])[0].copy())

    try:
        for step_i in range(horizon):
            if step_i < 3:
                print("[DEBUG] obs keys:", obs.keys())
                if "force" in obs:
                    print("[DEBUG] obs['force'] shape:", np.array(obs["force"]).shape)
                    print("[DEBUG] obs['force']:", obs["force"])
            act = policy(ob=obs)

            next_obs, r, done, _ = env.step(act)

            # overwrite env-provided force every step
            next_obs = inject_force_into_obs(
                env, next_obs, debug_prefix=f"after step {step_i}", verbose=force_verbose
            )

            total_reward += r
            success = env.is_success()["task"]

            if "force" in next_obs:
                ep_force_raw.append(next_obs["force"].copy())
                ep_force_norm.append(normalize_force_array(next_obs["force"][None])[0].copy())

            if render:
                env.render(mode="human", camera_name=camera_names[0])

            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])

            if return_obs:
                traj["obs"].append(deepcopy(obs))
                traj["next_obs"].append(deepcopy(next_obs))

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print(f"WARNING: got rollout exception {e}")

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    ep_force_raw = np.asarray(ep_force_raw, dtype=np.float32)
    ep_force_norm = np.asarray(ep_force_norm, dtype=np.float32)

    return stats, traj, ep_force_raw, ep_force_norm

# -------------------------
# Main
# -------------------------
def run_trained_agent(args):
    global FORCE_LOG_RAW, FORCE_LOG_NORM
    global FORCE_LOG_FIRST_RAW, FORCE_LOG_FIRST_NORM
    global FORCE_EPISODE_RAW, FORCE_EPISODE_NORM

    FORCE_LOG_RAW = []
    FORCE_LOG_NORM = []
    FORCE_LOG_FIRST_RAW = []
    FORCE_LOG_FIRST_NORM = []
    FORCE_EPISODE_RAW = []
    FORCE_EPISODE_NORM = []

    write_video = args.video_path is not None
    assert not (args.render and write_video)
    if args.render:
        assert len(args.camera_names) == 1

    ckpt_path = args.agent
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    ObsUtils.OBS_ENCODER_CORES["ForceEncoderCore"] = ForceEncoderCore

    # Load checkpoint config first
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    print("[DEBUG] observation modalities:", config.observation.modalities)
    print("[DEBUG] all obs keys:", config.all_obs_keys if hasattr(config, "all_obs_keys") else "N/A")
    print("[DEBUG] encoder config:", config.observation.encoder)

    # Use dataset path saved in checkpoint config
    # train_data_cfg = config.train.data
    # if isinstance(train_data_cfg, str):
    #     train_dataset_path = train_data_cfg
    # else:
    #     train_dataset_path = train_data_cfg[0]["path"]
    train_dataset_path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_84_square_force_v15.hdf5"


    force_mean, force_std = compute_force_stats(train_dataset_path)
    ForceModality.set_normalization_stats(force_mean, force_std, clip=5.0)

    print("[DEBUG] train_dataset_path:", train_dataset_path)
    print("[DEBUG] Force mean:", force_mean)
    print("[DEBUG] Force std :", force_std)

    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    if args.compare_force_demo is not None:
        out_dir = os.path.join(os.path.dirname(args.agent), "force_compare_debug")
        compare_dataset_vs_env_force(
            dataset_path=train_dataset_path,
            env=env,
            demo_key=args.compare_force_demo,
            out_dir=out_dir,
            normalize_fn=normalize_force_array,
            bias_alpha=0.1,
            max_steps=args.compare_force_max_steps,
        )
        return

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=True
    )

    handles = []
    if args.debug_force_hook:
        handles = attach_force_encoder_shape_hook(policy)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    write_dataset = args.dataset_path is not None
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj, ep_force_raw, ep_force_norm = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            force_verbose=args.force_verbose,
        )
        rollout_stats.append(stats)

        if ep_force_raw.shape[0] > 0:
            FORCE_EPISODE_RAW.append(ep_force_raw)
            FORCE_EPISODE_NORM.append(ep_force_norm)

        if write_dataset:
            ep_data_grp = data_grp.create_group(f"demo_{i}")
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    out_dir = os.path.join(os.path.dirname(args.agent), "force_debug_plots")
    os.makedirs(out_dir, exist_ok=True)

    if len(FORCE_LOG_RAW) > 0:
        raw_force = np.asarray(FORCE_LOG_RAW, dtype=np.float32)
        norm_force = np.asarray(FORCE_LOG_NORM, dtype=np.float32)

        np.save(os.path.join(out_dir, "eval_rollouts_raw_force.npy"), raw_force)
        np.save(os.path.join(out_dir, "eval_rollouts_normalized_force.npy"), norm_force)

        plot_force_distributions(raw_force, norm_force, out_dir, prefix="eval_rollouts")
        print(f"Saved pooled force plots to: {out_dir}")
        print("Logged force samples:", len(FORCE_LOG_RAW))
    else:
        print("No injected force samples were logged.")

    if len(FORCE_LOG_FIRST_RAW) > 0:
        first_raw = np.asarray(FORCE_LOG_FIRST_RAW, dtype=np.float32)
        first_norm = np.asarray(FORCE_LOG_FIRST_NORM, dtype=np.float32)
        np.save(os.path.join(out_dir, "eval_rollouts_firststep_raw_force.npy"), first_raw)
        np.save(os.path.join(out_dir, "eval_rollouts_firststep_normalized_force.npy"), first_norm)
        plot_first_step_histograms(first_raw, first_norm, out_dir, prefix="eval_rollouts_firststep")
        print("Saved first-step force plots.")

    if len(FORCE_EPISODE_NORM) > 0:
        # Save ragged lists as object arrays for later debugging
        np.save(os.path.join(out_dir, "eval_rollouts_episode_raw_force.npy"),
                np.array(FORCE_EPISODE_RAW, dtype=object), allow_pickle=True)
        np.save(os.path.join(out_dir, "eval_rollouts_episode_normalized_force.npy"),
                np.array(FORCE_EPISODE_NORM, dtype=object), allow_pickle=True)

        plot_force_time_profile(FORCE_EPISODE_NORM, out_dir, prefix="eval_rollouts", num_points=args.profile_points)
        print("Saved normalized-time mean±std force profiles.")
    else:
        print("No per-episode force sequences were available for time-profile plots.")

    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print(f"Wrote dataset trajectories to {args.dataset_path}")

    # Clean up hooks
    for h in handles:
        h.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, required=True, help="path to saved checkpoint pth file")
    parser.add_argument("--n_rollouts", type=int, default=27, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=None, help="override maximum horizon of rollout")
    parser.add_argument("--env", type=str, default=None, help="override env name from checkpoint")
    parser.add_argument("--render", action="store_true", help="on-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="write rollout video")
    parser.add_argument("--video_skip", type=int, default=5, help="write video frames every n steps")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["agentview"],
                        help="camera name(s) for rendering")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="if provided, write rollout dataset hdf5 here")
    parser.add_argument("--dataset_obs", action="store_true",
                        help="include observations in written rollout dataset")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--profile_points", type=int, default=100,
                        help="number of normalized-time points for mean±std profile plot")
    parser.add_argument("--debug_force_hook", action="store_true",
                        help="attach verbose force encoder hooks")
    parser.add_argument("--force_verbose", action="store_true",
                    help="print detailed force injection debug info")
    parser.add_argument("--compare_force_demo", type=str, default=None,
                        help="demo key to compare dataset obs/force vs env-recomputed force, e.g. demo_0")
    parser.add_argument("--compare_force_max_steps", type=int, default=200,
                        help="max number of timesteps to compare in matched-state force replay")
    args = parser.parse_args()
    run_trained_agent(args)