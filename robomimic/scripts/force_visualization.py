import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Candidate paths
candidates = [
    "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_84_square_force_v15.hdf5",
    "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_84_square_v15.hdf5",
]

dataset_path = None
for p in candidates:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError("Could not find the expected HDF5 dataset.")

output_dir = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square_force_bias_difference"
os.makedirs(output_dir, exist_ok=True)

num_points = 100

def interp_to_normalized_time(x, num_points=100):
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]

    if T < 2:
        if x.ndim == 1:
            return np.repeat(x, num_points)
        else:
            return np.repeat(x[None, ...], num_points, axis=0)

    old_t = np.linspace(0.0, 1.0, T)
    new_t = np.linspace(0.0, 1.0, num_points)

    if x.ndim == 1:
        return np.interp(new_t, old_t, x).astype(np.float32)

    out = np.zeros((num_points, x.shape[1]), dtype=np.float32)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(new_t, old_t, x[:, d])
    return out

def plot_mean_std(x, mean, std, labels, title, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    if mean.ndim == 1:
        plt.plot(x, mean, label=labels[0])
        plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    else:
        for i, lab in enumerate(labels):
            plt.plot(x, mean[:, i], label=lab)
            plt.fill_between(x, mean[:, i] - std[:, i], mean[:, i] + std[:, i], alpha=0.2)
    plt.xlabel("Normalized time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

all_raw_profiles = []
all_obs_profiles = []
all_diff_profiles = []

all_F_diff_profiles = []
all_T_diff_profiles = []
all_diff_mag_profiles = []
all_abs_diff_profiles = []

with h5py.File(dataset_path, "r") as f:
    data_grp = f["data"]
    demo_keys = sorted(data_grp.keys())

    for demo_key in demo_keys:
        obs = data_grp[demo_key]["obs"]

        if "force_rawbias" not in obs or "force_obsbias" not in obs:
            print(f"Skipping {demo_key} because one of force_rawbias / force_obsbias is missing.")
            continue

        rawbias = np.asarray(obs["force_rawbias"], dtype=np.float32)   # (T, 6)
        obsbias = np.asarray(obs["force_obsbias"], dtype=np.float32)   # (T, 6)

        if rawbias.shape[0] < 2 or obsbias.shape[0] < 2:
            print(f"Skipping {demo_key} because it is too short.")
            continue

        T = min(rawbias.shape[0], obsbias.shape[0])
        rawbias = rawbias[:T]
        obsbias = obsbias[:T]

        diff = obsbias - rawbias
        abs_diff = np.abs(diff)

        F_diff = diff[:, :3]
        Tau_diff = diff[:, 3:6]
        diff_mag = np.linalg.norm(diff, axis=1)

        all_raw_profiles.append(interp_to_normalized_time(rawbias, num_points))
        all_obs_profiles.append(interp_to_normalized_time(obsbias, num_points))
        all_diff_profiles.append(interp_to_normalized_time(diff, num_points))
        all_abs_diff_profiles.append(interp_to_normalized_time(abs_diff, num_points))

        all_F_diff_profiles.append(interp_to_normalized_time(F_diff, num_points))
        all_T_diff_profiles.append(interp_to_normalized_time(Tau_diff, num_points))
        all_diff_mag_profiles.append(interp_to_normalized_time(diff_mag, num_points))

if len(all_diff_profiles) == 0:
    raise RuntimeError("No valid demos with both force_rawbias and force_obsbias were found.")

all_raw_profiles = np.stack(all_raw_profiles, axis=0)
all_obs_profiles = np.stack(all_obs_profiles, axis=0)
all_diff_profiles = np.stack(all_diff_profiles, axis=0)
all_abs_diff_profiles = np.stack(all_abs_diff_profiles, axis=0)

all_F_diff_profiles = np.stack(all_F_diff_profiles, axis=0)
all_T_diff_profiles = np.stack(all_T_diff_profiles, axis=0)
all_diff_mag_profiles = np.stack(all_diff_mag_profiles, axis=0)

raw_mean = all_raw_profiles.mean(axis=0)
raw_std = all_raw_profiles.std(axis=0)

obs_mean = all_obs_profiles.mean(axis=0)
obs_std = all_obs_profiles.std(axis=0)

diff_mean = all_diff_profiles.mean(axis=0)
diff_std = all_diff_profiles.std(axis=0)

abs_diff_mean = all_abs_diff_profiles.mean(axis=0)
abs_diff_std = all_abs_diff_profiles.std(axis=0)

F_diff_mean = all_F_diff_profiles.mean(axis=0)
F_diff_std = all_F_diff_profiles.std(axis=0)

Tau_diff_mean = all_T_diff_profiles.mean(axis=0)
Tau_diff_std = all_T_diff_profiles.std(axis=0)

diff_mag_mean = all_diff_mag_profiles.mean(axis=0)
diff_mag_std = all_diff_mag_profiles.std(axis=0)

t_norm = np.linspace(0.0, 1.0, num_points)

# Compare rawbias and obsbias force components directly
plot_mean_std(
    t_norm,
    np.stack([raw_mean[:, 0], obs_mean[:, 0]], axis=1),
    np.stack([raw_std[:, 0], obs_std[:, 0]], axis=1),
    labels=["Fx rawbias", "Fx obsbias"],
    title="Fx comparison: rawbias vs obsbias",
    ylabel="Force",
    save_path=os.path.join(output_dir, "Fx_rawbias_vs_obsbias.png")
)

plot_mean_std(
    t_norm,
    np.stack([raw_mean[:, 1], obs_mean[:, 1]], axis=1),
    np.stack([raw_std[:, 1], obs_std[:, 1]], axis=1),
    labels=["Fy rawbias", "Fy obsbias"],
    title="Fy comparison: rawbias vs obsbias",
    ylabel="Force",
    save_path=os.path.join(output_dir, "Fy_rawbias_vs_obsbias.png")
)

plot_mean_std(
    t_norm,
    np.stack([raw_mean[:, 2], obs_mean[:, 2]], axis=1),
    np.stack([raw_std[:, 2], obs_std[:, 2]], axis=1),
    labels=["Fz rawbias", "Fz obsbias"],
    title="Fz comparison: rawbias vs obsbias",
    ylabel="Force",
    save_path=os.path.join(output_dir, "Fz_rawbias_vs_obsbias.png")
)

# Difference per force component
plot_mean_std(
    t_norm, F_diff_mean, F_diff_std,
    labels=["Fx diff", "Fy diff", "Fz diff"],
    title="Force component difference: obsbias - rawbias",
    ylabel="Difference",
    save_path=os.path.join(output_dir, "force_component_difference_mean_std.png")
)

# Difference per torque component
plot_mean_std(
    t_norm, Tau_diff_mean, Tau_diff_std,
    labels=["Tx diff", "Ty diff", "Tz diff"],
    title="Torque component difference: obsbias - rawbias",
    ylabel="Difference",
    save_path=os.path.join(output_dir, "torque_component_difference_mean_std.png")
)

# Magnitude of 6D difference
plot_mean_std(
    t_norm, diff_mag_mean, diff_mag_std,
    labels=["||obsbias - rawbias||"],
    title="Magnitude of difference between obsbias and rawbias",
    ylabel="6D difference magnitude",
    save_path=os.path.join(output_dir, "difference_magnitude_mean_std.png")
)

# Absolute difference per component
plot_mean_std(
    t_norm, abs_diff_mean, abs_diff_std,
    labels=["|Fx diff|", "|Fy diff|", "|Fz diff|", "|Tx diff|", "|Ty diff|", "|Tz diff|"],
    title="Absolute component-wise difference: obsbias vs rawbias",
    ylabel="Absolute difference",
    save_path=os.path.join(output_dir, "absolute_component_difference_mean_std.png")
)

print(f"Saved plots to: {output_dir}")
print(f"Used {all_diff_profiles.shape[0]} demos.")