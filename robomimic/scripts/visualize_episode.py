import os
import h5py
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Candidate paths based on the earlier conversation
candidates = [
    "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_224_square_force_v15.hdf5",
    "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_224_square_v15.hdf5",
]

dataset_path = None
for p in candidates:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError("Could not find the expected Square dataset HDF5 in the known locations.")

output_dir = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square_episode_viz"
os.makedirs(output_dir, exist_ok=True)

demo_key = "demo_0"
contact_threshold = 10.0  # per user request

with h5py.File(dataset_path, "r") as f:
    data_grp = f["data"]
    if demo_key not in data_grp:
        demo_key = sorted(data_grp.keys())[0]

    demo = data_grp[demo_key]
    obs = demo["obs"]

    if "force" not in obs:
        raise KeyError(f"'force' not found in obs keys. Available: {list(obs.keys())}")

    force = np.asarray(obs["force"], dtype=np.float32)  # (T, 6)
    T = force.shape[0]
    t = np.arange(T)

    F = force[:, :3]
    Tau = force[:, 3:6]
    force_mag = np.linalg.norm(F, axis=1)
    torque_mag = np.linalg.norm(Tau, axis=1)
    contact = (force_mag > contact_threshold).astype(np.float32)

    # Pull images if present
    agentview = np.asarray(obs["agentview_image"]) if "agentview_image" in obs else None
    eye = np.asarray(obs["robot0_eye_in_hand_image"]) if "robot0_eye_in_hand_image" in obs else None

# ---------- Plot 1: force components ----------
plt.figure(figsize=(10, 4.8))
plt.plot(t, F[:, 0], label="Fx")
plt.plot(t, F[:, 1], label="Fy")
plt.plot(t, F[:, 2], label="Fz")
plt.xlabel("Timestep")
plt.ylabel("Force")
plt.title(f"{demo_key} force components")
plt.legend()
plt.tight_layout()
force_plot = os.path.join(output_dir, "force_components.png")
plt.savefig(force_plot, dpi=180)
plt.close()

# ---------- Plot 2: torque components ----------
plt.figure(figsize=(10, 4.8))
plt.plot(t, Tau[:, 0], label="Tx")
plt.plot(t, Tau[:, 1], label="Ty")
plt.plot(t, Tau[:, 2], label="Tz")
plt.xlabel("Timestep")
plt.ylabel("Torque")
plt.title(f"{demo_key} torque components")
plt.legend()
plt.tight_layout()
torque_plot = os.path.join(output_dir, "torque_components.png")
plt.savefig(torque_plot, dpi=180)
plt.close()

# ---------- Plot 3: magnitudes + binary contact ----------
plt.figure(figsize=(10, 4.8))
plt.plot(t, force_mag, label=f"|F|")
plt.plot(t, torque_mag, label=f"|Tau|")
plt.plot(t, contact, label=f"contact(|F|>{contact_threshold})")
plt.xlabel("Timestep")
plt.ylabel("Magnitude / Binary")
plt.title(f"{demo_key} force magnitude, torque magnitude, and binary contact")
plt.legend()
plt.tight_layout()
contact_plot = os.path.join(output_dir, "contact_and_magnitudes.png")
plt.savefig(contact_plot, dpi=180)
plt.close()

# ---------- Video rendering ----------
video_path = None
if agentview is not None or eye is not None:
    def to_uint8(img):
        arr = np.asarray(img)
        # Handle CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        # Convert floats to uint8 if needed
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        # Grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    frames = []
    n_frames = T
    for i in range(n_frames):
        tiles = []
        if agentview is not None:
            a = to_uint8(agentview[i])
            tiles.append(a)
        if eye is not None:
            e = to_uint8(eye[i])
            # Make heights match if needed
            if tiles and e.shape[0] != tiles[0].shape[0]:
                target_h = tiles[0].shape[0]
                scale = target_h / e.shape[0]
                target_w = max(1, int(round(e.shape[1] * scale)))
                # nearest resize via PIL if available, else simple slicing fallback
                try:
                    from PIL import Image
                    e = np.array(Image.fromarray(e).resize((target_w, target_h)))
                except Exception:
                    e = e[:target_h, :target_w]
            tiles.append(e)

        if len(tiles) == 1:
            frame = tiles[0]
        else:
            frame = np.concatenate(tiles, axis=1)

        # add a small bottom bar showing timestep/contact as a visible indicator
        bar_h = 24
        bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
        # simple progress marker
        progress_w = max(1, int(frame.shape[1] * (i + 1) / n_frames))
        bar[:, :progress_w, :] = 180
        # highlight contact state on left side
        if contact[i] > 0.5:
            bar[:, :max(10, frame.shape[1] // 10), 1] = 255
        else:
            bar[:, :max(10, frame.shape[1] // 10), 2] = 255
        frame = np.concatenate([frame, bar], axis=0)
        frames.append(frame)

    video_path = os.path.join(output_dir, "demo0_side_by_side.mp4")
    imageio.mimsave(video_path, frames, fps=20)

summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Demo: {demo_key}\n")
    f.write(f"Timesteps: {T}\n")
    f.write(f"Contact threshold on |F|: {contact_threshold}\n")
    f.write(f"Mean |F|: {force_mag.mean():.4f}\n")
    f.write(f"Max |F|: {force_mag.max():.4f}\n")
    f.write(f"Mean |Tau|: {torque_mag.mean():.4f}\n")
    f.write(f"Max |Tau|: {torque_mag.max():.4f}\n")
    f.write(f"Contact fraction: {contact.mean():.4f}\n")

print({
    "dataset_path": dataset_path,
    "demo_key": demo_key,
    "force_plot": force_plot,
    "torque_plot": torque_plot,
    "contact_plot": contact_plot,
    "video_path": video_path,
    "summary_path": summary_path,
})
