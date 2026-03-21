import h5py

path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_224_square_force_v15.hdf5"

bad = []

with h5py.File(path, "r") as f:
    for ep in f["data"].keys():
        grp = f[f"data/{ep}"]
        num_samples = int(grp.attrs["num_samples"])

        obs_keys = list(grp["obs"].keys())
        has_force = "force" in grp["obs"]

        if not has_force:
            bad.append((ep, "missing obs/force", num_samples, None))
            continue

        force_shape = grp["obs"]["force"].shape

        if len(force_shape) != 2:
            bad.append((ep, "obs/force wrong rank", num_samples, force_shape))
            continue

        if force_shape[0] != num_samples:
            bad.append((ep, "obs/force length mismatch", num_samples, force_shape))

print("num bad demos:", len(bad))
for x in bad[:50]:
    print(x)