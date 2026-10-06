import h5py, numpy as np
p = "/home/aist/hisham/cami/cami_datasets/square_image_84_with_force_25_percent.hdf5"
with h5py.File(p, "r") as f:
    d = f["data"]; demos = sorted(d.keys())
    C  = np.concatenate([d[k]["contact_label"][:].reshape(-1) for k in demos])
    Co = np.concatenate([d[k]["obs"]["contact_label"][:].reshape(-1) for k in demos])
    print("contact frac:", C.mean(), "| top-level == obs/contact_label:", np.array_equal(C, Co))
    for name in ["force", "force_obsbias", "force_rawbias"]:
        F = np.concatenate([d[k]["obs"][name][:].reshape(len(d[k]["obs"][name]), -1) for k in demos])
        a = np.abs(F)
        print(f"\n[{name}] shape {F.shape}  min {F.min():.4g}  max {F.max():.4g}  mean {F.mean():.4g}  std {F.std():.4g}")
        print("  |F| quantiles 50/90/99/99.9:", np.quantile(a, [.5, .9, .99, .999]))
        print("  frac |F|>1e-3:", (a > 1e-3).mean(),
              "| mean|F| contact=1:", a[C == 1].mean(), " contact=0:", a[C == 0].mean())