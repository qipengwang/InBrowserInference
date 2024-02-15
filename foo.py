import json
import numpy as np

with open("tmp") as f:
    d = []
    for line in f:
        line = line.strip().split("&")
        w, w_t, w_s, w_t_s = line[3], line[6], line[9], line[12].replace("\\", "")
        d.append(
            [float(w.strip()), float(w_t.strip()), float(w_s.strip()), float(w_t_s.strip())]
        )
    d = np.array(d)
    print("thread", 1-np.min(d[:, 1] / d[:, 0]), 1-np.max(d[:, 1] / d[:, 0]), 1-np.mean(d[:, 1] / d[:, 0]))
    print("simd", 1-np.min(d[:, 2] / d[:, 0]), 1-np.max(d[:, 2] / d[:, 0]), 1-np.mean(d[:, 2] / d[:, 0]))
    print("S+T", 1-np.min(d[:, 3] / d[:, 0]), 1-np.max(d[:, 3] / d[:, 0]), 1-np.mean(d[:, 3] / d[:, 0]))