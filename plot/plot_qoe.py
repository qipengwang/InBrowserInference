import json, os
from collections import defaultdict
from turtle import tilt
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1,1,1)
with open("plot/qoe_data/wasm-4thread") as f:
    d = json.load(f)
for i in range(len(d)):
    d[i][0] -= 10000
print(np.array(d)[:, 1])
ax.plot(np.array(d)[:, 0]/1000, np.array(d)[:, 1]*1000, color='k', linestyle='--', label="tfjs-wasm")
with open("plot/qoe_data/webgl") as f:
    d = json.load(f)
for i in range(len(d)):
    d[i][0] -= 10000
ax.plot(np.array(d)[:, 0]/1000, np.array(d)[:, 1], color='r', linestyle='--', label="tfjs-WebGL-I")
ax.hlines(30, min(np.array(d)[:, 0]/1000), max(np.array(d)[:, 0]/1000), 'k', label="Ideal case: video")
ax.hlines(60, min(np.array(d)[:, 0]/1000), max(np.array(d)[:, 0]/1000), 'r', label="Ideal case: rendering")
# ax.legend(prop={"size": 15}, ncol=2, loc='lower right')
ax.set_xlim((0, 200))
ax.set_xlabel("Time/s", fontdict={"size": 24})
ax.set_ylabel("fps", fontdict={"size": 24})
ax.tick_params(axis="both", labelsize=24)
fig.tight_layout()
fig.savefig("figs/fps-example.pdf")
plt.show()
