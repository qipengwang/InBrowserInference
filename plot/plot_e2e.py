import json, os
from collections import defaultdict
from turtle import tilt
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from utils import model2numKernel, processed_op_name


def plot_e2e_resnet50():
    data = {
        "TF-CPU": [6407.3, 997.4, 82.2],
        "TF-GPU": [6223.4, 1000.4, 11.2],
        "ONNX-CPU": [488.0, 25.4, 23.4],
        "ONNX-GPU": [130.7, 187.7, 5.6],

        # "tfjs-wasm": [2362.6, 1984.3, 1710.3],
        # "tfjs-wasm-T": [2109.7, 1724.7, 1416.1],
        "tfjs-wasm": [2485.1, 899.7, 539.5],  # -S
        # "tfjs-wasm-T-S": [1665.2, 647.3, 300.8],

        # "ort-wasm": [699.1, 394.1, 395.7],
        # "ort-wasm-T": [539.3, 1299.5, 1275.3],
        "ortjs-wasm": [2412.3, 411.9, 358.1],  # -S
        # "ort-wasm-T-S": [1219.3, 306.9, 172.3],

        "tfjs-WebGL-I": [1635.7, 2814.2, 248.3],
        "tfjs-WebGL-D": [2482.7, 5443.6, 79.8],
        "ortjs-WebGL-I": [720.6, 1599.6, 183.0],
        "ortjs-WebGL-D": [2011.8, 2810.6, 78.2],
    }
    keys = list(data.keys())
    setup = [data[k][0] for k in keys]
    warmup = [data[k][1] for k in keys]
    inference = [data[k][2] for k in keys]
    print(sorted(np.array(list(data.values())).reshape(-1)))

    fig = plt.figure(figsize=(24, 6))
    # ax = brokenaxes(ylims=((0, 3000), (5000, 6500)), d=0.005, tilt=30, fig=fig, despine=False, yscale="log")
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale("log")
    x = np.arange(len(keys))
    width = 0.25
    ax.bar(x - width, setup, width, label='setup', color='k', edgecolor="black")
    ax.bar(x, warmup, width, label='warmup', color="gray", edgecolor="black")
    ax.bar(x + width, inference, width, label='inference', color="white", edgecolor="black")
    ax.set_ylabel('Latency/ms', font={"size": 24})
    ax.tick_params(axis='y', labelsize=24)
    # ax.set_title('ResNet50 Latency of differnet stage on different backend and framework')
    # x轴刻度标签位置不进行计算
    ax.set_xticks(x, labels=keys, rotation=15, font={"size": 24})
    # ax.legend(bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.subplots_adjust(left=0.1, bottom=0.55, right=0.95, top=0.95, wspace=0, hspace=0)
    fig.savefig("figs/e2e-resnet50.pdf")
    plt.show()





plot_e2e_resnet50()

