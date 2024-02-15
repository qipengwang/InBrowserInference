from cProfile import label
import json
from collections import defaultdict
from tkinter import font
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from matplotlib.ticker import FuncFormatter
from utils import model2numKernel, processed_op_name

subdirs = ["wasm", "wasm-thread", "wasm-simd", "wasm-thread-simd"]
backends = ["wasm", "webgl"]


def plot_kernel_latency():
    sizetype = "output"
    for figidx, backend in enumerate(backends):
        tfjs_op_d = {}
        with open(f"data/127.0.0.1/wasm-thread-simd/tfjs-{backend}-profile.json") as f:
            profile = json.load(f)
        for model in profile:
            for kernel in profile[model]["activeProfile"]["kernels"]:
                op_name = processed_op_name(kernel["name"])
                if op_name not in tfjs_op_d:
                    tfjs_op_d[op_name] = {
                        "size": [],
                        "latency": []
                    }
                tfjs_op_d[op_name]["latency"].append(kernel["kernelLatency"])
                tfjs_op_d[op_name]["size"].append(sum([np.prod(shape) for shape in kernel[f"{sizetype}Shapes"] if shape]))
        tfjs_top5_ops = sorted(tfjs_op_d.keys(), key=lambda k: sum(tfjs_op_d[k]["latency"]), reverse=True)[:5]
        # print(tfjs_op_d[tfjs_top5_ops[0]]["latency"])
        ops = list(tfjs_op_d.keys())
        op_latency = [sum(tfjs_op_d[op]["latency"]) for op in ops]
        op_latency = np.array(op_latency) / sum(op_latency)
        for i, (_x, l) in enumerate(zip(op_latency, ops)):
            if _x <= 0.02: 
                ops[i] = ''
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.pie(op_latency, labels=ops, autopct=lambda pct: f'{pct:.2f}' if pct > 2 else '')
        ax.legend()
        ax.set_title(f"tfjs-{backend}-opLatency-percentage")
        fig.tight_layout()
        fig.savefig(f"figs/tfjs-{backend}-opLatency-percentage.pdf")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        lines = ax.boxplot([tfjs_op_d[op]["latency"] for op in tfjs_top5_ops], vert=True, notch=False, showfliers=False)
        # print(tfjs_op_d[tfjs_top5_ops[0]]["latency"])
        # print(type(lines), lines.values())
        # K1: FusedConv2D; K2: GatherV2; K3: FusedDepthwiseConv2D; K4: Slice; K5: FusedMatMul; K6: MaxPool; K7: DepthwiseConv2dNative
        if backend == "wasm":
            ax.set_xticklabels(["K1", "K2", "K3", "K4", "K5"])
        else:
            ax.set_xticklabels(["K1", "K3", "K5", "K6", "K7"])

        ax.tick_params(axis="both", labelsize=32)
        print(backend, tfjs_top5_ops)
        ax.set_ylabel('Latency/ms', font={"size": 32})
        fig.tight_layout()
        fig.savefig(f"figs/tfjs-{backend}-top5op-latency-distribution.pdf")


def plot_kernel_latency_breakdown():
    # breakdown analysis: wasm, wasm-simd, wasm-thread, wasm-simd-thread
    backend = "wasm"
    sizetype = "output"
    fig1 = plt.figure(figsize=(48, 12))
    fig2 = plt.figure(figsize=(48, 12))
    for idx, subdir in enumerate(subdirs):
        tfjs_op_d = {}
        with open(f"data/127.0.0.1/{subdir}/tfjs-wasm-profile.json") as f:
            profile = json.load(f)
        for model in profile:
            for kernel in profile[model]["activeProfile"]["kernels"]:
                op_name = processed_op_name(kernel["name"])
                if op_name not in tfjs_op_d:
                    tfjs_op_d[op_name] = {
                        "size": [],
                        "latency": [],
                        "bytesAdded": []
                    }
                tfjs_op_d[op_name]["latency"].append(kernel["kernelLatency"])
                tfjs_op_d[op_name]["bytesAdded"].append(kernel["bytesAdded"])
                tfjs_op_d[op_name]["size"].append(sum([np.prod(shape) for shape in kernel[f"{sizetype}Shapes"] if shape]))
        tfjs_top5_ops = sorted(tfjs_op_d.keys(), key=lambda k: sum(tfjs_op_d[k]["latency"]), reverse=True)[:5]
        print(tfjs_top5_ops)
        ops = list(tfjs_op_d.keys())
        op_latency = [sum(tfjs_op_d[op]["latency"]) for op in ops]
        op_latency = np.array(op_latency) / sum(op_latency)
        _ops = ops.copy()
        for i, (_x, l) in enumerate(zip(op_latency, ops)):
            if _x <= 0.02: 
                _ops[i] = ''
        ax = fig1.add_subplot(1, 4, idx + 1)
        lines = ax.boxplot([tfjs_op_d[op]["latency"] for op in tfjs_top5_ops], vert=True, notch=False, showfliers=False)
        ax.set_xticklabels(tfjs_top5_ops)
        ax.set_ylim()
        ax.set_title(subdir)

        ax = fig2.add_subplot(1, 4, idx + 1)
        ax.pie(op_latency, labels=_ops, autopct=lambda pct: f'{pct:.2f}' if pct > 2 else '')
        ax.legend()
        ax.set_title(f"tfjs-{subdir}-breakdown-opLatency-percentage")
        ax.set_title(subdir)
        tfjs_top5_ops_latency = [np.mean(tfjs_op_d[op]["latency"]) for op in tfjs_top5_ops]
        tfjs_ops_latency = [np.mean(tfjs_op_d[op]["latency"]) for op in ops]
        # print(tfjs_top5_ops_latency, tfjs_ops_latency)
        # print(subdir, list(zip(tfjs_top5_ops, tfjs_top5_ops_latency, np.array(tfjs_top5_ops_latency) / sum([sum(tfjs_op_d[op]["latency"]) for op in ops]))))
        # print()
        
    fig1.tight_layout()
    fig1.savefig(f"figs/tfjs-breakdown-top5op-opLatency-distribution.pdf")
    fig2.tight_layout()
    fig2.savefig(f"figs/tfjs-breakdown-top5op-latency-percentage.pdf")


def plot_kernel_memory_proportion():
    tfjs_op_d = {}
    with open(f"data/127.0.0.1/wasm/tfjs-wasm-profile.json") as f:
        profile = json.load(f)
    for model in profile:
        for kernel in profile[model]["activeProfile"]["kernels"]:
            op_name = processed_op_name(kernel["name"])
            if op_name not in tfjs_op_d:
                tfjs_op_d[op_name] = {
                    "size": [],
                    "latency": [],
                    "bytesAdded": []
                }
            tfjs_op_d[op_name]["latency"].append(kernel["kernelLatency"])
            tfjs_op_d[op_name]["bytesAdded"].append(kernel["bytesAdded"])
    tfjs_top5_ops = sorted(tfjs_op_d.keys(), key=lambda k: sum(tfjs_op_d[k]["bytesAdded"]), reverse=True)[:5]
    all_memory = sum([sum(tfjs_op_d[op_name]["bytesAdded"]) for op_name in tfjs_op_d])
    print(tfjs_top5_ops)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    colors = ["black", "gray", "white", "gray", "white"]
    hatches = [None, None, None, "/", "|"]
    for i in range(len(tfjs_top5_ops)):
        ax.bar(i, sum(tfjs_op_d[tfjs_top5_ops[i]]["bytesAdded"]) / all_memory, width=0.45, label=tfjs_top5_ops[i].replace("Depthwise", "Dw"), color=colors[i], edgecolor="k", hatch=hatches[i])
    # ax.set_xticks(np.arange(len(tfjs_top5_ops)), [f"kernel #{i}" for i in range(len(tfjs_top5_ops))])
    ax.set_xticks([])
    ax.legend(prop={'size': 24})
    ax.set_ylabel("Memory footprint proportion/100%", fontdict={"size": 32})
    ax.set_yticks(np.arange(0, 0.5, 0.1))
    ax.tick_params(axis="both", labelsize=28)
    fig.tight_layout()
    fig.savefig(f"figs/tfjs-kernel-memory-proportion.pdf")

    


def plot_memory_growth():
    labels = {
        "imagenet_mobilenet_v2_100_224_classification_5": "MobileNetV2",
        "resnet_50_classification_1": "ResNet50",
        "movenet_singlepose_thunder_4": "MoveNet-SinglePose",
        "esrgan-tf2_1": "ESRGAN",
    }
    model_memory_footprint = defaultdict(list)
    with open(f"data/127.0.0.1/2022-10-07_17-37-13/tfjs-wasm-profile.json") as f:
        profile = json.load(f)
    for model in profile:
        for kernel in profile[model]["activeProfile"]["kernels"]:
            model_memory_footprint[model].append(kernel["curMemFootprint"]["totalJSHeapSize"] / 1024 ** 2)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    # bax = brokenaxes(ylims=((0, 500), (5000, 6500)))
    for _ in range(7): model_memory_footprint["esrgan-tf2_1"].pop(0)
    print(model_memory_footprint["esrgan-tf2_1"][:10])
    for model in model_memory_footprint:
        for i in range(len(model_memory_footprint[model])):
            if i > 0 and model_memory_footprint[model][i] < model_memory_footprint[model][i-1]:
                model_memory_footprint[model][i] = model_memory_footprint[model][i-1]
    colors=["k", "r", "k", "r"]
    linestyles = ["-", "-", "--","--"]
    for i, model in enumerate(model_memory_footprint):
        ax.plot(np.arange(len(model_memory_footprint[model])) / len(model_memory_footprint[model]), model_memory_footprint[model], label=labels[model], color=colors[i], linestyle=linestyles[i])
    ax.set_xlabel("Normalized inference progress", fontdict={"size": 28})
    ax.set_yscale("log")
    ax.set_ylabel("Memory footprint/MB", fontdict={"size": 28})
    ax.tick_params(axis="both", labelsize=28)
    ax.legend(prop={"size": 24})
    ax.set_xticks([0, 0.5, 1])
    fig.tight_layout()
    fig.savefig("figs/tfjs-memory-growth.pdf")



def plot_breakdown_wasm_kernel_latency_proportion_bar():
    breakaxex_ylims = [
        ([0, 0.02], [0.95, 0.97]),
        ([0, 0.10], [0.75, 0.80]),
        ([0, 0.05], [0.90, 0.92]),
        ([0, 0.10], [0.75, 0.80]),
    ]
    colors = {
        'FusedConv2D': "black", 
        'GatherV2': "white", 
        'DepthwiseConv2dNative': "gray", 
        'FusedMatMul': "white", 
        'Slice': "gray", 
        'FusedDepthwiseConv2D': "white"
    }
    hatches = {
        'FusedConv2D': None, 
        'GatherV2': None, 
        'DepthwiseConv2dNative': "X", 
        'FusedMatMul': "/", 
        'Slice': None, 
        'FusedDepthwiseConv2D': "\\"
    }
    results = defaultdict(dict)
    for idx, subdir in enumerate(subdirs):
        print(idx)
        tfjs_op_d = {}
        with open(f"data/127.0.0.1/{subdir}/tfjs-wasm-profile.json") as f:
            profile = json.load(f)
        for model in profile:
            for kernel in profile[model]["activeProfile"]["kernels"]:
                op_name = processed_op_name(kernel["name"])
                if op_name not in tfjs_op_d:
                    tfjs_op_d[op_name] = {
                        "size": [],
                        "latency": [],
                        "bytesAdded": []
                    }
                tfjs_op_d[op_name]["latency"].append(kernel["kernelLatency"])
                tfjs_op_d[op_name]["bytesAdded"].append(kernel["bytesAdded"])

        tfjs_top5_ops = sorted(tfjs_op_d.keys(), key=lambda k: sum(tfjs_op_d[k]["latency"]), reverse=True)[:5]
        all_latency = sum([sum(tfjs_op_d[op_name]["latency"]) for op_name in tfjs_op_d])
        fig = plt.figure(figsize=(8, 8))
        bax = brokenaxes(ylims=breakaxex_ylims[idx], despine=False, d=0.005, fig=fig)
        print(subdir, tfjs_top5_ops)
        for kernel in colors.keys():
            results[subdir][kernel] = sum(tfjs_op_d[kernel]["latency"]) 
        # all_ops.update(tfjs_top5_ops)
        for i in range(len(tfjs_top5_ops)):
            bax.bar(i, sum(tfjs_op_d[tfjs_top5_ops[i]]["latency"]) / all_latency, label=tfjs_top5_ops[i], color=colors[tfjs_top5_ops[i]], edgecolor="k", hatch=hatches[tfjs_top5_ops[i]])
        
        # if idx == 0:
        #     bax.axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}' if int(x * 1000) % 10 == 0 else ""))
        #     bax.axs[1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}' if int(x * 1000) % 10 == 0 else ''))
        # bax.legend(prop={'size': 16})
        bax.set_ylabel("Kernel latency proportion/100%", fontdict={"size": 20}, labelpad=45)
        bax.tick_params(axis="both", labelsize=16)
        # bax.set_title(subdir)
        bax.axs[1].set_xticks([])
        if idx == 0:
            bax.axs[0].set_yticks([0.95, 0.96, 0.97])
            bax.axs[1].set_yticks([0, 0.01, 0.02])
        elif idx == 1:
            bax.axs[0].set_yticks([0.75, 0.80])
            bax.axs[1].set_yticks([0, 0.05, 0.1])
        elif idx == 2:
            bax.axs[0].set_yticks([0.90, 0.92])
            bax.axs[1].set_yticks([0, 0.02, 0.04])
        elif idx == 3:
            bax.axs[0].set_yticks([0.75, 0.80])
            bax.axs[1].set_yticks([0, 0.05, 0.1])
        # break
        # plt.subplots_adjust(left=0.2, bottom=0.05, right=0.98, top=0.95, wspace=0, hspace=0)
        fig.savefig(f"figs/tfjs-wasm-breakdown-{subdir}-proportion-bar.pdf")
    print(results)
    all_kernels = colors.keys()
    for subdir in subdirs:
        print(subdir, "\t\t", end="")
    print()
    for kernel in all_kernels:
        print(kernel, '\t', end="")
        for subdir in subdirs:
            print(f"{results[subdir][kernel]:>10.1f} & ", end='')
        print()


def plot_breakdown_wasm_e2e_latency_proportion_bar():
    labels = {
        "imagenet_mobilenet_v2_100_224_classification_5": "MobileNetV2",
        "resnet_50_classification_1": "ResNet50",
        "ssd_mobilenet_v2_2": "SSD-MobileNetV2",
        "movenet_singlepose_thunder_4": "MoveNet-SinglePose",
    }
    hatches = {
        "imagenet_mobilenet_v2_100_224_classification_5": None,
        "resnet_50_classification_1": "/",
        "ssd_mobilenet_v2_2": "-",
        "movenet_singlepose_thunder_4": "\\",
    }
    data = {
        "imagenet_mobilenet_v2_100_224_classification_5": {
            "wasm": 148.6, 
            "wasm-thread": 122.4, 
            "wasm-simd": 106.5, 
            "wasm-thread-simd": 105.5 
        },
        "resnet_50_classification_1": {
            "wasm": 1275.3, 
            "wasm-thread": 395.7, 
            "wasm-simd": 358.1, 
            "wasm-thread-simd": 172.3 
        },
        "ssd_mobilenet_v2_2": {
            "wasm": 287.4, 
            "wasm-thread": 123.9, 
            "wasm-simd": 126.1, 
            "wasm-thread-simd": 84.4
        },
        "movenet_singlepose_thunder_4": {
            "wasm": 411.7, 
            "wasm-thread": 150.6, 
            "wasm-simd": 130.8, 
            "wasm-thread-simd": 67.3 
        },
    }
    results = defaultdict(dict)
    for idx, subdir in enumerate(subdirs):
        print(idx)
        model_kernel_latency = defaultdict(list)
        model_latency = {}
        with open(f"data/127.0.0.1/{subdir}/tfjs-wasm-profile.json") as f:
            profile = json.load(f)
        for model in profile:
            for kernel in profile[model]["activeProfile"]["kernels"]:
                model_kernel_latency[model].append(kernel["kernelLatency"])
            model_latency[model] = sum(model_kernel_latency[model])
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        bar_width = 0.4
        for i, model in enumerate(labels.keys()):
            ax.bar(i - bar_width/2, model_latency[model], label=labels[model], width=bar_width, color="white", hatch=hatches[model], edgecolor="k")
            ax.bar(i + bar_width/2, data[model][subdir],  label=labels[model], width=bar_width, color="gray", hatch=hatches[model], edgecolor="k")
        ax.set_xticks([])
        if idx == 2:
            ax.set_yticks(np.arange(0, 801, 200))
        ax.tick_params(axis="both", labelsize=24)
        ax.set_ylabel("Inference latency/ms", fontdict={"size": 32})
        plt.subplots_adjust(left=0.2, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
        fig.savefig(f"figs/tfjs-wasm-breakdown-{subdir}-e2e-latency-bar.pdf")

    


plot_breakdown_wasm_e2e_latency_proportion_bar()
plt.show()         
    