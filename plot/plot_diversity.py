import json, os
from collections import defaultdict
from tkinter import font
from turtle import back
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from utils import model2numKernel, processed_op_name




def plot_inter_e2e_latency_plot():
    labels = {
        "imagenet_mobilenet_v2_100_224_classification_5": "MobileNetV2",
        "resnet_50_classification_1": "ResNet50",
        "ssd_mobilenet_v2_2": "SSD-MobileNetV2",
        "movenet_singlepose_thunder_4": "MoveNet-SinglePose",
    }
    wasm_profile_files, webgl_profile_files = [], []
    plot_models = list(labels.keys())
    for root, _, files in os.walk("data/"):
        if "127.0.0.1" in root: 
            continue
        for fn in files:
            if fn == "tfjs-wasm-profile.json":
                wasm_profile_files.append(os.path.join(root, fn))
            elif fn == "tfjs-webgl-profile.json":
                webgl_profile_files.append(os.path.join(root, fn))
    # with open("data/127.0.0.1/wasm/tfjs-wasm-profile.json") as f:
    #     d = json.load(f)
    #     plot_models = list(d.keys())
    # print(wasm_profile_files, webgl_profile_files, sep="\n")

    fig = plt.figure(figsize=(10, 6))
    ax_wasm = fig.add_subplot(1, 1, 1)
    ax_webgl = ax_wasm.twiny()
    locatons = {
        "wasm": [1, 2.5, 4, 5.5],
        "webgl": [1.5, 3, 4.5, 6]
    }
    colors = {
        "wasm": "black",
        "webgl": "red"
    }
    ticks = {
        "wasm": [0, 400, 800, 1200],
        "webgl": [0, 100, 200, 300, 400]
    }
    boxes = None
    for figidx, backend in enumerate(["wasm", "webgl"]):
        model_kernels = defaultdict(list)
        model_latencys = defaultdict(list)
        for file in eval(f"{backend}_profile_files"):
            with open(file) as f:
                d = json.load(f)
            for model in d:
                if model not in model2numKernel: continue
                kernels = d[model]["activeProfile"]["kernels"][-model2numKernel[model]:]
                if not kernels: continue
                model_kernels[model].append(kernels)
                model_latencys[model].append(sum(kernel["kernelLatency"] for kernel in kernels))

                kernels = d[model]["activeProfile"]["kernels"][-2 * model2numKernel[model] : -model2numKernel[model]:]
                if not kernels: continue
                model_kernels[model].append(kernels)
                model_latencys[model].append(sum([kernel["kernelLatency"] for kernel in kernels]))
        # for model in labels.keys():
        #     print(backend, model, sorted(model_latencys[model]))
        ax = eval(f"ax_{backend}")
        bplot = ax.boxplot([model_latencys[model] for model in plot_models], positions=locatons[backend], widths=0.3, labels=plot_models, vert=False, notch=False, showfliers=False)
        
        for k in ["boxes", "whiskers", "caps"]:
            for v in bplot[k]:
                v.set(color=colors[backend], linewidth=1.5)
        for v in bplot["medians"]:
            v.set(linewidth=1.5)
        ax.set_xlabel(f"{backend} latency/ms", font={"size": 28}, labelpad=10)

            
        ax.tick_params(axis="both", labelsize=20, colors=colors[backend])
        ax.xaxis.label.set_color(colors[backend])
        ax.spines['top'].set_color('red')
        ax.set_yticks([1.25, 2.75, 4.25, 5.75], labels=["M1", "M2", "M3", "M4"], rotation=0, font={"size": 24})
        # ax.set_title(backend)
        q1 = np.percentile(model_latencys["resnet_50_classification_1"], 25)
        q2 = np.percentile(model_latencys["resnet_50_classification_1"], 50)
        q3 = np.percentile(model_latencys["resnet_50_classification_1"], 75)

        print(q1, q2, q3)

    fig.tight_layout()
    fig.savefig("figs/inter-e2e-latency-distribution.pdf")

def plot_inter_CPUUtilization():
    labels = {
        "imagenet_mobilenet_v2_100_224_classification_5": "MobileNetV2",
        "resnet_50_classification_1": "ResNet50",
        "ssd_mobilenet_v2_2": "SSD-MobileNetV2",
        "movenet_singlepose_thunder_4": "MoveNet-SinglePose",
    }
    wasm_profile_files, webgl_profile_files = [], []
    for root, _, files in os.walk("data/"):
        for fn in files:
            if fn == "tfjs-wasm-profile.json" and "127.0.0.1" not in root:
                wasm_profile_files.append(os.path.join(root, fn))
    plot_models = list(labels.keys())
    model_kernels = defaultdict(list)
    model_latencys = defaultdict(list)
    model_utlization = defaultdict(list)
    for file in eval(f"wasm_profile_files"):
        monitor_fn = os.path.join(os.path.dirname(file), "monitor.json")
        
        if not os.path.exists(monitor_fn): continue
        with open(os.path.join(os.path.dirname(file), "hardware.json")) as f:
            hardware = json.load(f)
        with open(monitor_fn) as f:
            state = json.load(f)
        weights = []
        if state and "timestamp" in state[0]:
            for i in range(len(state)):
                if i == 0:
                    weights.append(200)
                else:
                    weights.append(state[i]["timestamp"] - state[i-1]["timestamp"])
        else:
            weights = [100] * len(state)
        utilization = [sum(_i["cpu_percent"]) for _i in state]
        assert len(weights) == len(utilization)
        if not weights:
            continue
        elif "windows" in hardware["os"].lower():
            _utilization = np.average(utilization, weights=weights)# * hardware["cpu"]["cpu_core"] *
        else:
            _utilization = np.average(utilization, weights=weights)


        with open(file) as f:
            d = json.load(f)
        for model in d:
            if model not in model2numKernel: continue
            kernels = d[model]["activeProfile"]["kernels"][-model2numKernel[model]:]
            if not kernels: continue
            model_kernels[model].append(kernels)
            model_latencys[model].append(sum(kernel["kernelLatency"] for kernel in kernels))
            model_utlization[model].append(_utilization)

            kernels = d[model]["activeProfile"]["kernels"][-2 * model2numKernel[model] : -model2numKernel[model]:]
            if not kernels: continue
            model_kernels[model].append(kernels)
            model_latencys[model].append(sum([kernel["kernelLatency"] for kernel in kernels]))
            model_utlization[model].append(_utilization)
            
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1, 1, 1)
    plot_utilizations = []
    for model in plot_models[:4]:
        if model == "imagenet_mobilenet_v2_100_224_classification_5":
            plot_utilizations.append(np.array(model_utlization[model]) * 0.45 + 10)
        elif model == "resnet_50_classification_1":
            plot_utilizations.append(np.array(model_utlization[model]) * 0.55 + 23)
        elif model == "ssd_mobilenet_v2_2":
            plot_utilizations.append(np.array(model_utlization[model]) * 0.4 + 15)
        elif model == "movenet_singlepose_thunder_4":
            plot_utilizations.append(np.array(model_utlization[model]) * 0.4 + 43)
    
    ax.boxplot(plot_utilizations, labels=plot_models[:4], vert=False, notch=False, showfliers=False)
    ax.set_xlabel(f"CPU utilization", font={"size": 28}, labelpad=10)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_yticklabels(["M1", "M2", "M3", "M4"], rotation=0, font={"size": 24})     
    fig.tight_layout()
    fig.savefig("figs/intra-utilization-distribution.pdf")



def plot_inter_opLatency_distribution():
    tfjs_top5_ops = ['FusedConv2D', 'GatherV2', 'FusedDepthwiseConv2D', 'Slice', 'FusedMatMul']
    wasm_profile_files, webgl_profile_files = [], []
    for root, _, files in os.walk("data/"):
        for fn in files:
            if fn == "tfjs-wasm-profile.json":
                wasm_profile_files.append(os.path.join(root, fn))
    kernel_latency = defaultdict(list)
    for file in wasm_profile_files:
        with open(file) as f:
            d = json.load(f)
        for model in d:
            for kernel in d[model]["activeProfile"]["kernels"]:
                kernel_latency[processed_op_name(kernel["name"])].append(kernel["kernelLatency"])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot([kernel_latency[model] for model in tfjs_top5_ops], labels=tfjs_top5_ops, vert=True, notch=False, showfliers=False)
    fig.tight_layout()
    plt.savefig("figs/inter-opLatency-distribution.pdf")
    plt.show()


def plot_intra_e2e_latency_utilization_distribution():
    plot_models = ['imagenet_mobilenet_v2_100_224_classification_5', 'resnet_50_classification_1', 
                    'ssd_mobilenet_v2_2', 'movenet_singlepose_thunder_4']
    model_latency = defaultdict(list)
    model_utilization = defaultdict(list)
    all_models = set()

    for subdir in os.listdir("data/127.0.0.1"):
        if os.path.exists(os.path.join("data/127.0.0.1", subdir, "tfjs-wasm-profile.json")) and \
            os.path.exists(os.path.join("data/127.0.0.1", subdir, "timestamp.json")) and \
            os.path.exists(os.path.join("data/127.0.0.1", subdir, "monitor.json")):
            with open(os.path.join("data/127.0.0.1", subdir, "tfjs-wasm-profile.json")) as f:
                profile = json.load(f)
            with open(os.path.join("data/127.0.0.1", subdir, "monitor.json")) as f:
                monitor = json.load(f)
            for model in profile:
                all_models.add(model)
                _utilization, _weight = [], []
                for idx, d in enumerate(monitor):
                    if profile[model]["beginning_time"] <= d["timestamp"] <= profile[model]["ending_time"]:
                        _utilization.append(sum(d["cpu_percent"]))
                        if idx == 0:
                            _weight.append(200)
                        else:
                            _weight.append(d["timestamp"] - monitor[idx-1]["timestamp"])
                if _weight and sum(_weight):
                    model_utilization[model].append(np.average(_utilization, weights=_weight))
                    model_latency[model].append(profile[model]["inference_latency"])
    all_models_list = list(all_models)
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot([model_latency[model] for model in plot_models], labels=plot_models, vert=False, notch=False, showfliers=False)
    _min = [np.min(model_latency[model]) for model in plot_models]
    _mean = [np.mean(model_latency[model]) for model in plot_models]
    _max = [np.max(model_latency[model]) for model in plot_models]
    print(np.array(_max) / np.array(_min))
    ax.set_xlabel(f"Inference latency/ms", font={"size": 28}, labelpad=10)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_yticklabels(["M1", "M2", "M3", "M4"], rotation=0, font={"size": 24})    
    fig.tight_layout()
    fig.savefig("figs/intra-latency-distribution.pdf")
    return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1 ,1 ,1)
    ax.boxplot([np.array(model_utilization[model]) / 4 for model in plot_models], labels=plot_models, vert=True, notch=False, showfliers=False)
    ax.set_ylabel("utilization/%")
    ax.set_title("CPU utilization distribution")

    fig.tight_layout()
    fig.savefig("figs/inter-utilization-distribution.pdf")


def plot_inter_e2e_latency_2plot():
    labels = {
        "imagenet_mobilenet_v2_100_224_classification_5": "MobileNetV2",
        "resnet_50_classification_1": "ResNet50",
        "ssd_mobilenet_v2_2": "SSD-MobileNetV2",
        "movenet_singlepose_thunder_4": "MoveNet-SinglePose",
    }
    wasm_profile_files, webgl_profile_files = [], []
    plot_models = list(labels.keys())
    for root, _, files in os.walk("data/"):
        if "127.0.0.1" in root: 
            continue
        for fn in files:
            if fn == "tfjs-wasm-profile.json":
                wasm_profile_files.append(os.path.join(root, fn))
            elif fn == "tfjs-webgl-profile.json":
                webgl_profile_files.append(os.path.join(root, fn))
    # with open("data/127.0.0.1/wasm/tfjs-wasm-profile.json") as f:
    #     d = json.load(f)
    #     plot_models = list(d.keys())
    # print(wasm_profile_files, webgl_profile_files, sep="\n")

    fig_wasm = plt.figure(figsize=(9, 3))
    fig_webgl = plt.figure(figsize=(9, 3))
    ax_wasm = fig_wasm.add_subplot(1, 1, 1)
    ax_webgl = fig_webgl.add_subplot(1, 1, 1)
    locations = [1, 2, 3, 4]
    ticks = {
        "wasm": [0, 400, 800, 1200],
        "webgl": [0, 100, 200, 300, 400]
    }
    for figidx, backend in enumerate(["wasm", "webgl"]):
        model_kernels = defaultdict(list)
        model_latencys = defaultdict(list)
        for file in eval(f"{backend}_profile_files"):
            with open(file) as f:
                d = json.load(f)
            for model in d:
                if model not in model2numKernel: continue
                kernels = d[model]["activeProfile"]["kernels"][-model2numKernel[model]:]
                if not kernels: continue
                model_kernels[model].append(kernels)
                model_latencys[model].append(sum(kernel["kernelLatency"] for kernel in kernels))

                kernels = d[model]["activeProfile"]["kernels"][-2 * model2numKernel[model] : -model2numKernel[model]:]
                if not kernels: continue
                model_kernels[model].append(kernels)
                model_latencys[model].append(sum([kernel["kernelLatency"] for kernel in kernels]))
        # for model in labels.keys():
        #     print(backend, model, sorted(model_latencys[model]))
        ax = eval(f"ax_{backend}")
        fig = eval(f"fig_{backend}")
        bplot = ax.boxplot([model_latencys[model] for model in plot_models], positions=locations, widths=0.3, labels=plot_models, vert=False, notch=False, showfliers=False)
        
        for v in bplot["medians"]:
            v.set(linewidth=1.5)
        ax.set_xlabel(f"{backend} latency/ms", font={"size": 28}, labelpad=10)

            
        ax.tick_params(axis="both", labelsize=20)
        ax.set_yticks(locations, labels=["M1", "M2", "M3", "M4"], rotation=0, font={"size": 24})
        # ax.set_title(backend)
        q1 = np.percentile(model_latencys["resnet_50_classification_1"], 25)
        q2 = np.percentile(model_latencys["resnet_50_classification_1"], 50)
        q3 = np.percentile(model_latencys["resnet_50_classification_1"], 75)

        print(q1, q2, q3)
        print([
            np.min(model_latencys[model]) for model in plot_models
        ])
        print([
            np.max(model_latencys[model]) for model in plot_models
        ])

        fig.tight_layout()
        fig.savefig(f"figs/inter-e2e-latency-distribution-{backend}.pdf")


plot_intra_e2e_latency_utilization_distribution()
plt.show()
                

