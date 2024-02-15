import os
import json
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import warnings
from utils import model2numKernel, processed_op_name
warnings.filterwarnings("ignore")


NUM_ITERATION = 2
IP_DIRTIME = [
    ['10.172.142.52', '2022-08-29_12-01-35'],  # ok
    ['10.172.141.73', '2022-08-30_00-37-46'],  # ok
    # ['10.90.23.162', '2022-08-29_06-55-43'],  # 
    # ['10.90.7.138', '2022-08-28_15-10-49'],  # 
    ['10.172.150.79', '2022-08-30_09-04-47'],  # ok
    ['10.90.22.127', '2022-08-31_01-35-36'],  # ok
    ['10.90.6.18', '2022-09-01_23-21-43'], # very ok
]


def get_valide_data():
    valid_data = []
    for ip in os.listdir('data'):
        best_time = None
        max_num = 4
        for dirtime in os.listdir(os.path.join("data", ip)):
            if len(os.listdir(os.path.join("data", ip, dirtime))) == 9:
                try:
                    with open(os.path.join("data", ip, dirtime, "hardware.json")) as f:
                        hardware = json.load(f)
                    with open(os.path.join("data", ip, dirtime, "ort-wasm-profile.json")) as f:
                        ort_wasm = json.load(f)
                    with open(os.path.join("data", ip, dirtime, "ort-webgl-profile.json")) as f:
                        ort_webgl = json.load(f)
                    with open(os.path.join("data", ip, dirtime, "tfjs-wasm-profile.json")) as f:
                        tfjs_wasm = json.load(f)
                    with open(os.path.join("data", ip, dirtime, "tfjs-webgl-profile.json")) as f:
                        tfjs_webgl = json.load(f)
                    with open(os.path.join("data", ip, dirtime, "timestamp.json")) as f:
                        timestamp = json.load(f)
                    _tmp = [len(tfjs_wasm), len(tfjs_webgl), len(ort_wasm), len(ort_webgl)]
                    if all(_tmp) and sum(_tmp) > max_num:
                        max_num = sum(_tmp)
                        best_time = dirtime
                except (json.JSONDecodeError, IndexError) as e:
                    pass
        if best_time:
            valid_data.append([ip, best_time])
    print("-" * 20)
    print(*valid_data, sep="\n")


def count_hardware():
    all_cpu = []
    all_gpu = []
    all_memory = []

    for ip, dirtime in IP_DIRTIME:
        with open(os.path.join("data", ip, dirtime, "hardware.json")) as f:
            hardware = json.load(f)
        if not hardware["cpu"]["cpu_name"]:
            hardware["cpu"]["cpu_name"] = hardware["cpu_model"]
        all_cpu.append(hardware["cpu"])
        if (hardware["igpu"] or hardware["dgpu"]) == 'i386':
            all_gpu.append(hardware["cpu_model"])
        else:
            all_gpu.append(hardware["igpu"] or hardware["dgpu"])
        all_memory.append(round(hardware["memory"], 0))
    print(all_cpu, all_gpu, all_memory, sep="\n---------------\n")


def count_op():
    tfjs_op = set()
    ort_op = set()
    for ip, dirtime in IP_DIRTIME:
        try:
            with open(os.path.join("data", ip, dirtime, "ort-wasm-profile.json")) as f:
                ort_wasm = json.load(f)
            with open(os.path.join("data", ip, dirtime, "ort-webgl-profile.json")) as f:
                ort_webgl = json.load(f)
            with open(os.path.join("data", ip, dirtime, "tfjs-wasm-profile.json")) as f:
                tfjs_wasm = json.load(f)
            with open(os.path.join("data", ip, dirtime, "tfjs-webgl-profile.json")) as f:
                tfjs_webgl = json.load(f)
            for model in tfjs_wasm:
                for kernel in tfjs_wasm[model]["activeProfile"]["kernels"]:
                    tfjs_op.add(kernel["name"])
            for model in tfjs_webgl:
                for kernel in tfjs_webgl[model]["activeProfile"]["kernels"]:
                    tfjs_op.add(kernel["name"])
            for model in ort_wasm:
                try:
                    # print(model, ort_wasm[model].keys())
                    for kernel in ort_wasm[model]["kernels"]:
                        ort_op.add(kernel["args"]["op_name"])
                except:
                    print(model, ort_wasm[model].keys(), )
            for model in ort_webgl:
                for kernel in ort_webgl[model]["kernels"]:
                    ort_op.add(kernel.split("run")[1].strip().split()[0].replace(":flush", "").replace("'", ""))
        except KeyError:
            pass
    print(sorted(tfjs_op))
    print("---------------------")
    print(sorted(ort_op))


def plot_op_latency_size(sizetype="output"):
    '''
    For tfjs/ort, the relation between op latency and its input/output size
    select top-3 op whose latency account for more percentage
    '''
    tfjs_op_d, ort_op_d = {}, {}
    for ip, dirtime in IP_DIRTIME:
        with open(os.path.join("data", ip, dirtime, "ort-wasm-profile.json")) as f:
            ort_wasm = json.load(f)
        with open(os.path.join("data", ip, dirtime, "tfjs-wasm-profile.json")) as f:
            tfjs_wasm = json.load(f)
        for model in tfjs_wasm:
            for kernel in tfjs_wasm[model]["activeProfile"]["kernels"]:
                op_name = processed_op_name(kernel["name"])
                if op_name not in tfjs_op_d:
                    tfjs_op_d[op_name] = {
                        "size": [],
                        "latency": []
                    }
                tfjs_op_d[op_name]["latency"].append(kernel["kernelLatency"])
                tfjs_op_d[op_name]["size"].append(sum([np.prod(shape) for shape in kernel[f"{sizetype}Shapes"] if shape]))

        for model in ort_wasm:
            if "kernels" in ort_wasm[model]:
                for kernel in ort_wasm[model]["kernels"]:
                    op_name = processed_op_name(kernel["args"]["op_name"])
                    if op_name not in ort_op_d:
                        ort_op_d[op_name] = {
                            "size": [],
                            "latency": []
                        }
                    ort_op_d[op_name]["latency"].append(kernel["dur"] / 1000)
                    ort_op_d[op_name]["size"].append([sum([np.prod(list(shape.values())[0]) for shape in kernel["args"][f"{sizetype}_type_shape"] if list(shape.values())[0]])])
        
        tfjs_top3_ops = sorted(tfjs_op_d.keys(), key=lambda k: sum(tfjs_op_d[k]["latency"]), reverse=True)[:3]
        ort_top3_ops = sorted(ort_op_d.keys(), key=lambda x: sum(ort_op_d[x]["latency"]), reverse=True)[:3]

        fig = plt.figure(figsize=(24, 12))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        for i, op in enumerate(tfjs_top3_ops):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.scatter(tfjs_op_d[op]["size"], tfjs_op_d[op]["latency"], label=op)
            ax.set_xlabel(f"{sizetype} size")
            ax.set_ylabel("latency / ms")
            ax.ticklabel_format(style='sci',scilimits=(0, 0),axis='x')
            ax.set_title(f"tfjs {op}")
        for i, op in enumerate(ort_top3_ops):
            ax = fig.add_subplot(2, 3, i + 4)
            ax.scatter(ort_op_d[op]["size"], ort_op_d[op]["latency"], label=op)
            ax.set_xlabel(f"{sizetype} size")
            ax.set_ylabel("latency / ms")
            ax.ticklabel_format(style='sci',scilimits=(0, 0),axis='x')
            ax.set_title(f"ort {op}")
        
        fig.savefig(f"figures/op_latency_{sizetype}size.png")

# op_latency_size("input")
# op_latency_size("output")

def plot_model_latency_distribution():
    '''
    For each model, the distribution of model latency, which is shown in a boxplot
    '''
    latency_distribution = {
        "tfjs": {
            "webgl": defaultdict(list),
            "wasm": defaultdict(list),
        },
        "ort": {
            "webgl": defaultdict(list),
            "wasm": defaultdict(list),
        },
    }

    for ip, dirtime in IP_DIRTIME:
        with open(os.path.join("data", ip, dirtime, "ort-wasm-profile.json")) as f:
            ort_wasm = json.load(f)
        with open(os.path.join("data", ip, dirtime, "ort-webgl-profile.json")) as f:
            ort_webgl = json.load(f)
        with open(os.path.join("data", ip, dirtime, "tfjs-wasm-profile.json")) as f:
            tfjs_wasm = json.load(f)
        with open(os.path.join("data", ip, dirtime, "tfjs-webgl-profile.json")) as f:
            tfjs_webgl = json.load(f)
        for model in tfjs_wasm:
            inference_latency = sum([kernel["kernelLatency"] for kernel in tfjs_wasm[model]["activeProfile"]["kernels"]]) / 6
            latency_distribution["tfjs"]["wasm"][model].append(inference_latency)
        for model in tfjs_webgl:
            inference_latency = sum([kernel["kernelLatency"] for kernel in tfjs_webgl[model]["activeProfile"]["kernels"]]) / 6
            latency_distribution["tfjs"]["webgl"][model].append(inference_latency)
        for model in ort_wasm:
            if "kernels" in ort_wasm[model]:
                inference_latency = sum([kernel["dur"] / 1000 for kernel in ort_wasm[model]["kernels"]]) / 6
                latency_distribution["ort"]["wasm"][model].append(inference_latency)
        for model in ort_webgl:
            if "kernels" in ort_webgl[model]:
                inference_latency = sum([float(kernel.split("ms")[0]) for kernel in ort_webgl[model]["kernels"]]) / 6
                latency_distribution["ort"]["webgl"][model].append(inference_latency)
    
    fig = plt.figure(figsize=(40, 24))
    for i, framework in enumerate(["tfjs", "ort"]):
        for j, backend in enumerate(["wasm", "webgl"]):
            ax = fig.add_subplot(2, 2, i * 2 + j + 1)
            ax.set_title(f"{framework} {backend}")
            ax.set_xlabel("latency/ms")
            ax.boxplot(latency_distribution[framework][backend].values(), vert=False)
            ax.set_yticklabels(latency_distribution[framework][backend].keys())
    fig.tight_layout()
    fig.savefig("figures/latency_distribution.png")


def plot_op_latency_percentage():
    '''
    For each model, the percentage of each op latency, which is shown in a pie plot
    '''
    tfjs_model_oplatency = {
        "resnet_50_classification_1": defaultdict(int),
        "movenet_singlepose_thunder_4": defaultdict(int),
        "InceptionV3": defaultdict(int),
    }
    ort_model_oplatency = {
        "resnet_50_classification_1": defaultdict(int),
        "movenet_singlepose_thunder_4": defaultdict(int),
        "InceptionV3": defaultdict(int),
    }
    for ip, dirtime in IP_DIRTIME:
        with open(os.path.join("data", ip, dirtime, "ort-wasm-profile.json")) as f:
            ort_wasm = json.load(f)
        with open(os.path.join("data", ip, dirtime, "tfjs-wasm-profile.json")) as f:
            tfjs_wasm = json.load(f)
        
        for _, model in enumerate(tfjs_model_oplatency.keys()):
            for kernel in tfjs_wasm[model]["activeProfile"]["kernels"]:
                op_name = processed_op_name(kernel["name"])
                tfjs_model_oplatency[model][op_name] += kernel["kernelLatency"]
            for kernel in ort_wasm[model]["kernels"]:
                op_name = processed_op_name(kernel["args"]["op_name"])
                ort_model_oplatency[model][op_name] += kernel["dur"] / 1000

    fig = plt.figure(figsize=(24, 16))
    for ith, framework in enumerate(["tfjs", "ort"]):
        d = eval(f"{framework}_model_oplatency")
        for jth, model in enumerate(d.keys()):
            model_latency = sum(d[model].values())
            labels, x = list(zip(*sorted(d[model].items(), key=lambda x: -x[1])))
            labels = list(labels)
            x = np.array(x) / model_latency
            for i, (_x, l) in enumerate(zip(x, labels)):
                if _x <= 0.05: labels[i] = ''
            ax = fig.add_subplot(2, 3, jth+1 + ith*3)
            ax.pie(x, labels=labels, autopct=lambda pct: f'{pct:.2f}' if pct > 5 else '')
            ax.legend()
            ax.set_title(f"{framework}-wasm {model}")
    fig.savefig("figures/oplatency_percentage.png")
    

def plot_inference_cpu_utilization():
    '''
    For CPU/wasm backend, for each model, the relation between inference progress and latency/memory/CPU utilization
    '''

    for ip, dirtime in IP_DIRTIME[-1:]:
        with open(os.path.join("data", ip, dirtime, "ort-wasm-profile.json")) as f:
            ort_wasm = json.load(f)
        with open(os.path.join("data", ip, dirtime, "tfjs-wasm-profile.json")) as f:
            tfjs_wasm = json.load(f)
        with open(os.path.join("data", ip, dirtime, "timestamp.json")) as f:
            timestamp = json.load(f)
        with open(os.path.join("data", ip, dirtime, "monitor.json")) as f:
            monitor = json.load(f)
        with open(os.path.join("data", ip, dirtime, "hardware.json")) as f:
            hardware = json.load(f)
        tfjs_model_latency, ort_model_latency = defaultdict(list), defaultdict(list)
        for framework in ["tfjs", "ort"]:
            data = eval(f"{framework}_wasm")
            for model in data:
                if framework == "tfjs":
                    for kernel in data[model]["activeProfile"]["kernels"]:
                        eval(f"{framework}_model_latency")[model].append(kernel["kernelLatency"])
                elif "kernels" in data[model]:
                    for kernel in data[model]["kernels"]:
                        eval(f"{framework}_model_latency")[model].append(kernel["dur"] / 1000)
                inference_latency = sum(eval(f"{framework}_model_latency")[model])
                inference_start_time = float(timestamp[framework]["wasm"][model])
                avg_cpu_util = []
                peak_mem_usage = 0
                for state in monitor:
                    if inference_start_time <= state["timestamp"] <= inference_start_time + inference_latency:
                        avg_cpu_util.append(sum(state["cpu_percent"]))
                        peak_mem_usage = max(peak_mem_usage, state["virtual_memory"][0][0] + state["virtual_memory"][1][0])

                print(framework, model, np.mean(avg_cpu_util), peak_mem_usage, inference_latency / 6)
            print("-" * 20)
        print("*" * 40)

        tfjs_d = {
            "resnet_50_classification_1": defaultdict(list),
            "movenet_singlepose_thunder_4": defaultdict(list),
            "InceptionV3": defaultdict(list),
        }
        ort_d = {
            "resnet_50_classification_1": defaultdict(list),
            "movenet_singlepose_thunder_4": defaultdict(list),
            "InceptionV3": defaultdict(list),
        }
        fig =  plt.figure(figsize=(30, 16))
        for r, framework in enumerate(["tfjs", "ort"]):
            for c, model in enumerate(tfjs_d):
                ith, jth = 0, 0
                inference_time = float(timestamp[framework]["wasm"][model])
                print(framework)
                kernel_len = len(eval(f"{framework}_wasm")[model]["activeProfile"]["kernels"]) if framework == "tfjs" else len(eval(f"{framework}_wasm")[model]["kernels"])
                while ith < kernel_len and jth < len(monitor):
                    # print(f"cur {inference_time} ith={ith}, monitor at {monitor[jth]['timestamp']} jth={jth}", end="\t")
                    if float(monitor[jth]["timestamp"]) < inference_time:
                        jth += 1
                        # print(f"jth += 1 --> {jth}")
                    else:
                        if framework == "tfjs": 
                            eval(f"{framework}_d")[model]["latency"].append(eval(f"{framework}_wasm")[model]["activeProfile"]["kernels"][ith]["kernelLatency"])
                        else:
                            eval(f"{framework}_d")[model]["latency"].append(eval(f"{framework}_wasm")[model]["kernels"][ith]["dur"] / 1000)
                        inference_time += eval(f"{framework}_d")[model]["latency"][-1]
                        ith += 1
                        eval(f"{framework}_d")[model]["utilization"].append(sum(monitor[jth]["cpu_percent"]))
                        # print(f"append cpu percent {sum(monitor[jth]['cpu_percent'])} to list")
                
                per_inference_len = kernel_len // NUM_ITERATION
                # eval(f"{framework}_d")[model]["latency"] = eval(f"{framework}_d")[model]["latency"][-per_inference_len:]
                # eval(f"{framework}_d")[model]["utilization"] = eval(f"{framework}_d")[model]["utilization"][-per_inference_len:]
                print(len(eval(f"{framework}_d")[model]["latency"]), len(eval(f"{framework}_d")[model]["utilization"]))
                ax = fig.add_subplot(2, 3, r * 3 + c + 1)
                ln1 = ax.bar(range(len(eval(f"{framework}_d")[model]["latency"])), eval(f"{framework}_d")[model]["latency"], color="red", label="latency")
                rect = [patches.Rectangle((0,0),1,1, color="red", label="latency")]
                ax.add_patch(rect[0])
                ax.set_xlabel("inference progress/ith op")
                ax.set_ylabel(f"latency / ms")
                ax.yaxis.label.set_color('red')

                ax1 = ax.twinx()
                ln2 = ax1.plot(eval(f"{framework}_d")[model]["utilization"], color="blue", label="CPU-util")
                ax1.set_ylabel(f"CPU util")
                ax1.yaxis.label.set_color('blue')
                ax.set_title(model)
                ax.set_xticks(list(range(0, kernel_len, per_inference_len)), labels=["1st", "2nd"])
                # ax.set_xticklabels()
                # ax.legend(handles=ln2)
                # input()
                print('- ' * 20)
        fig.tight_layout()
        fig.legend(handles=rect + ln2, loc="upper center", ncol=2)
        fig.savefig("figures/inference_cpu_utilization.png")

plot_inference_cpu_utilization()



        


        
        

                



