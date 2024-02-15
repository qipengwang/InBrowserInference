from cProfile import label
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from more_itertools import chunked
import json


MODELS = [
    'imagenet_mobilenet_v2_100_224_classification_5',
    'resnet_50_classification_1',
    'ssd_mobilenet_v2_2',
    'esrgan-tf2_1'
]

BACKENDS_THREADS = [
    ['cpu', 1],
    ['wasm', 1],
    ['wasm', 2],
    ['wasm', 4],
]

def plotAtkernelLevel():
    for b, t in BACKENDS_THREADS:
        latency_perkernel = defaultdict(dict)
        required_bytes_perkernel = defaultdict(dict)
        added_bytes = defaultdict(list)
        total_bytes = defaultdict(list)
        expected_bytes = defaultdict(list)
        with open(f'log/{b}.{t}.log') as f:
            for line in f:
                if b in line and 'inference' in line:
                    _, model, _, kernel_id, kernel_name, bytesAdded, totalBytesSnapshot, kernelTimeMs = line.strip().split('\t')
                    kernel_name = kernel_name.split(':')[1]
                    bytesAdded = bytesAdded.split(':')[1]
                    totalBytesSnapshot = totalBytesSnapshot.split(':')[1]
                    kernelTimeMs = kernelTimeMs.split(':')[1]
                    if kernel_name not in latency_perkernel[model]:
                        latency_perkernel[model][kernel_name] = []
                    latency_perkernel[model][kernel_name].append(float(kernelTimeMs))
                    added_bytes[model].append(int(bytesAdded))
                    total_bytes[model].append(int(totalBytesSnapshot))
                    if kernel_name not in required_bytes_perkernel[model]:
                        required_bytes_perkernel[model][kernel_name] = []
                    required_bytes_perkernel[model][kernel_name].append(bytesAdded)
        
        ########################### plot latency per kernel ############################
        for model in latency_perkernel:
            plt.figure()
            mean_latency = [(kernel, np.mean(latency_perkernel[model][kernel])) for kernel in latency_perkernel[model]]
            mean_latency = sorted(mean_latency, key=lambda x: x[1], reverse=True)[:5]
            plt.boxplot([latency_perkernel[model][k] for k, _ in mean_latency])

            plt.ylabel('latency/ms')
            plt.title(f'{model}.{b}.{t}' + "top-5 compute intensive")
            plt.legend([k for k, _ in mean_latency])
            os.makedirs('figs/latency', exist_ok=True)
            plt.savefig(f'figs/latency/{model}.{b}.{t}.pdf')
            print()
            total_latency = 0
            for k in latency_perkernel[model]:
                total_latency+=sum(latency_perkernel[model][k])
            for k, _ in mean_latency:
                print(k, sum(latency_perkernel[model][k]) / total_latency)

                    

        ########################### plot memory footprint ############################
        for model in added_bytes:
            print(model)
            plt.figure()
            cur_bytes = total_bytes[model][0] - added_bytes[model][0];
            for i in added_bytes[model]:
                cur_bytes += i
                expected_bytes[model].append(cur_bytes)
            plt.plot(np.array(expected_bytes[model]) / 1024**2, label='expected_bytes')
            plt.plot(np.array(total_bytes[model]) / 1024**2, label='total_bytes')
            plt.xlabel('kernel id')
            plt.ylabel('memory/MB')
            plt.title(model)
            plt.legend()
            os.makedirs('figs/memory', exist_ok=True)
            plt.savefig(f'figs/memory/{model}.{b}.{t}.pdf')

        


def plot_breakdown(logname):
    print(logname)
    infos = []
    modelname, backend, thread_num, _, _ = logname.split('.')   # esrgan-tf2_1.wasm.4.all.log
    savedir = f'figs/breakdown/{modelname}'
    os.makedirs(savedir, exist_ok=True)
    with open(f'log/{logname}') as f:
        for line in f:
            if f'wasm	{modelname}	all' in line:
                _, model, _, index, kernel_name, bytesAdded, totalBytesSnapshot, kernelTimeMs = line.strip().split('\t')
                infos.append({
                    'index': int(index),
                    'kernel_name': kernel_name.split(':')[1],
                    'bytesAdded': int(bytesAdded.split(':')[1]),
                    'totalBytesSnapshot':int(totalBytesSnapshot.split(':')[1]),
                    'kernelTimeMs': float(kernelTimeMs.split(':')[1])
                })
    print(len(infos))
    infos = list(chunked(infos, int(len(infos) / 11)))
    warmup_info = infos[0]
    inference_infos = infos[1:]

    def plot_latency_percentage():
        warmup_latency = sum([i['kernelTimeMs'] for i in warmup_info])
        warmup_kernel_latency = defaultdict(list)
        for kernel_info in warmup_info:
            warmup_kernel_latency[kernel_info['kernel_name']].append(kernel_info['kernelTimeMs'])
        for kn in warmup_kernel_latency:
            print('warmup:', kn, np.sum(warmup_kernel_latency[kn]) / warmup_latency)
        warmup_latency_percentage = {}
        for kn in warmup_kernel_latency:
            warmup_latency_percentage[kn] = np.sum(warmup_kernel_latency[kn]) / warmup_latency
        labels, x = list(zip(*sorted(warmup_latency_percentage.items(), key=lambda x: -x[1])))
        for k, v in sorted(warmup_latency_percentage.items(), key=lambda x: -x[1]):
            print(k, v)
        plt.figure(figsize=(10, 10))

        filtered_labels = []
        for _x, l in zip(x, labels):
            if _x > 0.05: filtered_labels.append(l)
            else: filtered_labels.append('')
        plt.pie(x, labels=filtered_labels, autopct=lambda pct: f'{pct:.2f}' if pct > 5 else '')
        plt.legend()
        plt.title(f'{modelname}.{backend}.{thread_num}.warmup')
        plt.savefig(f'{savedir}/{modelname}.{backend}.{thread_num}.warmup.percentage.pdf', bbox_inches='tight')
        print()

        total_inference_latency = 0
        inference_kernel_latency = defaultdict(list)
        for inference_info in inference_infos:
            total_inference_latency += sum([i['kernelTimeMs'] for i in inference_info])
            for kernel_info in inference_info:
                inference_kernel_latency[kernel_info['kernel_name']].append(kernel_info['kernelTimeMs'])
        for kn in inference_kernel_latency:
            print('inference:', kn, np.sum(inference_kernel_latency[kn]) / total_inference_latency)
        inference_latency_percentage = {}
        for kn in inference_kernel_latency:
            inference_latency_percentage[kn] = np.sum(inference_kernel_latency[kn]) / total_inference_latency
        labels, x = list(zip(*sorted(inference_latency_percentage.items(), key=lambda x: -x[1])))
        for k, v in sorted(inference_latency_percentage.items(), key=lambda x: -x[1]):
            print(k, v)
        plt.figure(figsize=(10, 10))
        filtered_labels = []
        for _x, l in zip(x, labels):
            if _x > 0.05: filtered_labels.append(l)
            else: filtered_labels.append('')
        plt.pie(x, labels=filtered_labels, autopct=lambda pct: f'{pct:.2f}' if pct > 5 else '')
        plt.legend()
        plt.title(f'{modelname}.{backend}.{thread_num}.warmup')
        plt.savefig(f'{savedir}/{modelname}.{backend}.{thread_num}.inference.percentage.pdf', bbox_inches='tight')

    def plot_progressively():
        expected_memory, real_memory = [], []
        total_latency = []
        cur_mem = warmup_info[0]['totalBytesSnapshot'] - warmup_info[0]['bytesAdded']
        cur_latency = 0
        for info in warmup_info:
            cur_mem += info['bytesAdded']
            expected_memory.append(cur_mem)
            real_memory.append(info['totalBytesSnapshot'])
            cur_latency += info['kernelTimeMs']
            total_latency.append(cur_latency)
        
        ax = plt.figure(figsize=(12, 8)).add_subplot(111)
        ax.set_ylabel('memory/MB')
        ln1 = ax.plot(np.array(expected_memory) / 1024**2, label='expected_memory', color='b')
        ln2 = ax.plot(np.array(real_memory) / 1024**2, label='allocated_memory-tfjs', color='r')
        ax2 = ax.twinx()
        ln3 = ax2.plot(total_latency, label='total_latency', color='k')
        ax2.set_ylabel('latency/ms')

        plt.xticks(range(0, len(expected_memory), len(expected_memory)//5))
        plt.xlabel('inference process')
        plt.title(f'{modelname}.{backend}.{thread_num}.warmup')
        lns = ln1 + ln2 + ln3
        plt.legend(lns, [l.get_label() for l in lns])
        plt.savefig(f'{savedir}/{modelname}.{backend}.{thread_num}.warmup.pdf', bbox_inches='tight')

        ax = plt.figure(figsize=(12, 8)).add_subplot(111)
        ax.set_ylabel('memory/MB')
        ax2 = ax.twinx()
        ax2.set_ylabel('latency/ms')

        ln1 = ax.plot(np.array(expected_memory) / 1024**2, label='expected_memory', color='b')
        ln3 = ax2.plot(total_latency, label='total_latency', color='k')

        for idx, infos in enumerate(inference_infos):
            cur_mem = infos[0]['totalBytesSnapshot'] - infos[0]['bytesAdded']
            cur_latency = 0
            expected_memory, total_latency = [], []
            for info in infos:
                cur_mem += info['bytesAdded']
                expected_memory.append(cur_mem)
                real_memory.append(info['totalBytesSnapshot'])
                cur_latency += info['kernelTimeMs']
                total_latency.append(cur_latency)
            ln1 = ax.plot(range((idx+1) * len(warmup_info), (idx+2) * len(warmup_info)), np.array(expected_memory) / 1024**2, label='expected_memory', color='b')
            ln3 = ax2.plot(range((idx+1) * len(warmup_info), (idx+2) * len(warmup_info)), total_latency, label='total_latency', color='k')
        
        ln2 = ax.plot(np.array(real_memory) / 1024**2, label='real_memory', color='r')

        for i in range(11):
            ax.vlines(i*len(warmup_info), 0, max(expected_memory) / 1024**2, colors='gray', linestyles='dashed')
        plt.xlabel('inference process')
        plt.xticks(range(0, len(real_memory), len(warmup_info)))
        plt.title(f'{modelname}.{backend}.{thread_num}.all')
        lns = ln1 + ln2 + ln3
        plt.legend(lns, [l.get_label() for l in lns])
        plt.savefig(f'{savedir}/{modelname}.{backend}.{thread_num}.all.pdf', bbox_inches='tight')
    
    plot_latency_percentage()
    plot_progressively()


def parse_log():
    ort_wasm_log_dict = defaultdict(dict)
    ort_webgl_log_dict = defaultdict(dict)
    ort_cur_model = None
    ort_webgl_model_list = []
    new_model_log_flag = True
    ort_webgl_cur_model_index = 0
    ort_webgl_mem_cache = []

    with open('data/all-log.log') as f:
        for line in f:
            if "ORT_BEGIN_INFERENCE:wasm" in line:
                ort_cur_model = line.split("ORT_BEGIN_INFERENCE:wasm")[1].strip()
            elif "logEventInmediately" in line:
                if "kernels" not in ort_wasm_log_dict[ort_cur_model]:
                    ort_wasm_log_dict[ort_cur_model]["kernels"] = []
                ort_wasm_log_dict[ort_cur_model]["kernels"].append(json.loads(line.split("logEventInmediately")[1].strip()))
            elif "ORT_BEGIN_INFERENCE:webgl" in line or "ORT_FINISH_INFERENCE:webgl" in line:
                _model = line.split(":webgl")[1].strip().split()[0]
                if _model not in ort_webgl_model_list:
                    ort_webgl_model_list.append(_model)
                new_model_log_flag = True

            elif "Profiler.op" in line:
                if new_model_log_flag:
                    new_model_log_flag = False
                    ort_cur_model = ort_webgl_model_list[ort_webgl_cur_model_index]
                    ort_webgl_cur_model_index += 1
                if "kernels" not in ort_webgl_log_dict[ort_cur_model]:
                    ort_webgl_log_dict[ort_cur_model]["kernels"] = []
                ort_webgl_log_dict[ort_cur_model]["kernels"].append(line.split("|")[1].strip())

            elif "ORT_INFERENCE_BEGIN_MEMORY:wasm" in line:
                ort_wasm_log_dict[ort_cur_model]["inference_begin_memory"] = json.loads(line.split("ORT_INFERENCE_BEGIN_MEMORY:wasm")[1].strip())
            elif "ORT_INFERENCE_SESSION_MEMORY:wasm" in line:
                ort_wasm_log_dict[ort_cur_model]["session_setup_memory"] = json.loads(line.split("ORT_INFERENCE_SESSION_MEMORY:wasm")[1].strip())
            elif "ORT_INFERENCE_FINISH_MEMORY:wasm" in line:
                ort_wasm_log_dict[ort_cur_model]["inference_finish_memory"] = json.loads(line.split("ORT_INFERENCE_FINISH_MEMORY:wasm")[1].strip())
            elif "ORT_INFERENCE_BEGIN_MEMORY:webgl" in line:
                ort_webgl_mem_cache.append(["inference_begin_memory", json.loads(line.split("ORT_INFERENCE_BEGIN_MEMORY:webgl")[1].strip())])
            elif "ORT_INFERENCE_SESSION_MEMORY:webgl" in line:
                ort_webgl_mem_cache.append(["session_setup_memory", json.loads(line.split("ORT_INFERENCE_SESSION_MEMORY:webgl")[1].strip())])
            elif "ORT_INFERENCE_FINISH_MEMORY:webgl" in line:
                ort_webgl_mem_cache.append(["inference_finish_memory", json.loads(line.split("ORT_INFERENCE_FINISH_MEMORY:webgl")[1].strip())])

        for model in ort_webgl_model_list:
            for _ in range(3):
                k, v = ort_webgl_mem_cache.pop(0)
                ort_webgl_log_dict[model][k] = v

    with open('data/ort-wasm.json', 'w') as f:
        json.dump(ort_wasm_log_dict, f, indent=2)
    with open('data/ort-webgl.json', 'w') as f:
        json.dump(ort_webgl_log_dict, f, indent=2)


def plot_ort_wasm():
    with open('data/ort-wasm.json') as f:
        d = json.load(f)
    for modelname in d:
        kernel = []
        kernel_time = []
        bytes_add = []

        for op in d[modelname]["kernels"]:
            kernel.append(op["args"]["op_name"])
            kernel_time.append(op["dur"] / 1000)
            bytes_add.append(int(op["args"]["output_size"]))

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('inference-progress/ith op')
        ax1.set_ylabel('latency', color='red')
        ln1 = ax1.plot(np.cumsum(kernel_time), color='red', label='cum-lat')

        ax2 = ax1.twinx()
        ax2.set_ylabel('memory/MB', color='blue')
        ax2.hlines([d[modelname]["inference_begin_memory"]["totalJSHeapSize"] / 1024 ** 2,
                    d[modelname]["session_setup_memory"]["totalJSHeapSize"] / 1024 ** 2,
                    d[modelname]["inference_finish_memory"]["totalJSHeapSize"] / 1024 ** 2], 0, len(kernel), color=['blue', 'black', 'gray'])
        bytes_add[0] += d[modelname]["session_setup_memory"]["totalJSHeapSize"]
        ln2 = ax2.plot(np.cumsum(bytes_add) / 1024 ** 2,
                       color='green', label='cum-mem w/o gc')
        plt.legend(handles=ln1 + ln2)
        plt.title(modelname)
        plt.show()
plot_breakdown(f'esrgan-tf2_1.wasm.1.all.log')          


