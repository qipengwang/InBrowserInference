import os, csv, json
import weakref
import cv2
import numpy as np
import argparse
from collections import  defaultdict
import multiprocessing as mp
from more_itertools import chunked

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0, choices=list(range(8)))
args = parser.parse_args()


def detect():
    import tensorflow as tf
    import tensorflow_hub as hub
    model = hub.KerasLayer("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    dir_list = []
    for file in os.listdir("frames/frame_images_DB"):
        if os.path.isdir(f"frames/frame_images_DB/{file}") and os.path.exists(f"frames/frame_images_DB/{file}.labeled_faces.txt"):
            dir_list.append(file)
    dir_list = sorted(dir_list)
    _len = len(dir_list) // 8
    if args.index == 0:
        dir_list = dir_list[:_len]
    elif args.index == 7:
        dir_list = dir_list[_len * 7:]
    else:
        dir_list = dir_list[_len * args.index : _len *(args.index + 1)]

    for idx, subdir in enumerate(dir_list):
        print(idx + 1, subdir)

        with open(f"frames/frame_images_DB/{subdir}.labeled_faces.txt") as f:
            annotations = list(csv.reader(f))
        
        all_detections = []
        for fn, _, center_x, center_y, width, height, _, _ in annotations:
            fn = fn.replace("\\", "/")
            gt_xmin = int(center_x) - int(width) / 2
            gt_ymin = int(center_y) - int(height) / 2
            gt_xmax = int(center_x) + int(width) / 2
            gt_ymax = int(center_y) + int(height) / 2
            img = cv2.imread(os.path.join("frames/frame_images_DB", fn))
            H, W, C = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(cv2.resize(img, (224, 224)), [1, 224, 224, 3])
            y = model(img)  
            # ['detection_boxes', 'detection_anchor_indices', 'detection_multiclass_scores', 'num_detections', 'raw_detection_scores', 'raw_detection_boxes', 'detection_classes', 'detection_scores']
            selected_box = y["detection_boxes"][tf.logical_and(y["detection_scores"] > 0.5, y["detection_classes"] == 1)]
            selected_box = (selected_box.numpy() * np.array([H, W, H, W])).astype(np.int32)
            if not selected_box.size:
                continue
            ixmin = np.maximum(selected_box[:, 1], gt_xmin)
            iymin = np.maximum(selected_box[:, 0], gt_ymin)
            ixmax = np.minimum(selected_box[:, 3], gt_xmax)
            iymax = np.minimum(selected_box[:, 2], gt_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # person covers face, so use the gt area as union
            # uni = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) + (selected_box[:, 2] - selected_box[:, 0] + 1.) * (selected_box[:, 3] - selected_box[:, 1] + 1.) - inters
            uni = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            ymin, xmin, ymax, xmax = selected_box[np.argmax(overlaps)]
            if ovmax == 1:
                all_detections.append([fn, xmin, ymin, xmax, ymax])
            
        with open(f"frames/{subdir}.detected.txt", 'w') as f:
            csv.writer(f).writerows(all_detections)
        print(f"finish [{idx + 1} / {len(dir_list)}]")



def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _analyze(subdirs, overlap_thres=0.9):
    for _, subdir in enumerate(subdirs):
        detections, annotations = {}, {}
        fns = []
        results = {}
        with open(f"results/inference/{subdir}.detected.txt") as f:
            for fn, xmin, ymin, xmax, ymax in csv.reader(f):
                detections[fn] = [xmin, ymin, xmax, ymax]
        if not detections: 
            continue
        with open(f"frames/frame_images_DB/{subdir}.labeled_faces.txt") as f:
            for fn, _, center_x, center_y, width, height, _, _ in csv.reader(f):
                fn = fn.replace("\\", "/")
                fns.append(fn)
                gt_xmin = int(center_x) - int(width) / 2
                gt_ymin = int(center_y) - int(height) / 2
                gt_xmax = int(center_x) + int(width) / 2
                gt_ymax = int(center_y) + int(height) / 2
                img = cv2.imread(os.path.join("frames/frame_images_DB", fn))
                H, W, C = img.shape
                annotations[fn] = [gt_xmin, gt_ymin, gt_xmax, gt_ymax]
        for inference_rate in np.arange(0.1, 1.1, 0.1):
            tp = np.zeros(len(fns))
            fp = np.zeros(len(fns))
            inferred_cnt = 0
            pre_res = None
            for idx, fn in enumerate(fns):
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = annotations[fn]
                xmin, ymin, xmax, ymax = None, None, None, None
                # intersection = max(ixmax - ixmin + 1.0, 0.0) * max(iymax - iymin + 1., 0.)
                # union = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) + (xmax - xmin + 1.0) * (ymax - ymin + 1.0) - intersection
                if fn in detections and 0 <= idx - inferred_cnt * inference_rate < 1:  # inference
                    xmin, ymin, xmax, ymax = gt_xmin, gt_ymin, gt_xmax, gt_ymax
                    inferred_cnt += 1
                elif pre_res:
                    xmin, ymin, xmax, ymax = pre_res
                else:
                    fp[idx] = 1
                    continue
                pre_res = [xmin, ymin, xmax, ymax]
                ixmin = max(xmin, gt_xmin)
                iymin = max(ymin, gt_ymin)
                ixmax = max(xmax, gt_xmax)
                iymax = max(ymax, gt_ymax)
                intersection = max(ixmax - ixmin + 1.0, 0.0) * max(iymax - iymin + 1., 0.)
                union = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) + (xmax - xmin + 1.0) * (ymax - ymin + 1.0) - intersection
                if intersection / union >= overlap_thres:
                    tp[idx] = 1
                else:
                    fp[idx] = 1
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(len(annotations))
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            results[inference_rate] = voc_ap(rec, prec)
        with open(f"results/analysis/{subdir}.analyzed.txt", "w") as f:
            json.dump(results, f, indent=2)        
        


def analyze_detect(overlap_thres = 0.9):
    detected_dir = [file.replace(".detected.txt", "") for file in os.listdir("results/inference/") if file.endswith("detected.txt")]
    if args.index == 0:
        detected_dirs = detected_dir[:len(detected_dir) // 2]
    elif args.index == 1:
        detected_dirs = detected_dir[len(detected_dir) // 2:]
    else:
        assert False, "invalid index"
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(_analyze, chunked(detected_dirs, 10))


def analyze_all():
    aps, weights = defaultdict(list), defaultdict(list)
    for file in os.listdir("results/analysis"):
        with open(f"results/analysis/{file}") as f:
            d = json.load(f)
        with open(f"frames/frame_images_DB/{file.replace('analyzed', 'labeled_faces')}") as f:
            weight = len(f.readlines())
            for k in d:
                aps[k].append(d[k])
                weights[k].append(weight)
    for k in aps:
        print(k, np.average(aps[k], weights=weights[k]))

# analyze_detect()
analyze_all()
