import tensorflow as tf
import os, json
import cv2
import csv
import numpy as np
import sys

# with open("yt_bb_classification_validation.csv", "r") as f:
#     annots = list(csv.reader(f))
# dumped_annots = [annot for annot in annots if os.path.exists(f"frames/yt_bb_classification_validation/{annot[2]}/{annot[0]}+{annot[1]}.jpg")]

# with open(f"frames/yt_bb_classification_validation/annotations.csv", 'w') as f:
#     csv.writer(f).writerows(dumped_annots)



def ideal_case_inference():
    inference_results = []
    model = tf.keras.applications.mobilenet_v2.MobileNetV2()
    for class_id in os.listdir("frames/yt_bb_classification_validation"):
        if not os.path.isdir(f"frames/yt_bb_classification_validation/{class_id}"):
            continue
        frame_dir = f"frames/yt_bb_classification_validation/{class_id}"
        for fn in os.listdir(frame_dir):
            img = cv2.imread(os.path.join(frame_dir, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(img.shape)
            img = np.reshape(cv2.resize(img, (224, 224)), [1, 224, 224, 3]) / 255.0
            # print(type(img), img.shape)
            x = tf.convert_to_tensor(img)
            y = tf.argmax(model(x), axis=1)[0]
            inference_results.append([class_id, fn, y])
    with open(f"inference_ideal.csv", 'w') as f:
        csv.writer(f).writerows(inference_results)

def statistic_ideal():
    with open("inference_ideal.csv") as f:
        results = list(csv.reader(f))
    imagenet2coco = {}
    with open("imagenet2coco.txt") as f:
        for line in f:
            line = line.strip().split("\t")
            imagenet2coco[line[0]] = line[1]
    with open("ytbbID2cocoLabel.json") as f:
        ytbbID2cocoLabel = json.load(f)
    with open("imagenet_label_dict.json") as f:
        imagenet_id2label = json.load(f)
    correct = 0
    for ytbb_id, fn, infer_id in results:
        infered_label = imagenet2coco[imagenet_id2label[infer_id]]
        ytbb_label = ytbbID2cocoLabel[ytbb_id]
        correct += infered_label == ytbb_label
    print(correct, correct / len(results))

def statistic_qoe(inference_fps = 24, ideal_fps = 30):
    with open("inference_ideal.csv") as f:
        results = list(csv.reader(f))
    inference_result = {result[1]: result[2] for result in results}
    with open("yt_bb_classification_validation_dumped_annotations.csv") as f:
        annots = list(csv.reader(f))
    imagenet2coco = {}
    with open("imagenet2coco.txt") as f:
        for line in f:
            line = line.strip().split("\t")
            imagenet2coco[line[0]] = line[1]
    with open("ytbbID2cocoLabel.json") as f:
        ytbbID2cocoLabel = json.load(f)
    with open("imagenet_label_dict.json") as f:
        imagenet_id2label = json.load(f)
    inference_rate = ideal_fps / inference_fps
    inferred_cnt = 0
    correct = 0
    pre_res = None
    for idx, annot in enumerate(annots):
        inference_res = None
        # print(idx, inferred_cnt, inference_rate)
        if 0 <= idx - inferred_cnt * inference_rate < 1:
            # print("inference")
            inferred_cnt += 1
            inference_res = inference_result[f"{annot[0]}+{annot[1]}.jpg"]
        else:
            # print("reuse")
            # print(annot, inference_result[f"{annot[0]}+{annot[1]}.jpg"], pre_res)
            inference_res = pre_res
        pre_res = inference_res
        correct += imagenet2coco[imagenet_id2label[inference_res]] == ytbbID2cocoLabel[annot[2]]
    print(correct, correct / len(results))


ideal_case_inference()
statistic_ideal()

for fps in [1/1.3, 1/1.1, 1/0.13, 23, 24, 27.2, 26.4]:
    print(fps)
    statistic_qoe(fps)


        

            

        
