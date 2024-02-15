
from collections import defaultdict
import csv
import os
import multiprocessing as mp
from more_itertools import chunked
import cv2

def _download(cmds):
    print(len(cmds))
    for cmd in cmds:
        os.system(cmd)


def download():
    with open("yt_bb_classification_validation.csv", "r") as f:
        annots = list(csv.reader(f))
    yt_ids = set()
    for annot in annots:
        yt_ids.add(annot[0])
        
    commands = []
    for id in yt_ids:
        if not os.path.exists(f"videos/yt_bb_classification_validation/{id}_temp.mp4"):
            commands.append(f"youtube-dl -o videos/yt_bb_classification_validation/{id}.mp4 -f best[ext=mp4] youtu.be/{id}")
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(_download, list(chunked(commands, mp.cpu_count())))

def _extract(yt_ids):
    with open("yt_bb_classification_validation.csv", "r") as f:
        annots = list(csv.reader(f))
    annot_dict = defaultdict(list)
    for annot in annots:
        annot_dict[annot[0]].append([annot[1], annot[2]])
    for yt_id in yt_ids:
        cap = cv2.VideoCapture(f"videos/yt_bb_classification_validation/{yt_id}.mp4")
        for timestamp, class_id in annot_dict[yt_id]:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(timestamp))
            successful, frame = cap.read()
            if successful:
                cv2.imwrite(f"frames/yt_bb_classification_validation/{class_id}/{yt_id}+{timestamp}.jpg", frame)
            




def extract_frame():
    yt_ids = []
    for i in range(23):
        os.makedirs(f"frames/yt_bb_classification_validation/{i}/", exist_ok=True)
    for file in os.listdir(f"videos/yt_bb_classification_validation/"):
        if file.endswith(".mp4"):
            yt_ids.append(file.replace(".mp4", ""))
    
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(_extract, chunked(yt_ids, len(yt_ids) // mp.cpu_count()))

if __name__ == "__main__":
    extract_frame()