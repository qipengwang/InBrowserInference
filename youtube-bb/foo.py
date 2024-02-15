import csv
import json


ytbbID2cocoLabel = {}
with open("yt_bb_classification_validation.csv", "r") as f:
    annots = list(csv.reader(f))
for annot in annots:
    ytbbID2cocoLabel[annot[2]] = annot[3]
with open("ytbbID2cocoLabel.json", 'w') as f:
    json.dump(ytbbID2cocoLabel, f, indent=2)
    