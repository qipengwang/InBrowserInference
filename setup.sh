#!/bin/bash

workdir=`pwd`

mkdir -p models dist data
mkdir -p models/jsModel models/onnxModel models/savedModel models/zippedModel/

wget https://storage.googleapis.com/tfhub-modules/google/imagenet/mobilenet_v2_100_224/classification/5.tar.gz -O models/zippedModel/imagenet_mobilenet_v2_100_224_classification_5.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/tensorflow/resnet_50/classification/1.tar.gz -O models/zippedModel/resnet_50_classification_1.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/tensorflow/ssd_mobilenet_v2/2.tar.gz -O models/zippedModel/ssd_mobilenet_v2_2.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/google/movenet/singlepose/thunder/4.tar.gz -O models/zippedModel/movenet_singlepose_thunder_4.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/google/movenet/multipose/lightning/1.tar.gz -O models/zippedModel/movenet_multipose_lightning_1.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1.tar.gz -O models/zippedModel/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/google/edgetpu/nlp/mobilebert-edgetpu/xs/1.tar.gz -O models/zippedModel/edgetpu_nlp_mobilebert-edgetpu_xs_1.tar.gz > /dev/null
wget https://storage.googleapis.com/tfhub-modules/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2.tar.gz -O models/zippedModel/small_bert_bert_en_uncased_L-2_H-128_A-2_2.tar.gz > /dev/null
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz -O models/zippedModel/efficientdet_d1_coco17_tpu-32.tar.gz > /dev/null
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz -O models/zippedModel/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz  > /dev/null

echo "finish download model from model zoo"


for file in `ls models/zippedModel`; do
    model=`basename $file .tar.gz`
    echo $model
    mkdir -p models/savedModel/${model}
    tar -zxvf models/zippedModel/$file -C models/savedModel/${model}/ > /dev/null
done

for model in "efficientdet_d1_coco17_tpu-32" "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8"; do
    mv models/savedModel/${model}/${model}/saved_model/* models/savedModel/${model}/
    rm -rf models/savedModel/${model}/${model}/
done

python build_model.py  > /dev/null
cd Yolov5/yolo && python train.py > /dev/null && cd ../..
cp -r Yolov5/weights/yolov5/ models/savedModel/ > /dev/null

echo "finish build model from source"

for model in `ls models/savedModel`; do
    tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/${model} models/jsModel/${model}  > /dev/null
    python -m tf2onnx.convert --saved-model models/savedModel/${model} --opset 15 --output models/onnxModel/${model}.onnx > /dev/null
done