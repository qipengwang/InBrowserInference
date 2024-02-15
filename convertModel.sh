#!/bin/bash

source /data/v-qipengwang/Software/anaconda3/etc/profile.d/conda.sh
conda activate webai

models=(  \
	"VGG16"  \  # ok
	"ShuffleNetV2"  \  # GG
	"yolov5"  \  # ok
	"Xception"  \  # ok
	"InceptionV3"  \  # ok
	"EfficientNetV2"  \  # ok
	"efficientdet_d1_coco17_tpu-32"  \  # ok
	"centernet_resnet50_v1_fpn_512x512_coco17_tpu-8"  \  # ok
)

for model in ${models[@]}; do 
    # if [[ -f "models/onnxModel/${model}.onnx" ]]; then
    #     continue
    # fi
	echo $model
	saved_model_cli show --dir  models/savedModel/${model} --all
	# du -sh models/savedModel/${model}
	# du -sh models/jsModel/${model}
	echo "\n"
	# tensorflowjs_converter \
    #     --input_format=tf_hub \
    #     --signature_name=serving_default \
	#     models/savedModel/${model} models/jsModel/${model} 
    
    # python -m tf2onnx.convert \
	# 	--saved-model models/savedModel/${model} \
	# 	--opset 15 \
	# 	--output models/onnxModel/${model}.onnx
	# echo "\n\n\n\n\n\n";
done