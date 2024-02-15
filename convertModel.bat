tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/albert_en_base_3 models/jsModel/albert_en_base_3
python -m tf2onnx.convert --saved-model models/savedModel/albert_en_base_3 --opset 15 --output models/onnxModel/albert_en_base_3.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/esrgan-tf2_1 models/jsModel/esrgan-tf2_1
python -m tf2onnx.convert --saved-model models/savedModel/esrgan-tf2_1 --opset 15 --output models/onnxModel/esrgan-tf2_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/electra_small_2 models/jsModel/electra_small_2
python -m tf2onnx.convert --saved-model models/savedModel/electra_small_2 --opset 15 --output models/onnxModel/electra_small_2.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/edgetpu_nlp_mobilebert-edgetpu_xs_1 models/jsModel/edgetpu_nlp_mobilebert-edgetpu_xs_1
python -m tf2onnx.convert --saved-model models/savedModel/edgetpu_nlp_mobilebert-edgetpu_xs_1 --opset 15 --output models/onnxModel/edgetpu_nlp_mobilebert-edgetpu_xs_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1 models/jsModel/edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1
python -m tf2onnx.convert --saved-model models/savedModel/edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1 --opset 15 --output models/onnxModel/edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/edgetpu_vision_deeplab-edgetpu_default_argmax_s_1 models/jsModel/edgetpu_vision_deeplab-edgetpu_default_argmax_s_1
python -m tf2onnx.convert --saved-model models/savedModel/edgetpu_vision_deeplab-edgetpu_default_argmax_s_1 --opset 15 --output models/onnxModel/edgetpu_vision_deeplab-edgetpu_default_argmax_s_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/movenet_multipose_lightning_1 models/jsModel/movenet_multipose_lightning_1
python -m tf2onnx.convert --saved-model models/savedModel/movenet_multipose_lightning_1 --opset 15 --output models/onnxModel/movenet_multipose_lightning_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/imagenet_mobilenet_v2_100_224_classification_5 models/jsModel/imagenet_mobilenet_v2_100_224_classification_5
python -m tf2onnx.convert --saved-model models/savedModel/imagenet_mobilenet_v2_100_224_classification_5 --opset 15 --output models/onnxModel/imagenet_mobilenet_v2_100_224_classification_5.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/faster_rcnn_resnet50_v1_1024x1024_1 models/jsModel/faster_rcnn_resnet50_v1_1024x1024_1
python -m tf2onnx.convert --saved-model models/savedModel/faster_rcnn_resnet50_v1_1024x1024_1 --opset 15 --output models/onnxModel/faster_rcnn_resnet50_v1_1024x1024_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/resnet_50_classification_1 models/jsModel/resnet_50_classification_1
python -m tf2onnx.convert --saved-model models/savedModel/resnet_50_classification_1 --opset 15 --output models/onnxModel/resnet_50_classification_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1 models/jsModel/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1
python -m tf2onnx.convert --saved-model models/savedModel/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1 --opset 15 --output models/onnxModel/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1.onnx
tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/ssd_mobilenet_v2_2 models/jsModel/ssd_mobilenet_v2_2
python -m tf2onnx.convert --saved-model models/savedModel/ssd_mobilenet_v2_2 --opset 15 --output models/onnxModel/ssd_mobilenet_v2_2.onnx
