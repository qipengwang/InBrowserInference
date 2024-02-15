model2numKernel = {
    'imagenet_mobilenet_v2_100_224_classification_5': 122,
    'resnet_50_classification_1': 150,
    'ssd_mobilenet_v2_2': 2410,
    'movenet_singlepose_thunder_4': 220,
    'movenet_multipose_lightning_1': 375,
    'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': 1652,
    'ShuffleNetV2': 165,
    'Xception': 149,
    'InceptionV3': 229,
    'yolov5': 508,
    'EfficientNetV2': 430,
    'efficientdet_d1_coco17_tpu-32': 3351,
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': 340,
    'small_bert_bert_en_uncased_L-2_H-128_A-2_2': 153
}

def processed_op_name(op: str):
    OP_SAME_NAME = {
        # for tfjs
        "_FusedMatMul": "FusedMatMul",
        "fusedConv2d__op": "FusedConv2D",
        "fusedDepthwiseConv2d__op": "FusedDepthwiseConv2D",
        "fusedMatMul__op": "FusedMatMul",
        # for ort
        "add_": "Add",
        "mul_": "Mul",
        "relu": "Relu",
        "sigmoid": "Sigmoid",
        "sub_": "Sub"
    }
    if op in OP_SAME_NAME:
        return OP_SAME_NAME[op]
    return op