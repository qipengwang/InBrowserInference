import os
import sys

import tensorflow as tf
import numpy as np
import time
from pympler import asizeof
import tensorflow_hub as hub
import onnxruntime as ort

if len(tf.config.list_physical_devices('GPU')):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    print('error to use GPU, fallback to CPU')


def inference_deprecated():
    modelnames = [
        # ['Resnet50', (1, 224, 224, 3), 'float32'],
        ['Faster-RCNN-Resnet50', (1, 224, 224, 3), 'uint8'],
        ['SSD-Mobilenet', (1, 224, 224, 3), 'uint8'],
        ['EfficientDet', (1, 224, 224, 3), 'uint8'],
        ['MobileBert', (1, 5,), 'int32'],  # GG
        ['Albert', (1, 5,), 'int32'],  # GG
        ['1Conv', (1, 28, 28, 1), 'float32'],
        ['5Conv', (1, 28, 28, 1), 'float32']
    ]

    for modelname, inputsize, datatype in modelnames[-2:]:
        print(modelname, inputsize, datatype)

        model = tf.saved_model.load(f"./models#model/{modelname}")
        model_keras = hub.KerasLayer(f"./models#model/{modelname}")
        infer = model.signatures["serving_default"]
        random = np.random.randint(1, 10, size=np.product(inputsize))
        image = tf.constant(random, shape=inputsize, dtype=eval(f'tf.{datatype}'))
        output = infer(image)
        model_keras(image)

        tolEpoch = 10
        tolLatency = 0

        for i in range(tolEpoch):
            # flops = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(), options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())

            image = tf.constant(random, shape=inputsize, dtype=eval(f'tf.{datatype}'))
            start = time.time()
            output = infer(image)
            latency = (time.time() - start) * 1000
            tolLatency += latency
            # print(i, ":Inference Latency: ", latency, 'flops:', flops.total_float_ops)

        averageLatency = tolLatency / tolEpoch
        print(modelname, "Average Inference Latency: ", averageLatency, '\n')


def get_dir_size(dirpath):
    dirsize = 0
    for root, _, files in os.walk(dirpath):
        dirsize += sum([os.path.getsize(os.path.join(root, f)) for f in files])
    return dirsize


def tfhub_based():
    d = {
        'imagenet_mobilenet_v2_100_224_classification_5': [
            [(1, 224, 224, 3)], ['float32']#, ['inputs']
        ],
        'resnet_50_classification_1': [
            [(1, 224, 224, 3)], ['float32']#, ['input_1']
        ],
        'ssd_mobilenet_v2_2': [
            [(1, 224, 224, 3)], ['uint8']#, ['input_tensor']
        ],
        'faster_rcnn_resnet50_v1_1024x1024_1': [
            [(1, 224, 224, 3)], ['uint8']#, ['input_tensor']
        ],
        'edgetpu_vision_deeplab-edgetpu_default_argmax_s_1': [
            [(1, 512, 512, 3)], ['float32']#, ['input_2']
        ],
        'edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1': [
            [(1, 512, 512, 3)], ['float32']#, ['input_2']
        ],
        'edgetpu_nlp_mobilebert-edgetpu_xs_1': [
            [(1, 128), (1, 128), (1, 128)], ['int32', 'float32', 'int32'], #['input_word_ids', 'input_mask', 'input_type_ids']
        ],
        'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [
            [(1, 128), (1, 128), (1, 128)], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
        ],
        'albert_en_base_3': [
            [(1, 128), (1, 128), (1, 128)], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
        ],
        'electra_small_2': [
            [(1, 128), (1, 128), (1, 128)], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
        ],
        'esrgan-tf2_1': [
            [(1, 224, 224, 3)], ['float32']#, ['input_0']
        ],
        'movenet_multipose_lightning_1': [
            [(1, 224, 224, 3)], ['int32']#, ['input']
        ],

        # 'albert_lite_base_1': [],
        'experts_bert_pubmed_2': [
            [(1, 128), (1, 128), (1, 128)], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
        ],
        # 'german-tacotron2_1': [],
        # 'mil-nce_s3d_1': [],
        # 'mmv_s3d_1': [],
        # 'mobilebert_1': [],
        'movenet_singlepose_thunder_4': [
            [(1, 256, 256, 3)], ['int32']#, ['input']
        ],
        # 'sentence-t5_st5-base_1': [],
        # 'silero-stt_de_1': [
        #     [(1, 128)], ['float32'], ['input']
        # ],  # ValueError: The first argument to `Layer.call` must always be passed.
        'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [
            [(1, 128), (1, 128), (1, 128)], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
        ],
        'trillsson2_1': [
            [(1, 128)], ['float32']#, ['audio_samples']
        ],
        # 'universal-sentence-encoder-lite_2': [],
        # 'wav2vec2_1': [
        #     [(1, 246000)], ['float32']#, ['input_1']
        # ],  # Fused conv implementation does not support grouped convolutions for now
        # 'wav2vec2-960h_1': [
        #     [(1, 246000)], ['float32']#, ['input_1']
        # ],  # Fused conv implementation does not support grouped convolutions for now
    }
    num_iter = 100
    # tf.config.threading.set_inter_op_parallelism_threads(4)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    for modelname in d:
        print(modelname)
        start_t = time.time()
        if modelname == 'movenet_multipose_lightning_1' or modelname == 'movenet_singlepose_thunder_4':
            model = hub.KerasLayer(f"./models/savedModel/{modelname}", signature='serving_default', signature_outputs_as_dict=True)
        else:
            model = hub.KerasLayer(f"./models/savedModel/{modelname}")
        setup_latency = time.time() - start_t
        input_shapes, input_types = d[modelname][0], d[modelname][1]
        inputs = []
        for in_shape, in_type in zip(input_shapes, input_types):
            inputs.append(tf.constant(np.random.randint(0, 2, size=np.product(in_shape)), shape=in_shape, dtype=eval(f'tf.{in_type}')))
        if len(d[modelname]) > 2:
            inputs = {
                k: v for k, v in zip(d[modelname][2], inputs)
            }
        start_t = time.time()
        if len(inputs) == 1 and isinstance(inputs, list):
            model(*inputs)
        else:
            model(inputs)
        cold_latency = time.time() - start_t
        tot_latency = []
        for _ in range(num_iter):
            start_t = time.time()
            if len(inputs) == 1:
                model(*inputs)
            else:
                model(inputs)
            tot_latency.append(time.time() - start_t)
        print(f'model = {modelname}:')
        print(f'\tsize: asizeof/dir-size = {asizeof.asizeof(model)}/{get_dir_size(f"./models/savedModel/{modelname}")},'
              f'\tjs-dir-size = {get_dir_size(f"./models/jsModel/{modelname}")},'
              f'\tonnx-file-size = {os.path.getsize(f"./models/onnxModel/{modelname}.onnx")}')
        print(f'\tsetup_latency = {setup_latency * 1000},\tcold_latency = {cold_latency*1000},\tinfer_latency = {np.mean(tot_latency) * 1000}')


def e2e_tf():
    num_iter = 1
    D_tfjs = {
        'imagenet_mobilenet_v2_100_224_classification_5': [  #wasm-ok, webgpu-ok
            [[1, 224, 224, 3]], ['float32']
        ],
        'resnet_50_classification_1': [   #wasm-ok, webgpu-ok
            [[1, 224, 224, 3]], ['float32']
        ],
        'ssd_mobilenet_v2_2': [   #wasm-ok
            [[1, 224, 224, 3]], ['uint8']
        ],
        'edgetpu_nlp_mobilebert-edgetpu_xs_1': [   #wasm-fail
            [[1, 10], [1, 10], [1, 10]], [ 'int32','float32', 'int32']
        ],
        'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [   #wasm-fail
            [[1, 10], [1, 10], [1, 10]], ['int32', 'int32', 'int32'], ['input_word_ids', 'input_type_ids', 'input_mask']
        ],
        'esrgan-tf2_1': [   #wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32']#, ['input_0']
        ],
        'movenet_multipose_lightning_1': [   #wasm-fail, webgl-ok
            [[1, 224, 224, 3]], ['int32']#, ['input']
        ],
        'movenet_singlepose_thunder_4': [   #wasm-ok, webgl-ok
            [[1, 256, 256, 3]], ['int32']#, ['input']
        ],
        'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [   #wasm-fail, webgl-ok
            [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
        ],
        'VGG16': [  # wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32'] #, ['input_1']  
        ],
        'ShuffleNetV2': [  # wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32'] #, ['input_1']
        ],
        'yolov5': [  # wasm-fail(Kernel 'Softplus' not registered for backend 'wasm'.), webgl-ok
            [[1, 224, 224, 3]], ['float32'] #, ['input_1']
        ],
        'Xception': [  # wasm-ok, webgl-ok
            [[1, 299, 299, 3]], ['float32'] #, ['input_3']
        ],
        'InceptionV3': [  # wasm-ok, webgl-ok
            [[1, 299, 299, 3]], ['float32'] #, ['input_1']
        ],
        'EfficientNetV2': [  # wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32'] #, ['input_1']
        ],
        'efficientdet_d1_coco17_tpu-32': [  # wasm-fail(Error: Kernel 'Reciprocal' not registered for backend 'wasm'.), webgl-fail
            [[1, 640, 640, 3]], ['uint8'] #, ['input_tensor']
        ],
        'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  # wasm-fail(Error: Kernel 'Reciprocal' not registered for backend 'wasm'.), webgl-ok
            [[1, 512, 512, 3]], ['uint8'] #, ['input_tensor']  
        ],
    }

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    for modelname in D_tfjs: #["resnet_50_classification_1"]:
        print(modelname)
        start_t = time.time()
        if modelname == 'movenet_multipose_lightning_1' or modelname == 'movenet_singlepose_thunder_4':
            model = hub.KerasLayer(f"./models/savedModel/{modelname}", signature='serving_default', signature_outputs_as_dict=True)
        else:
            model = hub.KerasLayer(f"./models/savedModel/{modelname}")
        setup_latency = time.time() - start_t
        input_shapes, input_types = D_tfjs[modelname][0], D_tfjs[modelname][1]
        inputs = []
        for in_shape, in_type in zip(input_shapes, input_types):
            inputs.append(tf.constant(np.random.randint(0, 2, size=np.product(in_shape)), shape=in_shape, dtype=eval(f'tf.{in_type}')))
        if len(D_tfjs[modelname]) > 2:
            inputs = {
                k: v for k, v in zip(D_tfjs[modelname][2], inputs)
            }
        start_t = time.time()
        if len(inputs) == 1 and isinstance(inputs, list):
            model(*inputs)
        else:
            model(inputs)
        cold_latency = time.time() - start_t
        tot_latency = []
        for _ in range(num_iter):
            start_t = time.time()
            if len(inputs) == 1:
                model(*inputs)
            else:
                model(inputs)
            tot_latency.append(time.time() - start_t)
        print(f'model = {modelname}:')
        print(f'\tsetup_latency = {setup_latency * 1000},\tcold_latency = {cold_latency*1000},\tinfer_latency = {np.mean(tot_latency) * 1000}')
        print(f"[& {1000 * setup_latency:<26.1f}& {1000 * cold_latency:<31.1f}& {np.mean(tot_latency) * 1000:<31.1f}]")
        # input("input:")

def e2e_onnx():
    num_iter = 5
    D_ort = {
        'imagenet_mobilenet_v2_100_224_classification_5': [  # wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32'], ['inputs'], ['logits']
        ],
        'resnet_50_classification_1': [  # wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32'], ['input_1'], ["activation_49"]
        ],
        'ssd_mobilenet_v2_2': [  # wasm-ok, webgl-fail
            [[1, 224, 224, 3]], ['uint8'], ['input_tensor'], [
                'detection_anchor_indices', 'detection_boxes', 'detection_classes', 'detection_multiclass_scores', 
                'detection_scores', 'num_detections', 'raw_detection_boxes', 'raw_detection_scores'
            ]
        ],
        'edgetpu_nlp_mobilebert-edgetpu_xs_1': [  # wasm-ok, webgl-fail
            [[1, 128], [1, 128], [1, 128]], ['float32', 'int32',  'int32'], ['input_mask', 'input_type_ids', 'input_word_ids'], [
                "tf.compat.v1.squeeze",
                "transformer_layer_0", "transformer_layer_0_1",
                "transformer_layer_1", "transformer_layer_1_1",
                "transformer_layer_2", "transformer_layer_2_1",
                "transformer_layer_3", "transformer_layer_3_1",
                "transformer_layer_4", "transformer_layer_4_1",
                "transformer_layer_5", "transformer_layer_5_1",
                "transformer_layer_6", "transformer_layer_6_1",
                "transformer_layer_7", "transformer_layer_7_1", "transformer_layer_7_2",
            ]
        ],
        'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [  # wasm-ok, webgl-fail
            [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids'],[
                'mobile_bert_encoder',
                'mobile_bert_encoder_1', 'mobile_bert_encoder_10', 'mobile_bert_encoder_11', 'mobile_bert_encoder_12',
                'mobile_bert_encoder_13', 'mobile_bert_encoder_14', 'mobile_bert_encoder_15', 'mobile_bert_encoder_16',
                'mobile_bert_encoder_17', 'mobile_bert_encoder_18', 'mobile_bert_encoder_19', 'mobile_bert_encoder_2',
                'mobile_bert_encoder_20', 'mobile_bert_encoder_21', 'mobile_bert_encoder_22',
                'mobile_bert_encoder_23', 'mobile_bert_encoder_24', 'mobile_bert_encoder_25', 'mobile_bert_encoder_26',
                'mobile_bert_encoder_27', 'mobile_bert_encoder_28', 'mobile_bert_encoder_29', 'mobile_bert_encoder_3', 
                'mobile_bert_encoder_30', 'mobile_bert_encoder_31', 'mobile_bert_encoder_32',
                'mobile_bert_encoder_33', 'mobile_bert_encoder_34', 'mobile_bert_encoder_35', 'mobile_bert_encoder_36',
                'mobile_bert_encoder_37', 'mobile_bert_encoder_38', 'mobile_bert_encoder_39', 'mobile_bert_encoder_4', 
                'mobile_bert_encoder_40', 'mobile_bert_encoder_41', 'mobile_bert_encoder_42',
                'mobile_bert_encoder_43', 'mobile_bert_encoder_44', 'mobile_bert_encoder_45', 'mobile_bert_encoder_46',
                'mobile_bert_encoder_47', 'mobile_bert_encoder_48', 'mobile_bert_encoder_49', 'mobile_bert_encoder_5', 
                'mobile_bert_encoder_50', 'mobile_bert_encoder_51', 'mobile_bert_encoder_6',
                'mobile_bert_encoder_7', 'mobile_bert_encoder_8', 'mobile_bert_encoder_9',
            ]
        ],
        'esrgan-tf2_1': [  # wasm-ok, webgl-fail
            [[1, 224, 224, 3]], ['float32'], ['input_0']
        ],
        'movenet_multipose_lightning_1': [  # wasm-ok, webgl-fail
            [[1, 224, 224, 3]], ['int32'], ['input'], ["output_0"]
        ],
        'movenet_singlepose_thunder_4': [  # wasm-ok, webgl-fail
            [[1, 256, 256, 3]], ['int32'], ['input'], ["output_0"]
        ],
        'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [  # wasm-ok, webgl-fail
            [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids'], [
                "bert_encoder", "bert_encoder_1", "bert_encoder_2", "bert_encoder_3", "bert_encoder_4", 
            ]
        ],
        'VGG16': [  # wasm-ok, webgl-ok(Source data too small. Allocating larger array)
            [[1, 224, 224, 3]], ['float32'], ['input_1'], ["predictions"]
        ],
        'ShuffleNetV2': [  # wasm-ok, webgl-fail
            [[1, 224, 224, 3]], ['float32'], ['input_5'], ["dense"]
        ],
        'yolov5': [  # wasm-ok, webgl-fail
            [[1, 224, 224, 3]], ['float32'], ['input_1'] , ["detect", "detect_1", "detect_2", ]
        ],
        'Xception': [  # wasm-ok, webgl-ok
            [[1, 299, 299, 3]], ['float32'], ['input_4'], ["predictions"]
        ],
        'InceptionV3': [  # wasm-ok, webgl-ok
            [[1, 299, 299, 3]], ['float32'], ['input_2'], ["predictions"]
        ],
        'EfficientNetV2': [  # wasm-ok, webgl-ok
            [[1, 224, 224, 3]], ['float32'], ['input_3'], ["predictions"]
        ],
        'efficientdet_d1_coco17_tpu-32': [  # wasm-ok, webgl-fail
            [[1, 640, 640, 3]], ['uint8'], ['input_tensor'], [
                'detection_anchor_indices', 'detection_boxes', 'detection_classes',
                'detection_multiclass_scores', 'detection_scores', 'num_detections',
            ]
        ],
        'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  # wasm-ok, webgl-fail
            [[1, 512, 512, 3]], ['uint8'], ['input_tensor'], ["detection_boxes", "detection_classes", "detection_scores", "num_detections"]
        ],
    }
    for modelname in D_ort:
        print(modelname)
        start_t = time.time()
        session = ort.InferenceSession(f"models/onnxModel/{modelname}.onnx", providers=["CPUExecutionProvider"])
        setup_latency = time.time() - start_t
        
        input_shapes = D_ort[modelname][0]
        input_types = D_ort[modelname][1]
        input_names = D_ort[modelname][2]
        inputData, feed = None, {}
        for i in range(len(input_shapes)) :
            # feed[input_names[i]] = ort.OrtValue.ortvalue_from_shape_and_type(input_shapes[i], input_types[i])
            feed[input_names[i]] = ort.OrtValue.ortvalue_from_numpy(np.random.randint(low=-1, high=2, size=input_shapes[i]).reshape(input_shapes[i]).astype(eval(f"np.{input_types[i]}")))
        start_t = time.time()
        outputs = session.run(D_ort[modelname][3], feed)
        cold_latency = time.time() - start_t
        tot_latency = []
        for _ in range(num_iter):
            start_t = time.time()
            outputs = session.run(D_ort[modelname][3], feed)
            tot_latency.append(time.time() - start_t)
        print(f'model = {modelname}:')
        inference_latency = 1000 * np.mean(tot_latency)
        print(f'\tsetup_latency = {setup_latency * 1000},\tcold_latency = {cold_latency*1000},\tinfer_latency = {inference_latency}')
        print(f" {1000 * setup_latency:<26.1f}& {1000 * cold_latency:<31.1f}& {inference_latency:<31.1f}")


e2e_tf()
e2e_onnx()