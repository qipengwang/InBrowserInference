// const ort = require('onnxruntime-node');

let D_ort = {
    'imagenet_mobilenet_v2_100_224_classification_5': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['inputs']
    ],
    'resnet_50_classification_1': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['input_1']
    ],
    'ssd_mobilenet_v2_2': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['uint8'], ['input_tensor']
    ],
    'faster_rcnn_resnet50_v1_1024x1024_1': [  // wasm-GG, webgl-GG
        [[1, 224, 224, 3]], ['uint8'], ['input_tensor']
    ],
    'edgetpu_nlp_mobilebert-edgetpu_xs_1': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['float32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'edgetpu_vision_deeplab-edgetpu_default_argmax_s_1': [  // wasm-ok, webgl-GG
        [[1, 512, 512, 3]], ['float32'], ['input_2']
    ],
    'edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1': [  // wasm-ok, webgl-GG
        [[1, 512, 512, 3]], ['float32'], ['input_2']
    ],
    'albert_en_base_3': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'electra_small_2': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'esrgan-tf2_1': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['float32'], ['input_0']
    ],
    'movenet_multipose_lightning_1': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['int32'], ['input']
    ],
    
    'experts_bert_pubmed_2': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'movenet_singlepose_thunder_4': [  // wasm-ok, webgl-GG
        [[1, 256, 256, 3]], ['int32'], ['input']
    ],
    'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'language_model': [  // wasm-ok, webgl-GG
        [[1, 128]], ['float32'], ['embedding_input']
    ],
    'VGG16': [  // wasm-ok, webgl-ok(Source data too small. Allocating larger array)
        [[1, 224, 224, 3]], ['float32'], ['input_1'] 
    ],
    'ShuffleNetV2': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['float32'], ['input_1']
    ],
    'yolov5': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['float32'], ['input_1']  
    ],
    'Xception': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'], ['input_3']
    ],
    'InceptionV3': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'], ['input_1']
    ],
    'EfficientNetV2': [  // wasm-ok, webgl-ok
        [[1, 480, 480, 3]], ['float32'], ['input_2']
    ],
    'efficientdet_d1_coco17_tpu-32': [  // wasm-ok, webgl-GG
        [[1, 640, 640, 3]], ['uint8'], ['input_tensor'] 
    ],
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  // wasm-ok, webgl-GG
        [[1, 512, 512, 3]], ['uint8'], ['input_tensor'] 
    ],
}


async function main(provider, modelnames) {
    console.log("begining");
    
    let tolEpoch = 1;

    for (let modelname of modelnames) {
        let modelSetupLatency = 0;
        let coldStartLatency = 0;
        let inferenceLatency = 0;
        let modelPath = "./models/onnxModel/" + modelname + ".onnx";
        console.log(modelname);
        try {
            // Model Load & Setup
            let start_t = performance.now();
            const session = await ort.InferenceSession.create(modelPath, {
                executionProviders: [provider]
            });
            modelSetupLatency = performance.now() - start_t;

            let input_shapes = D_ort[modelname][0], input_types = D_ort[modelname][1], input_names = D_ort[modelname][2];
            let inputData, feed = {};
            for (let i = 0; i < input_shapes.length; i++) {
                if (input_types[i] === "float32") {
                    inputData = new Float32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "int32") {
                    inputData = new Int32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "uint8") {
                    inputData = new Uint8Array(input_shapes[i].reduce((a, b) => a * b));
                }
                feed[input_names[i]] = new ort.Tensor(input_types[i], inputData, input_shapes[i]);
            }
            // if (d[modelname].length > 2) {
            //     for (let i=0; i<input_shapes.length; i++) {
            //         inputs[d[modelname][2][i]] = feed[i];
            //     }
            // } else {
            //     inputs = feed;
            // }
            start_t = performance.now();
            session.startProfiling();
            await session.run(feed);
            session.endProfiling();
            coldStartLatency = performance.now() - start_t;

            // feed inputs and run
            start_t = performance.now();
            session.startProfiling();
            for (let epoch = 0; epoch < tolEpoch; epoch++) {
                await session.run(feed);
            }
            session.endProfiling();
            inferenceLatency = (performance.now() - start_t) / tolEpoch;
        } catch (e) {
            console.log(`failed to inference model: ${e}.`, e.stack);
        }

        let log = {
            ModelSetup: modelSetupLatency,
            CodeStart: coldStartLatency,
            InferenceLatency: inferenceLatency
        }
        console.log("finish", modelname, D_ort[modelname], log);
    }
}

async function entry_func() {
    await main("webgl", [
        "imagenet_mobilenet_v2_100_224_classification_5", 
        "resnet_50_classification_1", 
    ]);
    console.log("finish all");
}

entry_func()

/*
                 v Profiler.op 2022-08-18T10:05:14.749Z|0.29ms on event 'ProgramManager.run mul_' at 0.29
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.750Z|0.20ms on event 'ProgramManager.run sub_' at 0.20
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.750Z|0.23ms on event 'ProgramManager.run Transpose' at 0.23
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.750Z|0.42ms on event 'ProgramManager.run Im2Col' at 0.42
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.750Z|1.99ms on event 'ProgramManager.run ConvDotProduct' at 1.99
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.750Z|2.19ms on event 'ProgramManager.run GroupedConv' at 2.19
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.751Z|0.44ms on event 'ProgramManager.run Im2Col' at 0.44
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.751Z|1.15ms on event 'ProgramManager.run ConvDotProduct' at 1.15
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.751Z|0.25ms on event 'ProgramManager.run Im2Col' at 0.25
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.751Z|2.94ms on event 'ProgramManager.run ConvDotProduct' at 2.94
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.752Z|1.38ms on event 'ProgramManager.run BatchNormalization' at 1.38
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.752Z|0.46ms on event 'ProgramManager.run clip' at 0.46
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.752Z|1.83ms on event 'ProgramManager.run GroupedConv' at 1.83
instrument.ts:98 v Profiler.op 2022-08-18T10:05:14.752Z|0.32ms on event 'ProgramManager.run Im2Col' at 0.32
*/