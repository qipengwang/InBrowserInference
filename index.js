// const tf = require("@tensorflow/tfjs")
// const tfconv = require('@tensorflow/tfjs-converter')

console.log("first line of file: ");

tf.ENV.registerFlag('WASM_HAS_MULTITHREAD_SUPPORT');
tf.ENV.set('WASM_HAS_MULTITHREAD_SUPPORT', true);
tf.wasm.setThreadsCount(4);
tf.ENV.registerFlag('WASM_HAS_SIMD_SUPPORT');
tf.ENV.set('WASM_HAS_SIMD_SUPPORT', true);

var PROFILING_RESULT = {};
var tolEpoch = 10;
window.fps_arr=[];
window.fps_rounder = [];
var stop_flag = false;
var server_url = ".";

let D_tfjs = {
    'imagenet_mobilenet_v2_100_224_classification_5': [  //wasm-ok, webgpu-ok
        [[1, 224, 224, 3]], ['float32']
    ],
    'resnet_50_classification_1': [   //wasm-ok, webgpu-ok
        [[1, 224, 224, 3]], ['float32']
    ],
    'ssd_mobilenet_v2_2': [   //wasm-ok
        [[1, 224, 224, 3]], ['int32']
    ],
    'faster_rcnn_resnet50_v1_1024x1024_1': [   //wasm-fail
        [[1, 224, 224, 3]], ['int32']
    ],
    'edgetpu_nlp_mobilebert-edgetpu_xs_1': [   //wasm-fail
        [[1, 10], [1, 10], [1, 10]], ['int32', 'float32', 'int32']
    ],
    'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [   //wasm-fail
        [[1, 10], [1, 10], [1, 10]], ['int32', 'int32', 'int32'], ['input_word_ids', 'input_type_ids', 'input_mask']
    ],
    'albert_en_base_3': [   //wasm-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'electra_small_2': [   //wasm-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'esrgan-tf2_1': [   //wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32']//, ['input_0']
    ],
    'movenet_multipose_lightning_1': [   //wasm-fail, webgl-ok
        [[1, 224, 224, 3]], ['int32']//, ['input']
    ],
    'experts_bert_pubmed_2': [   //wasm-fail, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'movenet_singlepose_thunder_4': [   //wasm-ok, webgl-ok
        [[1, 256, 256, 3]], ['int32']//, ['input']
    ],
    'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [   //wasm-fail, webgl-ok
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'trillsson2_1': [   //wasm-fail, webgl-fail
        [[1, 128]], ['float32'] //, ['audio_samples']
    ],
    'language_model': [   //wasm-fail
        [[1, 128]], ['float32'], ['embedding_input']
    ],
    'Vfail16': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']  
    ],
    'ShuffleNetV2': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']
    ],
    'yolov5': [  // wasm-fail(Kernel 'Softplus' not registered for backend 'wasm'.), webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']
    ],
    'Xception': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'] //, ['input_3']
    ],
    'InceptionV3': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'] //, ['input_1']
    ],
    'EfficientNetV2': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']
    ],
    'efficientdet_d1_coco17_tpu-32': [  // wasm-fail(Error: Kernel 'Reciprocal' not registered for backend 'wasm'.), webgl-fail
        [[1, 640, 640, 3]], ['int32'] //, ['input_tensor']
    ],
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  // wasm-fail(Error: Kernel 'Reciprocal' not registered for backend 'wasm'.), webgl-ok
        [[1, 512, 512, 3]], ['int32'] //, ['input_tensor']  
    ],
}

let D_ort = {
    'imagenet_mobilenet_v2_100_224_classification_5': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['inputs']
    ],
    'resnet_50_classification_1': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['input_1']
    ],
    'ssd_mobilenet_v2_2': [  // wasm-ok, webgl-fail
        [[1, 224, 224, 3]], ['uint8'], ['input_tensor']
    ],
    'faster_rcnn_resnet50_v1_1024x1024_1': [  // wasm-fail, webgl-fail
        [[1, 224, 224, 3]], ['uint8'], ['input_tensor']
    ],
    'edgetpu_nlp_mobilebert-edgetpu_xs_1': [  // wasm-ok, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['float32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [  // wasm-ok, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'edgetpu_vision_deeplab-edgetpu_default_argmax_s_1': [  // wasm-ok, webgl-fail
        [[1, 512, 512, 3]], ['float32'], ['input_2']
    ],
    'edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1': [  // wasm-ok, webgl-fail
        [[1, 512, 512, 3]], ['float32'], ['input_2']
    ],
    'albert_en_base_3': [  // wasm-ok, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'electra_small_2': [  // wasm-ok, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'esrgan-tf2_1': [  // wasm-ok, webgl-fail
        [[1, 224, 224, 3]], ['float32'], ['input_0']
    ],
    'movenet_multipose_lightning_1': [  // wasm-ok, webgl-fail
        [[1, 224, 224, 3]], ['int32'], ['input']
    ],
    
    'experts_bert_pubmed_2': [  // wasm-ok, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'movenet_singlepose_thunder_4': [  // wasm-ok, webgl-fail
        [[1, 256, 256, 3]], ['int32'], ['input']
    ],
    'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [  // wasm-ok, webgl-fail
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'language_model': [  // wasm-ok, webgl-fail
        [[1, 128]], ['float32'], ['embedding_input']
    ],
    'Vfail16': [  // wasm-ok, webgl-ok(Source data too small. Allocating larger array)
        [[1, 224, 224, 3]], ['float32'], ['input_1'] 
    ],
    'ShuffleNetV2': [  // wasm-ok, webgl-fail
        [[1, 224, 224, 3]], ['float32'], ['input_5']
    ],
    'yolov5': [  // wasm-ok, webgl-fail
        [[1, 224, 224, 3]], ['float32'], ['input_1']  
    ],
    'Xception': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'], ['input_4']
    ],
    'InceptionV3': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'], ['input_2']
    ],
    'EfficientNetV2': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['input_3']
    ],
    'efficientdet_d1_coco17_tpu-32': [  // wasm-ok, webgl-fail
        [[1, 640, 640, 3]], ['uint8'], ['input_tensor'] 
    ],
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  // wasm-ok, webgl-fail
        [[1, 512, 512, 3]], ['uint8'], ['input_tensor'] 
    ],
}

function videoFPS() {
    var vid = document.querySelector("video");
    vid.play();
    var last_media_time = performance.now(), last_frame_num = 0;

    function get_fps_average() {
        return 1 / fps_rounder.reduce((a, b) => a + b) / fps_rounder.length;
    }

    function ticker(useless, metadata) {
        if (stop_flag) return;
        // https://web.dev/requestvideoframecallback-rvfc/
        var now = performance.now();
        if (now > 1000 + last_media_time) {
            var media_time_diff = now - last_media_time;
            var frame_num_diff = Math.abs(metadata.presentedFrames - last_frame_num);
            fps_rounder.push([now, frame_num_diff / media_time_diff]);
            last_media_time = now;
            last_frame_num = metadata.presentedFrames;
        }
        
        vid.requestVideoFrameCallback(ticker);
    }
    vid.requestVideoFrameCallback(ticker);
}

function MyAnimation() {    
    window.requestAnimFrame = (function(){return  window.requestAnimationFrame||window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame })();

    var useHighAnimation = false;
    var lastTime = performance.now();
    var frame = 0;
    var container, fpsDiv, fpsText, fpsGraph;

    /*var animation = function(){        
        container = document.createElement( 'div' );
        container.id = 'stats';
        container.style.cssText = 'width:80px;opacity:0.9;cursor:pointer';

        fpsDiv = document.createElement( 'div' );
        fpsDiv.id = 'fps';
        fpsDiv.style.cssText = 'padding:0 0 3px 3px;text-align:left;background-color:#002';
        container.appendChild( fpsDiv );

        fpsText = document.createElement( 'div' );
        fpsText.id = 'fpsText';
        fpsText.style.cssText = 'color:#0ff;font-family:Helvetica,Arial,sans-serif;font-size:9px;font-weight:bold;line-height:15px';
        fpsText.innerHTML = 'FPS';
        fpsDiv.appendChild( fpsText );

        fpsGraph = document.createElement( 'div' );
        fpsGraph.id = 'fpsGraph';
        fpsGraph.style.cssText = 'position:relative;width:74px;height:30px;';
        fpsDiv.appendChild( fpsGraph );
        if (useHighAnimation) {
            fpsFastGraph = document.createElement( 'div' );
            fpsFastGraph.id = 'fpsFastGraph';
            fpsFastGraph.style.cssText = 'position:relative;width:74px;height:30px; margin-top:5px';
            fpsDiv.appendChild( fpsFastGraph );
        }
        var initGraph = function (dom){
	        while ( dom.children.length < 74 ) {

		        var bar = document.createElement( 'span' );
		        bar.style.cssText = 'width:1px;height:30px;float:left;background-color:#000';
		        dom.appendChild( bar );
	        }
        };
        initGraph(fpsGraph);

        if (useHighAnimation) initGraph(fpsFastGraph);

        fpsDiv.style.display = 'block';
        container.style.position = 'absolute';
        container.style.left = '0px';
        container.style.top = '0px';
        container.id = "fpsContainer";

        container.style.position='fixed';
        container.style.zIndex='10000';


        document.body.appendChild( container );
        
    };  
      
    animation();   
      
    var updateGraph = function ( dom, value ) {
        if (!useHighAnimation) return;   
        if (value > 30)
            value = 30;     
        var child = dom.appendChild( dom.firstChild );
        child.style.height = (value) + 'px';
        child.style.marginTop = (30-value)+'px';
        child.style.backgroundColor = "#1eff1e";         
    };
    var updateGraphRed = function ( dom, value ) {
       
        if (value > 30)
            value = 30;     
        var child = dom.appendChild( dom.firstChild );
        child.style.height = (value) + 'px';
        child.style.marginTop = (30-value)+'px';
        child.style.backgroundColor = "#ff0000";           
    };*/

    var lastFameTime = performance.now();
    var fsMin = Infinity;
    var fsMax = 0;
    setTimeout(function(){loop(0)},1000);

    var loop = function(time) {
        if (stop_flag) return;
        var now =  performance.now();
        var fs = (now - lastFameTime);
        lastFameTime = now;
        var fps = Math.round(1000/fs);
        frame++;
        if (now > 1000 + lastTime){
            var fps = Math.round( ( frame * 1000 ) / ( now - lastTime ) );
            fsMin = Math.min( fsMin, fps );
            fsMax = Math.max( fsMax, fps );
            fps_arr.push([now, fps]);  

            // fpsText.textContent = fps + ' FPS (' + fsMin + '-' + fsMax + ')'; 
            // updateGraphRed( fpsGraph, Math.round(( fps / 60 ) * 30 ));          
            frame = 0;    
            lastTime = now;
            
        };
        // if (useHighAnimation)
        //     updateGraph( fpsFastGraph,  Math.round(( fps / 60 ) * 30 )  );              
        window.requestAnimFrame(loop);   
    }

      
};


async function inference_ort(provider, modelnames) {
    let ith = provider === "wasm" ? 3 : 4;
    let bknd = document.getElementById('backend');
    let mdl = document.getElementById('model');
    let stg = document.getElementById('stage');
    if (server_url === ".") {
        bknd.innerText = `BACKEND: [${ith}/4] ort-${provider}`;
    } else {
        ort.env.wasm.wasmPaths = `${server_url}/dist/`;
    }
    ort.env.wasm.numThreads = 4;
    ort.env.wasm.simd = false;
    console.log(ort.env);
    
    for (let i = 0; i < modelnames.length; i++) {
        try {
            let modelname = modelnames[i];
            console.log("ORT_INFERENCE_TIMESTAME", `${provider}:${modelname}:${Date.now()}`);
            console.log(`ORT_BEGIN_INFERENCE:${provider}`, modelname);
            console.log(`ORT_INFERENCE_BEGIN_MEMORY:${provider}`, JSON.stringify({
                "jsHeapSizeLimit": performance.memory.jsHeapSizeLimit,
                "totalJSHeapSize": performance.memory.totalJSHeapSize,
                "usedJSHeapSize": performance.memory.usedJSHeapSize
            }));
            if (server_url === ".") {
                mdl.innerText = `MODEL [${i+1}/${modelnames.length}]: ${modelname}`;
            }
            let modelSetupLatency = 0;
            let coldStartLatency = 0;
            let inferenceLatency = 0;
            let modelPath = `${server_url}/models/onnxModel/${modelname}.onnx`;
            // Model Load & Setup
            if (server_url === ".") {
                stg.innerText = `STAGE: loading and running`;
            }
            let start_t = performance.now();
            const session = await ort.InferenceSession.create(modelPath, {
                executionProviders: [provider]
            });
            console.log(`ORT_INFERENCE_SESSION_MEMORY:${provider}`, JSON.stringify({
                "jsHeapSizeLimit": performance.memory.jsHeapSizeLimit,
                "totalJSHeapSize": performance.memory.totalJSHeapSize,
                "usedJSHeapSize": performance.memory.usedJSHeapSize
            }));
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
            start_t = performance.now();
            session.startProfiling();
            await session.run(feed);
            session.endProfiling();
            coldStartLatency = performance.now() - start_t;

            // feed inputs and run
            start_t = performance.now();
            for (let epoch = 0; epoch < tolEpoch; epoch++) {
                session.startProfiling();
                _start_t = performance.now();
                await session.run(feed);
                timeInterval = performance.now() - _start_t;
                // await new Promise(r => setTimeout(r, timeInterval * 1));
                session.endProfiling();
            }
            console.log(`ORT_INFERENCE_FINISH_MEMORY:${provider}`, JSON.stringify({
                "jsHeapSizeLimit": performance.memory.jsHeapSizeLimit,
                "totalJSHeapSize": performance.memory.totalJSHeapSize,
                "usedJSHeapSize": performance.memory.usedJSHeapSize
            }));
            inferenceLatency = (performance.now() - start_t) / tolEpoch;
            if (server_url === ".") {
                stg.innerText = `STAGE: finish ${modelname}`;
            }
            console.log(`ORT_FINISH_INFERENCE:${provider}`, modelname,
                                "modelSetupLatency = ", modelSetupLatency.toFixed(1),
                                "coldStartLatency = ", coldStartLatency.toFixed(1),
                                "inferenceLatency = ", inferenceLatency.toFixed(1));
        } catch (e) {
            console.log(`failed to inference model: ${e}.`, e.stack);
        }
    }
}

async function updateProfileInfoPerModel(content, modelname) {
    for (let i = 0; i < content.activeProfile.kernels.length; i++) {
        // console.log(content.activeProfile.kernels[i].modelname);
        if (content.activeProfile.kernels[i].modelname === undefined) {
            content.activeProfile.kernels[i].modelname = modelname;
        }
        // console.log(JSON.stringify(content.activeProfile.kernels[i]));
        if (content.activeProfile.kernels[i].kernelTimeMs !== undefined) {
            content.activeProfile.kernels[i].kernelLatency = await content.activeProfile.kernels[i].kernelTimeMs;
            // content.activeProfile.kernels[i].kernelTimeMs.then(data => {
            //     content.activeProfile.kernels[i].kernelLatency = data;
            //     cnt += 1;
            // });
        } else {
            content.activeProfile.kernels[i].kernelLatency = 0;
        }
    }
}

async function uploadResult(obj, path) {
    var xhr = new XMLHttpRequest(); 
    xhr.open("post", path, true); 
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify(obj));
    console.log("finish upload.");
}


async function download(content, filename) {
    console.log("download function: ", filename);
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(content));
    var a = document.createElement('a')
    a.href = dataStr
    a.download = filename;
    a.click();
    console.log("finish download");
}

async function logState(modelname, stage, content) {
    console.log("logState: ", modelname, stage);
    for (let i = 0; i < content.activeProfile.kernels.length; i++) {
        content.activeProfile.kernels[i].kernelTimeMs.then(data => {
            console.log(
                tf.getBackend() + "\t" + 
                modelname + "\t" + 
                stage + "\t" + 
                i + "\t" + 
                "name:" + content.activeProfile.kernels[i].name + "\t" + 
                "bytesAdded:" + content.activeProfile.kernels[i].bytesAdded + "\t" + 
                "totalBytesSnapshot:" + content.activeProfile.kernels[i].totalBytesSnapshot + "\t" + 
                "kernelTimeMs:" + data
            );
        })
        
    }
    
}


async function inference_tfjs(backend, modelnames) {
    let ith = backend === "wasm" ? 1 : 2;
    let bknd = document.getElementById('backend');
    let mdl = document.getElementById('model');
    let stg = document.getElementById('stage');
    console.log("try tfjs now...");
    if (server_url === ".") {
        bknd.innerText = `BACKEND: [${ith}/4] tfjs-${backend}`;
    }
    if (backend === "cpu" || backend === "webgl" || backend === "webgpu") {
        await tf.setBackend(backend);
        await tf.ready();
    } else if (backend === "wasm") {
        console.log(server_url);
        if (server_url !== ".") {
            tf.wasm.setWasmPaths(`${server_url}/dist/`);
        }
        // tf.wasm.setThreadsCount(4);
        await tf.setBackend(backend);
        await tf.ready();
        console.log("ready now!!!");
    } else {
        throw Error("invalid backend");
    }
    console.log("tfjs", backend, "ready");
    for (let i = 0; i < modelnames.length; i++) {
        try {
            let modelname = modelnames[i];
            console.log("TFJS_INFERENCE_TIMESTAME", `${backend}:${modelname}:${Date.now()}`);
            if (server_url === ".") {
                mdl.innerText = `MODEL [${i+1}/${modelnames.length}]: ${modelname}`;
            }
            console.log("current model is: [" + modelname + "]", D_tfjs[modelname]);
            let modelSetupLatency = 0;
            let coldStartLatency = 0;
            let inferenceLatency = 0;
            // Model Load & Setup
            if (server_url === ".") {
                stg.innerText = `STAGE: loading and running`;
            }
            let start_t = performance.now();
            let model = await tf.loadGraphModel(`${server_url}/models/jsModel/${modelname}/model.json`);
            console.log(modelname, "finish load model")
            modelSetupLatency += performance.now() - start_t;
            let input_shapes = D_tfjs[modelname][0], input_types = D_tfjs[modelname][1];
            let inputData, feed = [];

            for (let i = 0; i < input_shapes.length; i++) {
                if (input_types[i] === "float32") {
                    inputData = new Float32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "int32") {
                    inputData = new Int32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "uint8") {
                    inputData = new Uint8Array(input_shapes[i].reduce((a, b) => a * b));
                }
                feed.push(tf.tensor(inputData, input_shapes[i], input_types[i]))
            }
            if (server_url === ".") {
                stg.innerText = `STAGE: warming up`;
            }
            start_t = performance.now();
            output = await model.executeAsync(feed);
            if (typeof(output.dispose) === "function") {
                output.dispose();
            }
            coldStartLatency += performance.now() - start_t;
            console.log(modelname, "finish warm up");
            // feed inputs and run
            let totalInferenceLatency = 0;
            let beginning_time = Date.now();
            for (let epoch = 0; epoch < tolEpoch; epoch++) {
                start_t = performance.now();
                output = await model.executeAsync(feed);
                if (typeof(output.dispose) === "function") {
                    output.dispose();
                }
                let timeInterval = performance.now() - start_t;
                if (server_url === ".") {
                    stg.innerText = `STAGE: infering [${epoch + 1}/${tolEpoch}] cost ${timeInterval.toFixed(3)} ms`;
                }
                totalInferenceLatency += timeInterval;
                // await new Promise(r => setTimeout(r, timeInterval * 1));
            }
            let ending_time = Date.now();
            inferenceLatency = totalInferenceLatency / tolEpoch;
            if (server_url === ".") {
                stg.innerText = `STAGE: finish ${modelname}`;
            }
            console.log("finish", modelname,
                "modelSetupLatency = ", modelSetupLatency,
                "coldStartLatency = ", coldStartLatency,
                "inferenceLatency = ", inferenceLatency);
            await updateProfileInfoPerModel(tf.ENGINE.state, modelname);
            PROFILING_RESULT[modelname] = JSON.parse(JSON.stringify(tf.ENGINE.state));
            PROFILING_RESULT[modelname]['backend'] = backend;
            PROFILING_RESULT[modelname]["num_iteration"] = tolEpoch;
            PROFILING_RESULT[modelname]["beginning_time"] = beginning_time;
            PROFILING_RESULT[modelname]["ending_time"] = ending_time;
            PROFILING_RESULT[modelname]["inference_latency"] = inferenceLatency;
            tf.ENGINE.state.resetActiveProfile();
            feed.forEach(t => t.dispose())
            model.dispose();
            await download(PROFILING_RESULT, "tfjs-wasm-profile.json");
        } catch (e) {
            console.log(`failed to inference model: ${e}.`, e.stack);
        }
    }
    tf.disposeVariables();
}

async function entry_func(url="") {
    if (url.length > 1) {
        server_url = url;
    }
    console.log(server_url);
    // MyAnimation();
    // videoFPS();
    // return;
    await inference_tfjs("wasm", [
        "imagenet_mobilenet_v2_100_224_classification_5",
        "resnet_50_classification_1",
        "ssd_mobilenet_v2_2",
        "movenet_singlepose_thunder_4",
        "Vfail16", 
        "ShuffleNetV2",
        "esrgan-tf2_1",
        "Xception",
        "InceptionV3",
        "EfficientNetV2",
    ]);
    await uploadResult(PROFILING_RESULT, "./data/tfjs/wasm-kernel");
    // console.log("TFJS_PROFILING_RESULT_WASM", JSON.stringify(PROFILING_RESULT));
    // await download(PROFILING_RESULT, "tfjs-wasm-profile.json");
    PROFILING_RESULT = {};

    await inference_tfjs("webgl", [
        "imagenet_mobilenet_v2_100_224_classification_5",
        "resnet_50_classification_1",
        "ssd_mobilenet_v2_2",
        "movenet_singlepose_thunder_4",
        "movenet_multipose_lightning_1",
        "edgetpu_nlp_mobilebert-edgetpu_xs_1",
        "mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1",
        "Vfail16", 
        "ShuffleNetV2",
        "Xception", 
        "InceptionV3",
        "yolov5",
        "EfficientNetV2",
        "efficientdet_d1_coco17_tpu-32", 
        "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8",
        "small_bert_bert_en_uncased_L-2_H-128_A-2_2",
        "esrgan-tf2_1",
    ]);
    await uploadResult(PROFILING_RESULT, "./data/tfjs/webgl-kernel");
    // console.log("TFJS_PROFILING_RESULT_WEBGL", JSON.stringify(PROFILING_RESULT));
    // await download(PROFILING_RESULT, "tfjs-webgl-profile.json");
    PROFILING_RESULT = {};

    console.log("BEGIN_INFERENCE_ORT_WASM");
    await inference_ort("wasm", [
        "imagenet_mobilenet_v2_100_224_classification_5",
        "resnet_50_classification_1",
        "ssd_mobilenet_v2_2",
        "movenet_singlepose_thunder_4",
        "movenet_multipose_lightning_1",
        "edgetpu_nlp_mobilebert-edgetpu_xs_1",
        "mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1",
        "small_bert_bert_en_uncased_L-2_H-128_A-2_2",
        "Vfail16", 
        "ShuffleNetV2",
        "Xception", 
        "InceptionV3",
        "yolov5",
        "EfficientNetV2",
        "efficientdet_d1_coco17_tpu-32", 
        "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8",
    ]);

    console.log("BEGIN_INFERENCE_ORT_WEBGL");
    await inference_ort("webgl", [
        "imagenet_mobilenet_v2_100_224_classification_5",
        "resnet_50_classification_1",
        "Vfail16", 
        "Xception", 
        "InceptionV3",
        "EfficientNetV2",
    ]);
    if (server_url === ".") {
        document.getElementById('backend').innerText = "finish, close alert box and close Chrome.";
        document.getElementById('model').innerText = "finish, close alert box and close Chrome.";
        document.getElementById('stage').innerText = "finish, close alert box and close Chrome.";
    }
    console.log("finish all");
    stop_flag = true;
    alert("finish all");
}

async function clickStart() {
    let flag = true;
    if (document.getElementById("cpu").value === "") {
        console.log('cpu error');
        flag = false;
    } else if (document.getElementById("memory").value === "") {
        console.log('memory error');
        flag = false;
    } else if (document.getElementById("igpu").value === ""  && document.getElementById("dgpu").value === "") {
        console.log('gpu error');
        flag = false;
    }
    if (!flag) {
        alert("input CPU, RAM, and GPU");
    } else {
        alert("Start");
        let obj_str = JSON.stringify({
            cpu: document.getElementById("cpu").value,
            memory: document.getElementById("memory").value,
            igpu: document.getElementById("igpu").value,
            dgpu: document.getElementById("dgpu").value,
        });
        var xhr = new XMLHttpRequest();  // XMLHttpRequest
        xhr.open("post", "./data/hardware", true);
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr.send(obj_str);
        console.log('finish');
        await entry_func();
        alert("finish");
    }
}


// if ("serviceWorker" in navigator) {
//     // Register service worker
//     navigator.serviceWorker.register(new URL("./sw.js", import.meta.url)).then(
//       function (registration) {
//         console.log("COOP/COEP Service Worker registered", registration.scope);
//         // If the registration is active, but it's not controlling the page
//         if (registration.active && !navigator.serviceWorker.controller) {
//             window.location.reload();
//         }
//       },
//       function (err) {
//         console.log("COOP/COEP Service Worker failed to register", err);
//       }
//     );
//   } else {
//     console.warn("Cannot register a service worker");
//   }


