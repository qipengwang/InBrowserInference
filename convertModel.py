import os

with open('convertModel.bat', 'w') as f:
    for model in os.listdir('models/savedModel'):
        if not os.path.isdir(f'models/savedModel/{model}'):
            continue
        os.system(f'tensorflowjs_converter --input_format=tf_hub --signature_name=serving_default models/savedModel/{model} models/jsModel/{model}')
        os.system(f'python -m tf2onnx.convert --saved-model models/savedModel/{model} --opset 15 --output models/onnxModel/{model}.onnx')