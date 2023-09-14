# Introduce
### This project is run the model on jetson nano with tensorRT
### ```flower``` & ```yolov8``` folder contain two part , one is training model on colab , another is inference on nano

# nano requirement

### install pycuda
```bash
sudo su
bash script/pycuda_install.sh
```

### install torch 
```bash
bash script/torch_install.sh
```

### install onnxruntime
```bash
bash script/onnxruntime_install.sh
```

# exprot onnx => tensorRT engine

### ```onnx2trt.py``` can export your onnx model to tensorRT engine
```bash
# example
sudo python3 onnx2trt.py --onnx {model_path}
```

# trtexec tool ([official](https://developer.nvidia.com/zh-cn/blog/tensorrt-trtexec-cn/))
```bash
# export to onnx
/usr/src/tensorrt/bin/trtexec --onnx={onnx_model} --saveEngine={engine_path}
# check model performance
/usr/src/tensorrt/bin/trtexec --loadEngine={engine_path}  --warmUp=5000
```

## [Flower Classfiy with TensorRT](flower/README.md)
## [Yolov8 with TensorRT](yolov8/README.md)


## Yolov8 Inference Speed Benchmark on Nano with MAXN mode

| | ONNX RUNTIME CPU | ONNX RUNTIME CUDA | TensorRT FP32 | TensorRT FP16 |
|-| ---------------- | ----------------- | ------------- | ------------- |
|320*320 FPS|  4.6   |       23.4        |  33.8         |          36.5 |
