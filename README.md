# Introduce
### This project is run the model on jetson nano with tensorRT
### Lab ```flower``` & ```yolov8``` folder contain two part , one is training model on colab , another is inference on nano
### homework ```direction``` contain TODO Part and need complete by your self 

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

### trtexec tool ([official](https://developer.nvidia.com/zh-cn/blog/tensorrt-trtexec-cn/))
```bash
# export to onnx
/usr/src/tensorrt/bin/trtexec --onnx={onnx_model} --saveEngine={engine_path}
# check model performance
/usr/src/tensorrt/bin/trtexec --loadEngine={engine_path}  --warmUp=5000
```

# Classify
### Lab1 : [Flower Classfiy with TensorRT](flower/)
### Homework1 : [Direction Classfiy](homework/direction/)

# Object Detection
### Lab2 : [Yolov8 with TensorRT](yolov8/)
### Homework2 : [Transportation Detection](homework/transportation/)
<br>

### Yolov8 Inference Speed Benchmark on Nano with MAXN mode

| | ONNX RUNTIME CPU | ONNX RUNTIME CUDA | TensorRT FP32 | TensorRT FP16 |
|-| ---------------- | ----------------- | ------------- | ------------- |
|320*320 FPS|  4.60   |      23.4        |  33.8         |          36.5 |
|640*640 FPS| 1.26   |      8.56         |  11.6         |          15.5 |
