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

# exprot onnx => tensorRT engine

### ```onnx2trt.py``` can export your onnx model to tensorRT engine
```bash
# example
sudo python3 onnx2trt.py --onnx {model_path}
```
