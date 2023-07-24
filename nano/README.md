# install
## install pycuda
```bash
sudo su
bash pycuda_install.sh
```

## install torch
```bash
bash torch_install.sh
```

## install torch2trt
```bash
bash torch2trt_install.sh
```

# pt to onnx to trt

## gen trt engine
```bash
sudo python3 pt2onnx2trt.py
```
## inference trt engine
```bash
sudo python3 trt_inference.py
```

# torch2trt

## gen trt pt file
```bash
sudo python3 torch2trt_model.py
```

## inference trt pt file
```bash
sudo python3 torch2trt_inference.py
```
