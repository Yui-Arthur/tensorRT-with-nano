# install requirement
## install pycuda (for tensorRT)
```bash
sudo su
bash script/pycuda_install.sh
```

## install torch (for pt => onnx & torch2trt )
```bash
bash script/torch_install.sh
```

## install torch2trt
```bash
bash script/torch2trt_install.sh
```

# pt => onnx => trt engine

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

# yolov8
## run ``` colab/yolov8_trt.ipynb ``` in colab & download yolov8 onnx & .yaml
## yolov8 onnx => trt engine
```bash
sudo python3 onnx2trt.py --onnx {model_path}
```

## yolov8 engine inference
```bash
# see parameter
sudo python3 yolov8_trt_inference.py --help 
# run with img
sudo python3 yolov8_trt_inference.py --weights best.engine --source dog.jpeg --imgsz 320 320 --output-shape 1 6 2100 --data ./sample.yaml
# run with video
sudo python3 yolov8_trt_inference.py --weights best.engine --source dog_cat.mp4 --imgsz 320 320 --output-shape 1 6 2100 --data ./sample.yaml
# run with webcam
```

