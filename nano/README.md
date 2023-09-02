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

# yolov8
## run ``` colab/yolov8_trt.ipynb ``` in colab & download yolov8 onnx & .yaml
## yolov8 onnx => trt engine

```bash
sudo python3 onnx2trt.py --onnx {model_path}
# will show the input/output shape used in inference
# ex.
# input shape : [1, 3, 320, 320]
# output shape : (1, 6, 2100)
```


## yolov8 engine inference


```bash
# see parameter
sudo python3 yolov8_trt_inference.py --help 
# --weights is the engine path
# --imgsz is the input image w,h & define when the yolov8 training 
# --output-shape is the model output size & define when the yolov8 training
# --data is the yolov8 training yaml used to get class name
# --source is the inference target : 0,1,2 => webcam / .jpg .png => image / .mp4 => video
# imgsz/output-shape can get in yolov8_trt.ipynb or  onnx2trt.py

# run with img
sudo python3 yolov8_trt_inference.py --weights models/best.engine --source images/dog.jpeg --imgsz 320 320 --output-shape 1 6 2100 --data ./sample.yaml
# run with video
sudo python3 yolov8_trt_inference.py --weights models/best.engine --source video/dog_cat.mp4 --imgsz 320 320 --output-shape 1 6 2100 --data ./sample.yaml
# run with webcam
sudo python3 yolov8_trt_inference.py --weights models/best.engine --source 0 --imgsz 320 320 --output-shape 1 6 2100 --data ./sample.yaml
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
