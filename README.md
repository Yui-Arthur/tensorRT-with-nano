# install requirement
## install pycuda (for tensorRT)
```bash
sudo su
bash script/pycuda_install.sh
```

## install torch 
```bash
bash script/torch_install.sh
```

# yolov8 in nano

## dogs and cats dataset
run ``` yolov8/colab/yolov8_with_facemask.ipynb ``` on colab & download ```best.onnx``` & ```dataset.yaml```
## facemask dataset 
run ``` yolov8/colab/yolov8_with_dogs_cats.ipynb ``` on colab & download ```best.onnx``` & ```dog_cat.yaml```

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
sudo python3 yolov8/nano/yolov8_trt_inference.py --help 
# --weights is the engine path
# --imgsz is the input image w,h & define when the yolov8 training 
# --output-shape is the model output size & define when the yolov8 training
# --data is the yolov8 training yaml used to get class name
# --source is the inference target : 0,1,2 => webcam / .jpg .png => image / .mp4 => video
# imgsz/output-shape can get in yolov8_trt.ipynb or  onnx2trt.py

# run with img
sudo python3 yolov8/nano/yolov8_trt_inference.py --weights models/best.engine --source images/dog.jpeg --imgsz 320 320 --output-shape 1 6 2100 --data yolov8/nano/sample.yaml
# run with video
sudo python3 yolov8/nano/yolov8_trt_inference.py --weights models/best.engine --source video/dog_cat.mp4 --imgsz 320 320 --output-shape 1 6 2100 --data yolov8/nano/sample.yaml
# run with webcam
sudo python3 yolov8/nano/yolov8_trt_inference.py --weights models/best.engine --source 0 --imgsz 320 320 --output-shape 1 6 2100 --data yolov8/nano/sample.yaml
```

# efficientnet_b0 with flower dataset

## run ```pytorch_with_flowers_dataset.ipynb``` on pytorch & dowload ```flower_classfication_model_best.onnx```

## onnx => trt engine
```bash
sudo python3 onnx2trt.py --onnx {model_path}
```
## inference trt engine
```bash
# see parameter
sudo python3 flower/nano/trt_inference.py --help
# example
sudo python3 flower/nano/trt_inference.py --engine models/flower_classfication_model_best.engine --source images/sunflower.jpg --imgsz 320 263 
```


