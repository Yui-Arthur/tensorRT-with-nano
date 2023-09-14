# Train yolov8 with custom datset & run yolov8 on nano with tensorRT

### dogs and cats dataset training
1. run ``` yolov8/colab/yolov8_with_facemask.ipynb ``` on colab 
2. download ```best.onnx``` & ```best_op16.onnx``` & ```dataset.yaml```
### facemask dataset training
1. run ``` yolov8/colab/yolov8_with_dogs_cats.ipynb ``` on colab 
2. download ```best.onnx``` & ```best_op16.onnx``` & ```dog_cat.yaml```

### export yolov8 model onnx => trt engine
```bash
sudo python3 onnx2trt.py --onnx {model_path}
# will show the input/output shape used in inference
# ex.
# input shape : [1, 3, 320, 320]
# output shape : (1, 6, 2100)
```

### ONNX RUNTIME inference (export need set opset=16)
```bash
# see parameter
sudo python3 yolov8/nano/yolov8_onnx_inference.py --help 
# --weights is the engine path
# --device [CPUE / CUDA]
# --data is the yolov8 training yaml used to get class name
# --source is the inference target : 0,1,2 => webcam / .jpg .png => image / .mp4 => video
# --show show the result

# inference with GPU and webcam
sudo python3 yolov8/nano/yolov8_onnx_inference.py \
--weights data/models/yolov8_with_facemask_op16.onnx \
--source 0 \
--data yolov8/nano/dataset.yaml \
--device CUDA
```
### yolov8 engine inference
```bash
# see parameter
sudo python3 yolov8/nano/yolov8_trt_inference.py --help 
# --weights is the engine path
# --data is the yolov8 training yaml used to get class name
# --source is the inference target : 0,1,2 => webcam / .jpg .png => image / .mp4 => video
# --show show the result

# run with img
sudo python3 yolov8/nano/yolov8_trt_inference.py \
--weights data/models/best.engine \
--source data/images/people.jpg \
--data yolov8/nano/sample.yaml \
--show
# run with video
sudo python3 yolov8/nano/yolov8_trt_inference.py \
--weights data/models/best.engine \
--source data/video/dog_cat.mp4 \
--data yolov8/nano/sample.yaml \
--show
# run with webcam
sudo python3 yolov8/nano/yolov8_trt_inference.py \
--weights data/models/best.engine \
--data yolov8/nano/sample.yaml \
--show
```
