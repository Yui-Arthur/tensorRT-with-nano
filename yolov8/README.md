# Train yolov8 with custom datset & run yolov8 on nano with tensorRT

### dogs and cats dataset training
run ``` yolov8/colab/yolov8_with_facemask.ipynb ``` on colab & download ```best.onnx``` & ```dataset.yaml```
### facemask dataset training
run ``` yolov8/colab/yolov8_with_dogs_cats.ipynb ``` on colab & download ```best.onnx``` & ```dog_cat.yaml```

### export yolov8 model onnx => trt engine
```bash
sudo python3 onnx2trt.py --onnx {model_path}
# will show the input/output shape used in inference
# ex.
# input shape : [1, 3, 320, 320]
# output shape : (1, 6, 2100)
```

### onnxruntime inference
```bash
# see parameter
sudo python3 yolov8/nano/yolov8_onnx_inference.py --help 
# --weights is the engine path
# --imgsz is the input image w,h & define when the yolov8 training 
# --device [CPUExecutionProvider / CUDAxecutionProvider]
# --data is the yolov8 training yaml used to get class name
# --source is the inference target : 0,1,2 => webcam / .jpg .png => image / .mp4 => video
# --show show the result

#
sudo python3 yolov8/nano/yolov8_onnx_inference.py \
--weights data/models/yolov8_with_facemask_op16.onnx \
--source 0 \
--imgsz 320 320  --data yolov8/nano/dataset.yaml \
--device CUDAExecutionProvider
```
### yolov8 engine inference
```bash
# see parameter
sudo python3 yolov8/nano/yolov8_trt_inference.py --help 
# --weights is the engine path
# --imgsz is the input image w,h & define when the yolov8 training 
# --output-shape is the model output size & define when the yolov8 training
# --data is the yolov8 training yaml used to get class name
# --source is the inference target : 0,1,2 => webcam / .jpg .png => image / .mp4 => video
# --show show the result

# imgsz/output-shape can get from yolov8_trt.ipynb or  onnx2trt.py

# run with img
sudo python3 yolov8/nano/yolov8_trt_inference.py \
--weights data/models/best.engine \
--source data/images/people.jpg \
--imgsz 320 320 --output-shape 1 6 2100 \
--data yolov8/nano/sample.yaml \
--show
# run with video
sudo python3 yolov8/nano/yolov8_trt_inference.py \
--weights data/models/best.engine \
--source data/video/dog_cat.mp4 \
--imgsz 320 320 --output-shape 1 6 2100 \
--data yolov8/nano/sample.yaml \
--show
# run with webcam
sudo python3 yolov8/nano/yolov8_trt_inference.py \
--weights data/models/best.engine \
--source 0 --imgsz 320 320 --output-shape 1 6 2100 \
--data yolov8/nano/sample.yaml \
--show
```



