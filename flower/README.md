# train flower classify model on pytorch &  run the model on nano with tensorRT

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
sudo python3 flower/nano/trt_inference.py \
--engine data/models/flower_classfication_model_best.engine  \
--source data/images/sunflower.jpg \
--imgsz 320 263 
```
