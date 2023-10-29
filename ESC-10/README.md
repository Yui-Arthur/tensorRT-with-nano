# Audio classification with tensorRT

### run ```pytorch_with_flowers_dataset.ipynb``` on pytorch & dowload ```flower_classfication_model_best.onnx```

1. run ```colab/ESC_10_train.ipynb``` train the model
2. dowload  ```bs10_l4_d2.onnx```  & upload to nano
3. export  ```bs10_l4_d2.onnx``` to tensorRT engine (use ```onnx2trt.py``` or ```trtexec```)
4. run ```nano/inference.py``` on nano 
```bash
# nano/inference.py argc
# --engine trt engine path
# --source inference wav  with sample rate 44100 HZ
# --window-size inference sliding window size (seconds)
# example
sudo python3 ESC-10/nano/inference.py \  
--engine data/models/bs10_l4_d2_op12.engine \
--source data/audio/youtube_dog.wav \ 
--window-size 3
```
