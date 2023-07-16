import engine as eng
import inference as inf
# import keras
import tensorrt as trt 
import numpy as np
from PIL import Image
import tensorrt as trt
# import labels  # from cityscapes evaluation script
# import skimage.transform

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)



input_file_path = "0_2_resize.png"
serialized_plan_fp32 = "resnet50.plan"
HEIGHT = 224
WIDTH = 224

img = np.asarray(Image.open(input_file_path))
im = np.array(img, dtype=np.float32, order='C')
im = im.transpose((2, 0, 1))


engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)

print(im.shape)
print(h_input.shape)

out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)

print(out.shape)