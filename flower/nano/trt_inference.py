
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import numpy as np
from PIL import Image
import tensorrt as trt
import argparse
import time


def allocate_buffers(engine, batch_size, data_type):

   """
   This is the function to allocate buffers for input and output in the device
   Args:
      engine : The path to the TensorRT engine.
      batch_size : The batch size for execution time.
      data_type: The type of the data for input and output, for example trt.float32.

   Output:
      h_input_1: Input in the host.
      d_input_1: Input in the device.
      h_output_1: Output in the host.
      d_output_1: Output in the device.
      stream: CUDA stream.

   """

   # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
   h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
   h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
   # Allocate device memory for inputs and outputs.
   d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

   d_output = cuda.mem_alloc(h_output.nbytes)
   # Create a stream in which to copy inputs/outputs and run inference.
   stream = cuda.Stream()
   return h_input_1, d_input_1, h_output, d_output, stream

def load_images_to_buffer(pics, pagelocked_buffer):
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
   """
   This is the function to run the inference
   Args:
      engine : Path to the TensorRT engine
      pics_1 : Input images to the model.
      h_input_1: Input in the host
      d_input_1: Input in the device
      h_output_1: Output in the host
      d_output_1: Output in the device
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image

   Output:
      The list of output images

   """

   load_images_to_buffer(pics_1, h_input_1)

   with engine.create_execution_context() as context:
      # Transfer input data to the GPU.
      cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

      # Run inference.

      context.profiler = trt.Profiler()
      context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

      # Transfer predictions back from the GPU.
      cuda.memcpy_dtoh_async(h_output, d_output, stream)
      # Synchronize the stream
      stream.synchronize()
      # Return the host output.
      # out = h_output.reshape((batch_size, -1, height, width))
      out = h_output
      return out.reshape(1, -1)
def softmax(x):
  # x -= np.max(x , axis=1 , keepdims = True)
  x = np.exp(x) / np.sum(np.exp(x))
  return x

def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--engine', type=str, help='engine path')
    parser.add_argument('--source', type=str, help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[320,263], help='inference size h,w')

    opt = parser.parse_args()
    return opt

def main(opt):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    input_image_path = opt['source']
    model_path = opt['engine']
    HEIGHT , WIDTH = opt['imgsz']

    class_dic = {
        0 : "daisy",
        1 : "dandelion",
        2 : "roses",
        3 : "sunflowers",
        4 : "tulips"
    }

    img = Image.open(input_image_path).resize((WIDTH , HEIGHT))
    # img.show()
    
    start_time = time.perf_counter()

    img = np.asarray(img)
    im = np.array(img, dtype=np.float32, order='C')
    im = im.transpose((2, 0, 1))
    im /=  255

    engine = load_engine(trt_runtime, model_path)
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)


    out = do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
    
    out = softmax(out.astype(np.float128))
    end_time = time.perf_counter()
    label = np.argmax(out)
    value = np.max(out)

    print(f"inference time : {(end_time - start_time)*1000} ms")
    print(class_dic[label] , value)

if __name__ == '__main__':
   opt = parse_opt()
   main(vars(opt))